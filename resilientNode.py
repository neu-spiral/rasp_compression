import os
import sys
import torch
import argparse
import socket
import signal
import time
import json
import logging
import threading
from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import precompute_freqs_cis
from tools.network_utils import receive_data, transmit_data_to_next_nodes, set_compression
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model
from tools.metrics import NodeMetrics, LastNodeMetrics, LinkProber


# ============================================================================
# Directive Listener — receives per-node eta updates from the controller
# ============================================================================

class DirectiveListener:
    """
    Listens on a UDP port for ControlDirective messages from the central
    controller.  Updates the node's compression config when a new directive
    arrives.

    Message format (JSON): {"eta_i": float, "codec": str}
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(1.0)
        self._running = False
        self._thread = None

        # Current directive (read by the inference loop)
        self.current_eta: float = 1.0
        self.current_codec: str = "none"
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logging.info(f"Directive listener on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._sock.close()

    def _listen_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                eta_i = msg.get("eta_i", 1.0)
                codec = msg.get("codec", "none")
                codec_params = msg.get("codec_params", None)

                with self._lock:
                    self.current_eta = eta_i
                    self.current_codec = codec

                # Apply immediately
                set_compression(codec, eta_i, llmint8_params=codec_params)
                logging.info(f"Directive applied: codec={codec}, eta={eta_i:.4f}")

            except socket.timeout:
                continue
            except Exception as e:
                logging.warning(f"Directive parse error: {e}")

    def get_current(self):
        with self._lock:
            return self.current_eta, self.current_codec


# ============================================================================
# Logging and LED helpers (unchanged)
# ============================================================================

def setup_logging(layer):
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'layer_{layer}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for layer {layer}")


def control_led(led_name, state):
    path = f'/sys/class/leds/{led_name}/brightness'
    if state == "on":
        os.system(f'echo 1 | sudo tee {path} > /dev/null')
    elif state == "off":
        os.system(f'echo 0 | sudo tee {path} > /dev/null')
    else:
        logging.error("Invalid state. Use 'on' or 'off'.")


# ============================================================================
# Model init (unchanged)
# ============================================================================

def initialize_model(model_path, is_last_layer, config, device):
    layer_model = GemmaLastLayerModel(config) if is_last_layer else GemmaLayerModel(config)
    load_model(layer_model, model_path)
    layer_model.to(device)
    layer_model.eval()
    rope_theta = getattr(config, 'rope_theta', 10000)
    prec_freqs_cis = precompute_freqs_cis(config.head_dim,
                                    config.max_position_embeddings * 2,
                                    theta=rope_theta).to(device)
    if is_last_layer:
        logging.info(f"Last layer initialized")
    else:
        logging.info(f"Layer initialized")
    return layer_model, prec_freqs_cis


def initialize_gen_aux(mask_tensor, kv_caches, data, config, device):
    """Legacy single-request init — used when no request_id is present."""
    batch_size = data['batch_size']
    max_seq_len = data['max_seq_len']
    kv_caches = []
    size = (batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
    dtype = config.get_dtype()
    k_cache = torch.zeros(size=size, dtype=dtype, device=device)
    v_cache = torch.zeros(size=size, dtype=dtype, device=device)
    kv_caches.append((k_cache, v_cache))
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    logging.info(f"Auxiliar generation structures initialized")
    return mask_tensor, kv_caches


class RequestStateManager:
    """
    Manages KV caches and mask tensors per request_id.
    Supports multiple concurrent requests flowing through the pipeline.
    """

    def __init__(self, config, device, max_requests=16):
        self.config = config
        self.device = device
        self.max_requests = max_requests
        self.states = {}       # request_id -> (mask_tensor, kv_caches)
        self.request_order = []

    def init_request(self, request_id, batch_size, max_seq_len):
        if len(self.states) >= self.max_requests:
            oldest = self.request_order.pop(0)
            if oldest in self.states:
                del self.states[oldest]
                logging.info(f"Evicted state for request {oldest}")

        dtype = self.config.get_dtype()
        size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                self.config.head_dim)
        k_cache = torch.zeros(size=size, dtype=dtype, device=self.device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=self.device)
        kv_caches = [(k_cache, v_cache)]

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(self.device)

        self.states[request_id] = (mask_tensor, kv_caches)
        self.request_order.append(request_id)
        logging.info(f"Initialized state for request {request_id}")

    def get_state(self, request_id):
        return self.states.get(request_id, (None, None))

    def has_request(self, request_id):
        return request_id in self.states


# ============================================================================
# handle_client — now instrumented with metrics
# ============================================================================

def handle_client(client_socket, model, prec_freqs_cis, request_mgr,
                  device, next_layers, is_last_layer, k, metrics,
                  config, reconnection_freq=10):
    """
    Handle a client connection with per-request KV caches and metrics.
    Supports pipeline parallelism: multiple requests interleave at each node.
    """
    try:
        metrics.mark_arrival()
        received_data = receive_data(client_socket)
        control_led("PWR", "on")

        # Safety check: empty or malformed data
        if not received_data:
            logging.warning("Received empty data, skipping")
            return 1

        # Skip probe packets
        if received_data.get("__probe__"):
            return "probe"

        # Extract request_id (tensor or string)
        rid_raw = received_data.get('request_id', None)
        if isinstance(rid_raw, torch.Tensor):
            request_id = f"req_{rid_raw.item()}"
            rid_tensor = rid_raw  # keep original tensor for forwarding
        elif rid_raw is not None:
            request_id = str(rid_raw)
            rid_tensor = rid_raw
        else:
            request_id = "req_0"
            rid_tensor = torch.tensor([0], dtype=torch.int64)

        if 'batch_size' in received_data:
            # Init message for a new request
            logging.info(f"Init request {request_id}")
            request_mgr.init_request(
                request_id,
                received_data['batch_size'],
                received_data['max_seq_len']
            )
            control_led("PWR", "off")
            if not is_last_layer:
                transmit_data_to_next_nodes(received_data, next_layers)
            return "init"

        else:
            # Forward pass for existing request
            mask_tensor, kv_caches = request_mgr.get_state(request_id)
            if mask_tensor is None:
                logging.error(f"NO STATE for request {request_id}! "
                              f"Init not received yet. Known requests: "
                              f"{list(request_mgr.states.keys())}. Dropping data.")
                return 1

            logging.info(f"Forward pass for request {request_id}")

            # Verify required keys
            if 'hidden_state' not in received_data or 'KV_index' not in received_data:
                logging.error(f"Missing keys in data for request {request_id}. "
                              f"Keys received: {list(received_data.keys())}. Dropping.")
                return 1

            input_positions = received_data['KV_index'].to(device)
            freqs_cis = prec_freqs_cis.index_select(0, input_positions)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions)

            metrics.mark_compute_start()

            hidden_states = model(
                hidden_states=received_data['hidden_state'].to(device),
                freqs_cis=freqs_cis,
                kv_write_indices=input_positions,
                kv_cache=kv_caches[0],
                mask=curr_mask_tensor,
            )

            metrics.mark_compute_end()

            # Carry request_id forward through the pipeline
            data = {
                'hidden_state': hidden_states,
                'KV_index': input_positions,
                'request_id': rid_tensor,
            }

            if is_last_layer:
                control_led("PWR", "off")
                metrics.mark_tx_end(tx_bytes=0, raw_bytes=0)
                metrics.report()
                exit_status = transmit_data_to_next_nodes(data, next_layers,
                                                          k % reconnection_freq == 0)
            else:
                raw_bytes = hidden_states.numel() * hidden_states.element_size()
                metrics.mark_tx_start()
                control_led("PWR", "off")
                exit_status = transmit_data_to_next_nodes(data, next_layers,
                                                          k % reconnection_freq == 0)
                from tools.network_utils import get_compression_config
                comp_cfg = get_compression_config()
                tx_bytes_est = int(raw_bytes * comp_cfg["ratio"])
                metrics.mark_tx_end(tx_bytes=tx_bytes_est, raw_bytes=raw_bytes)
                metrics.report()

    except Exception as e:
        logging.error(f"Unexpected error in handle_client: {e}")
        return 1
    finally:
        client_socket.close()

    return 0


# ============================================================================
# Server (now with metrics + directive listener)
# ============================================================================

def start_server(layer_info, next_layers, is_last_layer, metrics, config, device):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((layer_info['host'], layer_info['port']))
    server_socket.listen(5)  # Allow queued connections for pipeline parallelism
    logging.info("Server listening on {}:{}...".format(
        layer_info['host'], layer_info['port']))

    model, prec_freqs_cis = initialize_model(MODEL_PATH, is_last_layer, config, device)

    # Per-request state manager (replaces single global mask/kv_caches)
    request_mgr = RequestStateManager(config, device, max_requests=16)

    k = 0
    logging.info("Model ready for inference (pipeline-parallel mode)")

    def signal_handler(sig, frame):
        logging.info('Closing server...')
        metrics.close()
        server_socket.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            client_socket, client_addr = server_socket.accept()
            accept_time = time.perf_counter()
            logging.debug("Accepted connection from {}:{} at time {}".format(
                client_addr[0], client_addr[1], accept_time))

            response = handle_client(
                client_socket, model, prec_freqs_cis, request_mgr,
                device, next_layers, is_last_layer, k, metrics, config)

            if response == "init":
                logging.info("Request initialized")
            elif response == "probe":
                pass  # silently handled
            else:
                logging.info("Forward pass {}: {:.6f}\n".format(
                    k, time.perf_counter() - accept_time))
            k += 1
    finally:
        metrics.close()
        server_socket.close()


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Introduce host and port of previous and next node.")
    parser.add_argument("--host", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=12647)
    parser.add_argument("--next_layers", type=str, nargs='+', required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--compression_method", type=str, default="none")
    parser.add_argument("--compression_ratio", type=float, default=1.0)
    parser.add_argument("--llmint8_outlier_ratio", type=float, default=0.01,
                        help="LLMInt8: fraction of elements treated as outliers")
    parser.add_argument("--llmint8_outlier_prec", type=str, default="fp16",
                        help="LLMInt8: outlier precision (fp16 or int8)")
    parser.add_argument("--llmint8_regular_prec", type=str, default="int8",
                        help="LLMInt8: regular precision (fp16, int8, int4, int2)")
    # Controller connection
    parser.add_argument("--controller_host", type=str, default="",
                        help="Central controller IP (empty = no metrics)")
    parser.add_argument("--controller_port", type=int, default=9999,
                        help="UDP port for sending metrics to controller")
    parser.add_argument("--control_base_port", type=int, default=10000,
                        help="Base port for receiving directives (listens on base+layer)")
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    try:
        args = parse_args()
        i = args.layer
        setup_logging(i)

        # Compression parameters from command line (static fallback / llmint8 config).
        # NOTE: set_compression() is deferred until the topology is known, so we can
        # gate it on machine cut-points (see below, after next_layers is built).
        llmint8_params = None
        if args.compression_method.lower() in ("llmint8", "llm_int8", "llm.int8"):
            llmint8_params = [
                args.llmint8_outlier_ratio,
                args.llmint8_outlier_prec,
                args.llmint8_regular_prec,
            ]

        control_led("ACT", "on")
        command_used = ' '.join(sys.argv)
        logging.info(f"Command used: {command_used}")

        # Model config
        VARIANT = '1.1-2b-it'
        MACHINE_TYPE = 'cuda'
        MODEL_PATH = './weights/{}/layer_model_{}.pth'.format(VARIANT, i)

        config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
        config.quant = 'quant' in VARIANT
        total_layers = config.num_hidden_layers

        layer_info = {'host': args.host, 'port': args.port}
        next_layers = [
            {'host': layer.split(':')[0], 'port': layer.split(':')[1],
             'layer': (i + layer_idx + 1) % total_layers, 'available': True}
            for layer_idx, layer in enumerate(args.next_layers)
        ]

        exp_layers = 5 if total_layers - i > 5 else total_layers - i
        assert len(next_layers) == exp_layers

        # ---- Machine cut-point detection ----
        # Compress only where the activation crosses to a DIFFERENT machine, and
        # NOT on the return hop to the UE/laptop (primary next layer == 0). This
        # compresses only on the last layer of each *pipeline* machine, matching
        # the controller's per-machine link model (6 inter-Pi links here).
        primary_next = next_layers[0]
        is_cutpoint = (primary_next['host'] != args.host
                       and int(primary_next['layer']) != 0)
        if is_cutpoint:
            set_compression(args.compression_method, args.compression_ratio,
                            llmint8_params=llmint8_params)
            logging.info(f"Layer {i}: MACHINE CUT-POINT -> next host {primary_next['host']} "
                         f"(layer {primary_next['layer']}); compression active "
                         f"(method={args.compression_method}, ratio={args.compression_ratio})")
        else:
            set_compression("none", 1.0)
            reason = ("return to UE/laptop" if int(primary_next['layer']) == 0
                      else "intra-machine hop (next layer on same host)")
            logging.info(f"Layer {i}: {reason} -> compression DISABLED, activation sent raw")

        torch.set_default_dtype(config.get_dtype())
        device = torch.device(
            'cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda'
            else 'cpu')
        is_last_layer = i == total_layers - 1

        # ---- METRICS & DIRECTIVE LISTENER ----
        if args.controller_host:
            # All nodes report metrics (tau / throughput) so the controller sees the
            # whole pipeline; only cut-points carry a compression link.
            if is_last_layer:
                metrics = LastNodeMetrics(i, args.controller_host, args.controller_port)
            else:
                metrics = NodeMetrics(i, args.controller_host, args.controller_port)

            # Link prober + directive listener ONLY on machine cut-points — those are
            # the real inter-machine links the controller optimizes. Non-cut-point
            # layers stay uncompressed and their (localhost) link is not modeled.
            if is_cutpoint:
                prober = LinkProber(
                    primary_next['host'], int(primary_next['port']),
                    probe_interval=2.0)
                prober.start()
                metrics.prober = prober
                logging.info(f"Link prober started: -> {primary_next['host']}:{primary_next['port']}")

                directive = DirectiveListener(args.host, args.control_base_port + i)
                directive.start()
                logging.info(f"Connected to controller at {args.controller_host}:"
                             f"{args.controller_port} (cut-point: applies compression directives)")
            else:
                logging.info(f"Layer {i}: not a compression cut-point — reporting metrics "
                             f"only, no prober/directive listener, stays uncompressed")
        else:
            # No controller — use static compression from command line
            metrics = NodeMetrics(i, "127.0.0.1", 0)  # dummy, won't send
            metrics.report = lambda: None  # no-op
            logging.info("No controller configured, running with static compression")

        start_server(layer_info, next_layers, is_last_layer, metrics, config, device)

    finally:
        control_led("PWR", "off")
        control_led("ACT", "off")