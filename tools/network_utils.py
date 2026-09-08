"""
network_utils.py  –  Network transport with pluggable activation compression
=============================================================================

Compression is handled by compressors.py (TopKCompressor, QuantizationCompressor).
This module handles serialisation to JSON/Base64 for socket transmission and
provides a global config knob to select the method and ratio at runtime.

Usage:
    import network_utils as nu

    # --- configure once (or update per-slot from the optimizer) ---
    nu.set_compression("topk", 0.5)           # TopK, keep 50%
    nu.set_compression("quantization", 0.25)   # INT8
    nu.set_compression("none", 1.0)            # no compression

    # --- send / receive (unchanged call sites) ---
    nu.send_data(sock, {"hidden": tensor})
    out = nu.receive_data(sock)
"""

import numpy as np
import json
import socket
import base64
import torch
import logging
import time
from typing import Any, Dict, Optional

from compressors import (
    TopKCompressor,
    QuantizationCompressor,
    LLMInt8Compressor,
    BaseCompressor,
    get_compressor,
)

# ============================================================================
# Global compression configuration
# ============================================================================

_COMPRESSION_METHOD: str = "none"        # "topk" | "quantization" | "llmint8" | "none"
_COMPRESSION_RATIO: float = 1.0          # eta in (0, 1] — real compression ratio relative to FP16 baseline
_COMPRESSION_PARAMS: Any = None          # for llmint8: [outlier_ratio, outlier_prec, regular_prec]
_COMPRESSOR: Optional[BaseCompressor] = None
_COMPRESSOR_K: float = 1.0               # internal k value passed to compressor (may differ from eta for quant)

# Default LLMInt8 configurations keyed by ratio for the simple knob interface.
# ratio -> [outlier_ratio, outlier_precision, regular_precision]
# NOW COMPUTED FROM: eta = [regular_bits*(1-p) + outlier_bits*p] / 16
# where baseline = FP16 = 16 bits, p = outlier_ratio
_LLMINT8_PRESETS = {
    0.5:   [0.01, 'fp16', 'fp16'],    # legacy, kept for backward compat
    0.25:  [0.01, 'fp16', 'int8'],
    0.125: [0.05, 'fp16', 'int4'],
    0.0625:[0.05, 'fp16', 'int2'],
}

# Bit-widths for each precision label
_PRECISION_BITS = {
    'fp16': 16,
    'int8': 8,
    'int4': 4,
    'int2': 2,
}

# LLMInt8 configurations ordered by compression aggressiveness.
# Each entry: (eta_min, eta_max, outlier_prec, regular_prec)
# eta = [regular_bits*(1-p) + outlier_bits*p] / 16
_LLMINT8_CONFIGS = [
    (0.5,   1.0,   'fp16', 'int8'),   # mild: INT8 regular, FP16 outliers
    (0.25,  0.5,   'fp16', 'int4'),   # moderate: INT4 regular, FP16 outliers
    (0.125, 0.25,  'fp16', 'int2'),   # aggressive: INT2 regular, FP16 outliers
]


def _eta_to_llmint8_params(eta: float) -> list:
    """
    Convert a continuous eta to LLMInt8 parameters [outlier_ratio, outlier_prec, regular_prec].

    eta = transmitted_data / baseline_data (FP16)
        = [regular_bits * (1 - p) + outlier_bits * p] / 16

    Solving for outlier_ratio p:
        p = (16 * eta - regular_bits) / (outlier_bits - regular_bits)

    We select the (outlier_prec, regular_prec) pair whose eta range
    contains the requested eta, then solve for p.

    Examples:
        eta=0.75 → regular=INT8, outlier=FP16, p=(12-8)/(16-8)=0.50
        eta=0.50 → regular=INT8, outlier=FP16, p=(8-8)/(16-8)=0.00
        eta=0.40 → regular=INT4, outlier=FP16, p=(6.4-4)/(16-4)=0.20
        eta=0.25 → regular=INT4, outlier=FP16, p=(4-4)/(16-4)=0.00
        eta=0.20 → regular=INT2, outlier=FP16, p=(3.2-2)/(16-2)=0.086
    """
    if eta >= 1.0 - 1e-9:
        return [0.0, 'fp16', 'fp16']  # no compression

    # Find the right config for this eta
    for eta_min, eta_max, outlier_prec, regular_prec in _LLMINT8_CONFIGS:
        if eta >= eta_min - 1e-9:
            outlier_bits = _PRECISION_BITS[outlier_prec]
            regular_bits = _PRECISION_BITS[regular_prec]

            # p = (16*eta - regular_bits) / (outlier_bits - regular_bits)
            p = (16.0 * eta - regular_bits) / (outlier_bits - regular_bits)
            p = max(0.0, min(1.0, p))  # clamp to [0, 1]

            return [round(p, 4), outlier_prec, regular_prec]

    # Below all configs: most aggressive
    return [0.0, 'fp16', 'int2']


# Supported discrete eta levels for quantization (relative to FP16 baseline)
# eta = target_bits / 16
_QUANT_DISCRETE_ETAS = sorted([1.0, 0.5, 0.25, 0.125], reverse=True)

# Mapping from real eta to compressor k (relative to FP32, since data is cast to float32)
# k_compressor = target_bits / 32 = eta * 16 / 32 = eta / 2
_QUANT_ETA_TO_K = {
    1.0:   0.5,     # FP16: 16/32 = 0.5
    0.5:   0.25,    # INT8: 8/32 = 0.25
    0.25:  0.125,   # INT4: 4/32 = 0.125
    0.125: 0.0625,  # INT2: 2/32 = 0.0625
}


def _snap_quant_eta(eta: float) -> float:
    """
    Snap a continuous eta to the nearest supported quantization level.
    Uses FP16-relative eta values.

    Mapping:
        eta >= 1.0    -> 1.0   (no compression, FP16)
        0.5 <= eta    -> 0.5   (INT8)
        0.25 <= eta   -> 0.25  (INT4)
        eta < 0.25    -> 0.125 (INT2)
    """
    for level in _QUANT_DISCRETE_ETAS:
        if eta >= level - 1e-9:
            return level
    return _QUANT_DISCRETE_ETAS[-1]


def set_compression(method: str = "topk", ratio: float = 1.0,
                    llmint8_params: Optional[list] = None) -> None:
    """
    Set the active compression method and ratio.

    Parameters
    ----------
    method : str
        One of "topk", "quantization", "llmint8", "none".
    ratio : float
        Compression parameter eta in (0, 1].
        - For topk: fraction of elements to keep (continuous).
        - For quantization: snapped to nearest discrete bit-width
          (0.5 -> FP16, 0.25 -> INT8, 0.125 -> INT4, 0.0625 -> INT2).
        - For llmint8: snapped to nearest discrete preset.
        - For "none": ignored.
    llmint8_params : list, optional
        Explicit LLMInt8 parameters: [outlier_ratio, outlier_precision, regular_precision].
        If provided, overrides the ratio-based preset.
    """
    global _COMPRESSION_METHOD, _COMPRESSION_RATIO, _COMPRESSION_PARAMS, _COMPRESSOR, _COMPRESSOR_K
    _COMPRESSION_METHOD = method.lower().strip()
    _COMPRESSION_RATIO = float(ratio)
    _COMPRESSOR_K = float(ratio)  # default: same as eta

    if _COMPRESSION_METHOD == "none":
        _COMPRESSOR = None
        _COMPRESSION_PARAMS = None
    elif _COMPRESSION_METHOD in ("quantization", "quant"):
        _COMPRESSION_METHOD = "quantization"
        _COMPRESSOR = get_compressor("quantization")
        _COMPRESSION_PARAMS = None
        # Snap eta to nearest discrete level (FP16-relative)
        _COMPRESSION_RATIO = _snap_quant_eta(_COMPRESSION_RATIO)
        # Convert real eta to compressor k (FP32-relative)
        _COMPRESSOR_K = _QUANT_ETA_TO_K.get(_COMPRESSION_RATIO, _COMPRESSION_RATIO / 2.0)
        bit_label = {1.0: 'FP16', 0.5: 'INT8', 0.25: 'INT4', 0.125: 'INT2'}.get(_COMPRESSION_RATIO, '?')
        logging.info(f"Quantization: eta={_COMPRESSION_RATIO:.4f} ({bit_label}), "
                     f"compressor_k={_COMPRESSOR_K:.4f}")
    elif _COMPRESSION_METHOD in ("llmint8", "llm_int8", "llm.int8"):
        _COMPRESSION_METHOD = "llmint8"
        _COMPRESSOR = get_compressor("llmint8")
        if llmint8_params is not None:
            # Explicit params override: use as-is
            _COMPRESSION_PARAMS = llmint8_params
        else:
            # Compute params from eta using the physical model:
            # eta = [regular_bits*(1-p) + outlier_bits*p] / 16
            _COMPRESSION_PARAMS = _eta_to_llmint8_params(_COMPRESSION_RATIO)
        logging.info(f"LLMInt8: eta={_COMPRESSION_RATIO:.4f}, "
                     f"params={_COMPRESSION_PARAMS} "
                     f"(outlier_ratio={_COMPRESSION_PARAMS[0]:.4f}, "
                     f"outlier={_COMPRESSION_PARAMS[1]}, "
                     f"regular={_COMPRESSION_PARAMS[2]})")
    else:
        # topk: continuous eta, no snapping needed. eta = k directly.
        _COMPRESSOR = get_compressor(_COMPRESSION_METHOD)
        _COMPRESSION_PARAMS = None
        _COMPRESSOR_K = _COMPRESSION_RATIO

    logging.info(f"Compression set: method={_COMPRESSION_METHOD}, ratio={_COMPRESSION_RATIO}")


def get_compression_config() -> Dict[str, Any]:
    """Return the current compression config (for logging / control headers)."""
    return {
        "method": _COMPRESSION_METHOD,
        "ratio": _COMPRESSION_RATIO,
        "params": _COMPRESSION_PARAMS,
    }


# ============================================================================
# NumPy serialiser (unchanged)
# ============================================================================

class NumPySerializer:
    """
    A utility class for serializing and deserializing NumPy arrays using Base64 encoding.
    """

    @staticmethod
    def serialize(data):
        """
        Serialize a dictionary of NumPy arrays into a JSON string, encoding arrays as Base64.

        Args:
            data (dict): A dictionary where keys are strings and values are NumPy arrays.

        Returns:
            str: A JSON string with Base64 encoded arrays.
        """
        serialized_data = {}
        for key, value in data.items():
            serialized_data[key] = base64.b64encode(value.tobytes()).decode('utf-8')
        return json.dumps(serialized_data)

    @staticmethod
    def deserialize(data):
        """
        Deserialize a JSON string back into a dictionary of NumPy arrays.

        Args:
            data (str): A JSON string with Base64 encoded arrays.

        Returns:
            dict: A dictionary with deserialized NumPy arrays.
        """
        deserialized_data = {}
        for key, value in json.loads(data).items():
            deserialized_data[key] = np.frombuffer(base64.b64decode(value), dtype=np.int64)
        return deserialized_data


# ============================================================================
# Tensor serialisation with pluggable compression
# ============================================================================

def _b64enc(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def _b64dec(s: str, dtype: np.dtype) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(s), dtype=dtype)
    return arr.copy()  # writable


class TorchTensorSerializer:
    """
    Serialize / deserialize PyTorch tensors for socket transmission.

    Compression path (compress -> base64-encode each numpy buffer -> JSON):
      1. Use the global _COMPRESSOR to compress the tensor.
      2. Walk the resulting dict: any numpy ndarray -> base64 string + dtype tag.
      3. Wrap everything in a JSON payload with a "compression_meta" header
         so the receiver knows which compressor + params to use for decompression.

    Decompression path (JSON -> base64-decode -> decompress):
      1. Parse JSON, reconstruct numpy arrays from base64.
      2. Dispatch to the correct compressor's decompress().
    """

    # ------------------------------------------------------------------ #
    #  Serialize (compress + encode)
    # ------------------------------------------------------------------ #

    @staticmethod
    def serialize(data: Dict[str, torch.Tensor]) -> str:
        method = _COMPRESSION_METHOD
        ratio = _COMPRESSION_RATIO
        compressor_k = _COMPRESSOR_K
        params = _COMPRESSION_PARAMS

        payload: Dict[str, Any] = {
            "compression_meta": {
                "method": method,
                "ratio": ratio,
                "params": params,
            },
            "tensors": {},
            "metadata": {},  # non-tensor fields like request_id
        }

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                t = value.detach().cpu()

                is_float = t.is_floating_point()
                should_compress = (is_float
                                   and method != "none"
                                   and ratio < 1.0 - 1e-9)

                if should_compress:
                    t32 = t.to(torch.float32)
                    if method == "llmint8":
                        compressed = _COMPRESSOR.compress(t32, params)
                    else:
                        # topk uses eta directly, quantization uses FP32-relative k
                        compressed = _COMPRESSOR.compress(t32, compressor_k)
                    encoded = TorchTensorSerializer._encode_compressed(compressed)
                    encoded["compressed"] = True
                    encoded["orig_dtype"] = str(t.dtype)
                    payload["tensors"][key] = encoded
                else:
                    np_dtype = torch.float32 if t.dtype == torch.bfloat16 else t.dtype
                    t_np = t.to(np_dtype).numpy()
                    payload["tensors"][key] = {
                        "compressed": False,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "data_b64": _b64enc(t_np),
                    }
            else:
                # Non-tensor data (request_id, etc.) stored as plain JSON
                payload["metadata"][key] = value

        return json.dumps(payload)

    # ------------------------------------------------------------------ #
    #  Deserialize (decode + decompress)
    # ------------------------------------------------------------------ #

    @staticmethod
    def deserialize(data: str) -> Dict[str, torch.Tensor]:
        parsed = json.loads(data)

        meta = parsed.get("compression_meta", {"method": "none", "ratio": 1.0})
        method = meta["method"]
        tensors_in = parsed.get("tensors", {})

        # Select the right decompressor
        if method == "none":
            compressor = None
        else:
            compressor = get_compressor(method)

        output: Dict[str, torch.Tensor] = {}

        for key, entry in tensors_in.items():
            if not entry.get("compressed", False):
                # Raw path: decode using the stored dtype
                torch_dtype_str = entry.get("dtype", "torch.float32")
                torch_dtype = getattr(torch, torch_dtype_str.split(".")[-1])
                # Map torch dtype to numpy dtype for decoding
                np_dtype_map = {
                    torch.float32: np.float32,
                    torch.float16: np.float16,
                    torch.int64: np.int64,
                    torch.int32: np.int32,
                    torch.int16: np.int16,
                    torch.int8: np.int8,
                    torch.bfloat16: np.float32,  # was stored as float32
                }
                np_dtype = np_dtype_map.get(torch_dtype, np.float32)
                arr = _b64dec(entry["data_b64"], np_dtype)
                t = torch.from_numpy(arr).reshape(entry["shape"])
                output[key] = t.to(torch_dtype)
            else:
                # Decode base64 back to numpy arrays
                decoded = TorchTensorSerializer._decode_compressed(entry)
                out_tensor = compressor.decompress(decoded, device="cpu")
                # Cast back to original dtype (e.g. bfloat16)
                orig_dtype_str = entry.get("orig_dtype", "torch.float32")
                orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1])
                output[key] = out_tensor.to(orig_dtype)

        return output

    # ------------------------------------------------------------------ #
    #  Helpers: encode / decode compressed dicts for JSON transport
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_compressed(d: dict) -> dict:
        """
        Recursively encode a compressed-payload dict for JSON:
          - np.ndarray  -> {"__ndarray__": True, "b64": ..., "dtype": ..., "shape": ...}
          - tuple/list  -> list (JSON-safe)
          - torch.Size  -> list
          - everything else: pass through (int, float, str, bool, None)
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = {
                    "__ndarray__": True,
                    "b64": _b64enc(v),
                    "dtype": str(v.dtype),
                    "shape": list(v.shape),
                }
            elif isinstance(v, (tuple, list, torch.Size)):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = TorchTensorSerializer._encode_compressed(v)
            else:
                # int, float, str, bool, None — JSON-native
                out[k] = v
        return out

    @staticmethod
    def _decode_compressed(d: dict) -> dict:
        """
        Reverse of _encode_compressed: reconstruct np.ndarrays and tuples.
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if v.get("__ndarray__"):
                    dtype = np.dtype(v["dtype"])
                    arr = _b64dec(v["b64"], dtype)
                    shape = tuple(v["shape"])
                    out[k] = arr.reshape(shape) if shape else arr
                else:
                    out[k] = TorchTensorSerializer._decode_compressed(v)
            elif k == "shape" and isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
        return out


# ============================================================================
# Socket send / receive
# ============================================================================

def send_data(sock: socket.socket, data: Dict[str, Any]) -> None:
    """
    Send either raw JSON (when 'batch_size' present) or compressed tensors + masks.
    """
    logging.debug("Preparing to send data...")
    if "batch_size" in data.keys():
        serialized = json.dumps(data)
    else:
        serialized = TorchTensorSerializer.serialize(data)

    logging.debug("Sending bytes...")
    sock.sendall(serialized.encode("utf-8"))
    sock.sendall(b"END")
    logging.debug("Data sent, waiting for ACK...")

    try:
        ack = sock.recv(32).decode("utf-8")
        if ack != "ACK":
            raise Exception("Did not receive acknowledgment from the receiver.")
        logging.debug("ACK received.")
    except OSError as e:
        logging.error(f"Socket error while waiting for ACK: {e}")


def receive_data(client_socket: socket.socket) -> Dict[str, Any]:
    """
    Receive serialized data from a socket connection and deserialize it.
    """
    logging.debug("Starting to receive data...")
    buffer_size = 2048
    max_buffer_size = 65536
    min_buffer_size = 512
    data = ""
    while True:
        try:
            chunk = client_socket.recv(buffer_size).decode("utf-8")
            if not chunk:
                break
            if chunk.endswith("END"):
                data += chunk[:-3]
                break
            data += chunk

            # Simple adaptive buffering
            if len(chunk) < buffer_size and buffer_size > min_buffer_size:
                buffer_size = max(buffer_size // 2, min_buffer_size)
            elif len(chunk) == buffer_size and buffer_size < max_buffer_size:
                buffer_size = min(buffer_size * 2, max_buffer_size)
        except socket.timeout as e:
            logging.error(f"Socket timeout: {e}")
            break
        except OSError as e:
            logging.error(f"Socket error: {e}")
            break

    logging.debug("Data received, sending ACK...")
    try:
        client_socket.sendall(b"ACK")
        logging.debug("ACK sent.")
    except (BrokenPipeError, OSError) as e:
        logging.error(f"Error while sending ACK: {e}")

    # Decide path based on payload shape
    try:
        # Check for probe packet (starts with PROBE marker)
        if data.startswith("PROBE"):
            logging.debug("Received probe packet, ignoring.")
            return {"__probe__": True}

        parsed = json.loads(data)
    except json.JSONDecodeError:
        logging.error("Invalid JSON received.")
        return {}

    # Batch metadata path unchanged (init messages with batch_size + request_id)
    if isinstance(parsed, dict) and "batch_size" in parsed.keys():
        return parsed

    # New compressed tensor path (with compression_meta header + metadata)
    if isinstance(parsed, dict) and "compression_meta" in parsed:
        return TorchTensorSerializer.deserialize(data)

    # Fallback: legacy format without compression_meta
    logging.warning("Received payload without compression_meta; attempting legacy decode.")
    return TorchTensorSerializer.deserialize(data)


# ============================================================================
# Connection helpers (unchanged from original)
# ============================================================================

def connect_to_next_nodes(next_layers):
    """
    Connect to the next layers in a network.

    Args:
        next_layers (list): A list of dictionaries containing host and port information of the next layers.

    Returns:
        list: A list of socket connections to the next layers, with None for failed connections.
    """
    connections = []
    for i, layer in enumerate(next_layers):
        next_layer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            next_layer_socket.connect((layer['host'], int(layer['port'])))
            logging.info(f"Successfully connected to node {next_layers[i]['layer']}: {layer['host']}:{layer['port']}")
            connections.append(next_layer_socket)
        except Exception as e:
            logging.error(f"Failed to connect to node {next_layers[i]['layer']}: {layer['host']}:{layer['port']} - Error: {e}")
            connections.append(None)
    return connections


def close_sockets(connections):
    """
    Close multiple socket connections.

    Args:
        connections (list): A list of socket connections to be closed.
    """
    for socket in connections:
        if socket is not None:
            try:
                socket.close()
            except Exception as e:
                logging.warning(f"Socket cannot be closed - Error:{e}")


def close_socket(s):
    """
    Close a single socket connection.

    Args:
        s (socket.socket): The socket object to be closed.
    """
    try:
        s.close()
    except Exception as e:
        logging.warning(f"Socket cannot be closed - Error:{e}")


def reconnect_to_node(layer, max_retries=3, time_bt_retries=1):
    """
    Attempt to reconnect to a network node multiple times.

    Args:
        layer (dict): A dictionary containing host and port information for the layer.
        max_retries (int): The maximum number of reconnection attempts.
        time_bt_retries (int): The time to wait between reconnection attempts in seconds.

    Returns:
        socket.socket: The reconnected socket object, or None if reconnection fails.
    """
    for attempt in range(max_retries):
        try:
            socket_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_conn.settimeout(3)
            socket_conn.connect((layer['host'], int(layer['port'])))
            logging.info(f"Successfully reconnected to {layer['host']}:{layer['port']}")
            return socket_conn
        except Exception as e:
            logging.error(f"Reconnection attempt {attempt + 1} failed: {e}")
            time.sleep(time_bt_retries)
    logging.error(f"Failed to reconnect to {layer['host']}:{layer['port']} after {max_retries} attempts")
    return None


def connect_to_next_layer(next_layer_info):
    """
    Connect to a single next layer node.

    Args:
        next_layer_info (dict): Dictionary containing host and port information for the next layer.

    Returns:
        socket.socket: The connected socket object, or None if connection fails.
    """
    next_layer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    next_layer_socket.settimeout(10)
    try:
        next_layer_socket.connect((next_layer_info['host'], int(next_layer_info['port'])))
        logging.info(f"Successfully connected to node {next_layer_info['layer']}: {next_layer_info['host']}:{next_layer_info['port']}")
        return next_layer_socket
    except Exception as e:
        logging.error(f"Failed to connect to node {next_layer_info['layer']}: {next_layer_info['host']}:{next_layer_info['port']} - Error: {e}")
        next_layer_info['available'] = False
    return None


def transmit_data_to_next_nodes(data, next_layers, reconnect=True):
    """
    Transmit data to the next layers in a network. Attempts to reconnect if transmission fails.

    Args:
        data (dict): The data to be transmitted.
        next_layers (list): A list of dictionaries representing the next nodes in the network.
        reconnect (bool): Whether to attempt reconnection if transmission fails.

    Returns:
        int: 0 if transmission is successful, 1 if it fails.
    """
    if not next_layers:
        raise Exception("Next layers list is empty! Cannot send to any node.")
    
    # Initialize tracking for failed ip
    failed_ips = []
    
    for i, next_layer_info in enumerate(next_layers):
        # Skip batch_size message for the first layer
        if 'batch_size' in data.keys() and next_layer_info['layer'] == 0:
            return 0

        # Skip layer if we have already tried that ip for this message
        if next_layer_info['host'] in failed_ips and i != len(next_layers) - 1:
            continue

        next_layer_socket = connect_to_next_layer(next_layer_info) if next_layer_info['available'] else None
        if next_layer_socket is not None:
            try:
                send_data(next_layer_socket, data)
                logging.debug("Successful transmission!")
                return 0
            except Exception as e:
                logging.error(f"Failed to send data to node {next_layer_info['layer']}: {e}")
                next_layer_info['available'] = False
                failed_ips.append(next_layer_info['host'])
            finally:
                close_socket(next_layer_socket)
        else:
            failed_ips.append(next_layer_info['host'])
                
        if reconnect or i == len(next_layers) - 1:
            # Attempt to reconnect to the node
            new_socket = reconnect_to_node(next_layer_info, max_retries=2, time_bt_retries=1)
            if new_socket is not None:
                next_layer_info['available'] = True
                # Proceed with data sending
                try:
                    send_data(new_socket, data)
                    logging.debug("Successful transmission!")
                    return 0
                except Exception as ex:
                    logging.error(f"Failed to send data after new connection to node {next_layer_info['layer']}: {ex}")
                    next_layer_info['available'] = False
                finally:
                    close_socket(new_socket)
            else:
                logging.error(f"Unable to send data to node {next_layer_info['layer']}.")
    logging.error("Unable to transmit data to any hop, inference process has failed.")
    return 1