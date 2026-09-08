"""
batch_inference.py — Pipeline-parallel UE for experiments
==========================================================

Replaces UX.py (Streamlit). Reads prompts from a text file, runs them
through the distributed pipeline with overlapping requests, saves responses.

Pipeline parallelism flow with 3 requests:
  1. Init all 3 requests (send KV cache setup to all nodes)
  2. Prefill all 3 requests (staggered so they overlap in the pipeline)
  3. Decode in round-robin: when request A's token returns from layer 17,
     sample next token, embed, send back to layer 1.
     Meanwhile request B's token is still flowing through middle layers.

Usage:
    python batch_inference.py \
        --host <UE_ip> --port <UE_port> \
        --next_layers <layer1_ip:port> <layer2_ip:port> ... \
        --prompts_file prompts.txt \
        --output_dir results/csi_aware_topk_R1.0 \
        --max_tokens 30
"""

import os
import sys
import time
import json
import socket
import logging
import argparse
import threading
from collections import deque

import torch
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import Sampler, Embedding, precompute_freqs_cis
from tools.tokenizer import Tokenizer
from tools.model_utils import GemmaLayerModel, load_model
from tools.network_utils import send_data, receive_data, set_compression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [UE] %(message)s")


class PipelineUE:
    """
    User Equipment node (layer 0) that manages multiple concurrent requests.
    Runs layer 0's model, embedding, and sampling.
    """

    def __init__(self, layer_info, next_layers, config, device,
                 model, embedder, sampler, tokenizer, freqs_cis):
        self.layer_info = layer_info
        self.next_layers = next_layers
        self.config = config
        self.device = device
        self.model = model
        self.embedder = embedder
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.freqs_cis = freqs_cis
        self.total_layers = config.num_hidden_layers

        # Per-request state
        self.requests = {}  # request_id -> RequestState

        # Server socket to receive results from last layer
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((layer_info['host'], layer_info['port']))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)
        logging.info(f"UE listening on {layer_info['host']}:{layer_info['port']}")

        # KV cache for layer 0
        self.kv_caches = {}  # request_id -> (k_cache, v_cache)
        self.mask_tensors = {}  # request_id -> mask_tensor

    def _send_to_next(self, data):
        """Send data to layer 1 (first pipeline node after UE)."""
        from tools.network_utils import transmit_data_to_next_nodes
        transmit_data_to_next_nodes(data, self.next_layers)

    def init_request(self, request_id, prompt, max_tokens):
        """Initialize a request: tokenize, setup KV caches, send init to pipeline."""
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        max_seq_len = len(tokens) + max_tokens

        # Send init message to all pipeline nodes
        init_msg = {
            'batch_size': 1,
            'max_seq_len': max_seq_len,
            'request_id': request_id,
        }
        self._send_to_next(init_msg)
        time.sleep(0.2)  # let init propagate

        # Setup local KV cache for layer 0
        dtype = self.config.get_dtype()
        size = (1, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
        k_cache = torch.zeros(size=size, dtype=dtype, device=self.device)
        v_cache = torch.zeros(size=size, dtype=dtype, device=self.device)
        self.kv_caches[request_id] = (k_cache, v_cache)

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(self.device)
        self.mask_tensors[request_id] = mask_tensor

        # Store request state
        self.requests[request_id] = {
            'prompt_tokens': tokens,
            'generated_tokens': [],
            'max_tokens': max_tokens,
            'next_pos': 0,
            'done': False,
            'prompt': prompt,
        }

        logging.info(f"[{request_id}] Initialized: {len(tokens)} prompt tokens, "
                     f"max_seq={max_seq_len}")

    def prefill_request(self, request_id):
        """Run prefill: embed all prompt tokens, run layer 0, send to layer 1."""
        state = self.requests[request_id]
        tokens = state['prompt_tokens']
        seq_len = len(tokens)

        # Embed
        token_tensor = torch.tensor([tokens], dtype=torch.int64, device=self.device)
        hidden = self.embedder(token_tensor)
        hidden = hidden * (self.config.hidden_size ** 0.5)  # Gemma scaling

        # Positions
        positions = torch.arange(seq_len, device=self.device)
        freqs = self.freqs_cis.index_select(0, positions)
        mask = self.mask_tensors[request_id].index_select(2, positions)

        # Run layer 0
        hidden = self.model(
            hidden_states=hidden,
            freqs_cis=freqs,
            kv_write_indices=positions,
            kv_cache=self.kv_caches[request_id],
            mask=mask,
        )

        # Send to layer 1
        data = {
            'hidden_state': hidden,
            'KV_index': positions,
            'request_id': torch.tensor([int(request_id.split('_')[1])], dtype=torch.int64),
        }
        self._send_to_next(data)

        state['next_pos'] = seq_len
        logging.info(f"[{request_id}] Prefill sent ({seq_len} tokens)")

    def process_returned_token(self, received_data):
        """
        Process a token that completed the full pipeline (returned from layer 17).
        Sample next token, embed, run layer 0, send back to layer 1.
        Returns (request_id, generated_token_str, is_done).
        """
        # Extract request_id (comes back as tensor from pipeline)
        rid_raw = received_data.get('request_id', 0)
        if isinstance(rid_raw, torch.Tensor):
            request_id = f"req_{rid_raw.item()}"
        else:
            request_id = str(rid_raw)
        state = self.requests.get(request_id)
        if state is None or state['done']:
            return request_id, None, True

        hidden = received_data['hidden_state'].to(self.device)

        # Sample next token using the Sampler (same as stream_generation)
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))

        # During prefill return: select last position
        # During decode return: hidden is already single-token, select position 0
        if hidden.shape[1] > 1:
            # Prefill: sample from the last token position
            output_positions = torch.LongTensor([hidden.shape[1] - 1]).to(self.device)
        else:
            # Decode: single token
            output_positions = torch.tensor(0, dtype=torch.int64, device=self.device)

        temperatures = torch.FloatTensor([0.95]).to(self.device)
        top_ps = torch.FloatTensor([1.0]).to(self.device)
        top_ks = torch.LongTensor([100]).to(self.device)

        next_token_ids = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        next_token = next_token_ids.squeeze().item()

        # Check for EOS or max tokens
        token_str = self.tokenizer.decode([next_token])
        state['generated_tokens'].append(next_token)

        eos_tokens = [self.tokenizer.eos_id] if hasattr(self.tokenizer, 'eos_id') else [1]
        if next_token in eos_tokens or len(state['generated_tokens']) >= state['max_tokens']:
            state['done'] = True
            logging.info(f"[{request_id}] Done: {len(state['generated_tokens'])} tokens generated")
            return request_id, token_str, True

        # Embed the new token, run layer 0, send to layer 1
        token_tensor = torch.tensor([[next_token]], dtype=torch.int64, device=self.device)
        hidden_new = self.embedder(token_tensor)
        hidden_new = hidden_new * (self.config.hidden_size ** 0.5)  # Gemma scaling

        pos = state['next_pos']
        positions = torch.tensor([pos], device=self.device)
        freqs = self.freqs_cis.index_select(0, positions)
        mask = self.mask_tensors[request_id].index_select(2, positions)

        hidden_new = self.model(
            hidden_states=hidden_new,
            freqs_cis=freqs,
            kv_write_indices=positions,
            kv_cache=self.kv_caches[request_id],
            mask=mask,
        )

        data = {
            'hidden_state': hidden_new,
            'KV_index': positions,
            'request_id': torch.tensor([int(request_id.split('_')[1])], dtype=torch.int64),
        }
        self._send_to_next(data)

        state['next_pos'] = pos + 1
        return request_id, token_str, False

    def receive_one(self):
        """Wait for one result from the last layer."""
        try:
            client_socket, _ = self.server_socket.accept()
            data = receive_data(client_socket)
            client_socket.close()
            return data
        except socket.timeout:
            return None

    def run_pipeline(self, prompts, max_tokens, output_dir, stagger_delay=0.6, init_wait=15.0):
        """
        Run all prompts through the pipeline with overlapping requests.

        stagger_delay: seconds between starting consecutive prefills.
        init_wait: seconds to wait after all inits before starting prefills,
                   to ensure KV caches are set up at every layer.
        """
        os.makedirs(output_dir, exist_ok=True)
        num_prompts = len(prompts)

        # Phase 1: Initialize all requests
        logging.info(f"=== Initializing {num_prompts} requests ===")
        for idx, prompt in enumerate(prompts):
            rid = f"req_{idx}"
            self.init_request(rid, prompt, max_tokens)

        # Phase 1.5: Wait for inits to propagate through all pipeline layers
        logging.info(f"=== Waiting {init_wait}s for inits to propagate through all {self.total_layers} layers ===")
        time.sleep(init_wait)

        # Phase 2: Staggered prefill
        logging.info(f"=== Starting staggered prefill (delay={stagger_delay}s) ===")
        for idx in range(num_prompts):
            rid = f"req_{idx}"
            self.prefill_request(rid)
            if idx < num_prompts - 1:
                time.sleep(stagger_delay)

        # Phase 3: Decode loop — process returned tokens until all done
        logging.info(f"=== Decode phase ===")
        active = {f"req_{i}" for i in range(num_prompts)}
        token_times = {f"req_{i}": [] for i in range(num_prompts)}

        while active:
            data = self.receive_one()
            if data is None:
                continue
            if 'batch_size' in data:
                continue  # init echo, skip
            if 'hidden_state' not in data:
                continue

            t_recv = time.time()
            rid, token_str, is_done = self.process_returned_token(data)
            t_sent = time.time()

            if token_str is not None:
                token_times[rid].append({
                    'token': token_str,
                    'recv_time': t_recv,
                    'sent_time': t_sent,
                })
                logging.info(f"[{rid}] token {len(self.requests[rid]['generated_tokens'])}: "
                             f"'{token_str}' ({t_sent - t_recv:.4f}s layer0)")

            if is_done and rid in active:
                active.discard(rid)
                logging.info(f"[{rid}] Complete. {len(active)} requests remaining.")

        # Phase 4: Save results
        logging.info(f"=== All requests complete. Saving results. ===")
        for idx, prompt in enumerate(prompts):
            rid = f"req_{idx}"
            state = self.requests[rid]
            response_text = self.tokenizer.decode(state['generated_tokens'])

            # Save prompt
            with open(os.path.join(output_dir, f"prompt_{idx+1}.txt"), "w") as f:
                f.write(prompt)

            # Save response
            with open(os.path.join(output_dir, f"response_{idx+1}.txt"), "w") as f:
                f.write(response_text)

            # Save timing
            with open(os.path.join(output_dir, f"timing_{idx+1}.json"), "w") as f:
                json.dump({
                    'request_id': rid,
                    'prompt_tokens': len(state['prompt_tokens']),
                    'generated_tokens': len(state['generated_tokens']),
                    'token_times': token_times[rid],
                }, f, indent=2)

            logging.info(f"[{rid}] Response: {response_text[:80]}...")

    def close(self):
        self.server_socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline-parallel batch inference UE")
    parser.add_argument("--host", type=str, required=True, help="UE host IP")
    parser.add_argument("--port", type=int, required=True, help="UE listen port")
    parser.add_argument("--next_layers", type=str, nargs='+', required=True,
                        help="Layer 1-5 host:port pairs")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Text file with one prompt per line")
    parser.add_argument("--output_dir", type=str, default="results/experiment",
                        help="Directory to save responses and timing")
    parser.add_argument("--max_tokens", type=int, default=30,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--stagger_delay", type=float, default=0.6,
                        help="Seconds between prefills (~max tau_i)")
    parser.add_argument("--init_wait", type=float, default=15.0,
                        help="Seconds to wait after sending all inits before prefill "
                             "(must be long enough for inits to propagate through all layers)")
    parser.add_argument("--compression_method", type=str, default="none")
    parser.add_argument("--compression_ratio", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Compression (fallback, controller overrides)
    set_compression(args.compression_method, args.compression_ratio)

    # Model config
    VARIANT = '1.1-2b-it'
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.tokenizer = f'./weights/{VARIANT}/tokenizer.model'
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    device = torch.device('cpu')

    # Load layer 0 model
    logging.info("Loading layer 0 model...")
    layer0_model = GemmaLayerModel(config)
    load_model(layer0_model, f'./weights/{VARIANT}/layer_model_0.pth')
    layer0_model.to(device)
    layer0_model.eval()

    # Load embedder, sampler, tokenizer
    logging.info("Loading embedder, sampler, tokenizer...")
    embedder = Embedding(config.vocab_size, config.hidden_size, config.quant)
    load_model(embedder, f'./weights/{VARIANT}/embedding_weights.pth')
    embedder.to(device)
    embedder.eval()

    sampler = Sampler(config.vocab_size).to(device)
    tokenizer = Tokenizer(config.tokenizer)

    rope_theta = getattr(config, 'rope_theta', 10000)
    freqs_cis = precompute_freqs_cis(
        config.head_dim, config.max_position_embeddings * 2, theta=rope_theta
    ).to(device)

    # Build next_layers info
    total_layers = config.num_hidden_layers
    next_layers = [
        {'host': l.split(':')[0], 'port': l.split(':')[1],
         'layer': idx + 1, 'available': True}
        for idx, l in enumerate(args.next_layers)
    ]

    layer_info = {'host': args.host, 'port': args.port}

    # Load prompts
    USER_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'
    with open(args.prompts_file) as f:
        raw_prompts = [line.strip() for line in f if line.strip()]
    prompts = [USER_TEMPLATE.format(prompt=p) for p in raw_prompts]
    logging.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Create UE and run
    ue = PipelineUE(layer_info, next_layers, config, device,
                    layer0_model, embedder, sampler, tokenizer, freqs_cis)

    try:
        ue.run_pipeline(prompts, args.max_tokens, args.output_dir,
                        args.stagger_delay, args.init_wait)
    finally:
        ue.close()
        logging.info("Done.")


if __name__ == "__main__":
    main()