"""
measure_ppl.py — Compute perplexity using local split model weights
====================================================================

Uses the same layer_model_*.pth files from the deployment.
Loads all layers on one machine, runs prompt+response through
the full pipeline with no compression, computes PPL on response tokens.

Usage:
    python measure_ppl.py \
        --weights_dir weights/1.1-2b-it \
        --prompts_dir results/prompts \
        --responses_dir results/csi_aware_topk_R1.0 \
        --reference_dir results/no_compression

Directory structure:
    results/prompts/prompt_1.txt, prompt_2.txt, prompt_3.txt
    results/<experiment>/response_1.txt, response_2.txt, response_3.txt
"""

import os
import sys
import json
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from tools.config import get_config_for_2b, get_config_for_7b
from tools.model import precompute_freqs_cis
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model


class FullPipelineModel:
    """
    Loads all layer weights and runs the full Gemma pipeline on one machine.
    No compression, no sockets — just sequential forward passes.
    """

    def __init__(self, weights_dir: str, variant: str = "2b", device: str = "cpu"):
        self.device = torch.device(device)
        self.config = get_config_for_2b() if "2b" in variant else get_config_for_7b()
        self.config.quant = False
        self.total_layers = self.config.num_hidden_layers  # 18 for 2B

        torch.set_default_dtype(self.config.get_dtype())

        # Load tokenizer
        from sentencepiece import SentencePieceProcessor
        tokenizer_path = os.path.join(weights_dir, "tokenizer.model")
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={self.tokenizer.GetPieceSize()}")

        # Load embedding weights
        emb_path = os.path.join(weights_dir, "embedding_weights.pth")
        self.embedding_weights = torch.load(emb_path, map_location=self.device)
        # The embedding tensor — adjust key name based on your checkpoint format
        if isinstance(self.embedding_weights, dict):
            # Try common key names
            for key in ["embedder.weight", "weight", "embedding"]:
                if key in self.embedding_weights:
                    self.embedding_weights = self.embedding_weights[key]
                    break
        self.embedding_weights = self.embedding_weights.to(self.device)
        print(f"Embedding loaded: shape={self.embedding_weights.shape}")

        # Load all layer models
        self.layers = []
        for i in range(self.total_layers):
            model_path = os.path.join(weights_dir, f"layer_model_{i}.pth")
            is_last = (i == self.total_layers - 1)
            layer = GemmaLastLayerModel(self.config) if is_last else GemmaLayerModel(self.config)
            load_model(layer, model_path)
            layer.to(self.device)
            layer.eval()
            self.layers.append(layer)
            print(f"  Layer {i}/{self.total_layers-1} loaded {'(last)' if is_last else ''}")

        # Precompute rotary embeddings
        rope_theta = getattr(self.config, 'rope_theta', 10000)
        self.freqs_cis = precompute_freqs_cis(
            self.config.head_dim,
            self.config.max_position_embeddings * 2,
            theta=rope_theta
        ).to(self.device)

        print(f"Full pipeline ready: {self.total_layers} layers on {self.device}")

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to tensor of token IDs."""
        ids = self.tokenizer.Encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def detokenize(self, ids) -> str:
        """Convert token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        return self.tokenizer.Decode(ids)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings."""
        # Normalize embeddings as Gemma does
        hidden = F.embedding(token_ids, self.embedding_weights)
        hidden = hidden * (self.config.hidden_size ** 0.5)
        return hidden

    @torch.no_grad()
    def forward_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Run full pipeline and return logits at each position.

        Args:
            token_ids: [1, seq_len] tensor of token IDs

        Returns:
            logits: [1, seq_len, vocab_size] tensor
        """
        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]

        # Embed
        hidden = self.embed(token_ids)

        # Build position indices and mask
        positions = torch.arange(seq_len, device=self.device)
        freqs = self.freqs_cis.index_select(0, positions)

        # Causal mask — sized to seq_len, not max_position_embeddings
        mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38,
                          dtype=torch.float, device=self.device)
        mask = torch.triu(mask, diagonal=1)

        # Build KV caches for each layer — sized to seq_len
        dtype = self.config.get_dtype()

        # Run through all layers
        for i, layer in enumerate(self.layers):
            kv_size = (batch_size, seq_len,
                       self.config.num_key_value_heads, self.config.head_dim)
            k_cache = torch.zeros(kv_size, dtype=dtype, device=self.device)
            v_cache = torch.zeros(kv_size, dtype=dtype, device=self.device)

            hidden = layer(
                hidden_states=hidden,
                freqs_cis=freqs,
                kv_write_indices=positions,
                kv_cache=(k_cache, v_cache),
                mask=mask,
            )

        # hidden is now [1, seq_len, hidden_size] after last layer
        # Last layer should produce logits; if not, project manually
        if hidden.shape[-1] == self.config.hidden_size:
            # Project to vocab using embedding weights (weight tying)
            logits = torch.matmul(hidden.float(), self.embedding_weights.float().T)
        else:
            # Last layer already output logits
            logits = hidden

        return logits


def compute_perplexity(model: FullPipelineModel, prompt: str, response: str) -> dict:
    """
    Compute the perplexity of `response` conditioned on `prompt`.

    Only response tokens contribute to the PPL calculation.
    """
    # Tokenize
    prompt_ids = model.tokenize(prompt)
    response_ids = model.tokenize(response)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)

    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]
    total_len = full_ids.shape[1]

    print(f"  Tokens: prompt={prompt_len}, response={response_len}, total={total_len}")

    # Forward pass to get logits
    logits = model.forward_logits(full_ids)  # [1, total_len, vocab_size]

    # Compute per-token cross-entropy on response tokens only
    # logits[t] predicts token[t+1], so for response tokens starting at prompt_len:
    # we need logits[prompt_len-1 : total_len-1] predicting tokens[prompt_len : total_len]
    shift_logits = logits[0, prompt_len - 1: total_len - 1, :]  # [response_len, vocab_size]
    shift_labels = full_ids[0, prompt_len: total_len]  # [response_len]

    # Per-token log probabilities
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    # Perplexity = exp(-mean(log_probs))
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(avg_nll)

    # Per-token detail
    token_details = []
    for t in range(response_len):
        tok_id = shift_labels[t].item()
        tok_str = model.detokenize([tok_id])
        tok_lp = token_log_probs[t].item()
        token_details.append({
            "position": t,
            "token": tok_str,
            "log_prob": round(tok_lp, 4),
        })

    return {
        "ppl": round(ppl, 4),
        "avg_nll": round(avg_nll, 4),
        "response_tokens": response_len,
        "token_log_probs": token_log_probs,  # raw tensor for utility computation
        "token_details": token_details,
    }


def compute_utility(ref_result: dict, cmp_result: dict) -> dict:
    """
    Compute aggregate utility A(eta) from reference and compressed log-probs.

    Per-token:
        d_i = logp_cmp_i - logp_ref_i
        s_i = clip(exp(d_i), 0, 1)

    Aggregate:
        A(eta) = mean_i(s_i)

    Interpretation:
        1.0 = compressed matches or beats reference
        near 0 = strong degradation

    Equivalent view: PPL_ref / PPL_cmp, capped at 1.
    """
    logp_ref = ref_result["token_log_probs"]
    logp_cmp = cmp_result["token_log_probs"]

    # Align lengths (in case responses differ slightly)
    min_len = min(len(logp_ref), len(logp_cmp))
    logp_ref = logp_ref[:min_len]
    logp_cmp = logp_cmp[:min_len]

    # d = logp_cmp - logp_ref
    d = logp_cmp - logp_ref

    # s = clip(exp(d), 0, 1)
    s = torch.clamp(torch.exp(d), 0.0, 1.0)

    # A(eta) = mean(s)
    utility = s.mean().item()

    # Per-token detail
    per_token = []
    for i in range(min_len):
        per_token.append({
            "position": i,
            "logp_ref": round(logp_ref[i].item(), 4),
            "logp_cmp": round(logp_cmp[i].item(), 4),
            "d": round(d[i].item(), 4),
            "s": round(s[i].item(), 4),
        })

    return {
        "utility": round(utility, 4),
        "tokens_compared": min_len,
        "per_token": per_token,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure perplexity offline")
    parser.add_argument("--weights_dir", type=str, default="weights/1.1-2b-it",
                        help="Path to model weights directory")
    parser.add_argument("--variant", type=str, default="2b",
                        help="Model variant: 2b or 7b")
    parser.add_argument("--responses_dir", type=str, required=True)
    parser.add_argument("--prompts_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, default=None,
                        help="No-compression responses for comparison")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_prompts", type=int, default=3)
    args = parser.parse_args()

    # Load full pipeline
    print("Loading full model pipeline...")
    model = FullPipelineModel(args.weights_dir, args.variant, args.device)

    # Compute PPL for each response
    results = []
    for i in range(1, args.num_prompts + 1):
        prompt_file = os.path.join(args.prompts_dir, f"prompt_{i}.txt")
        response_file = os.path.join(args.responses_dir, f"response_{i}.txt")

        if not os.path.exists(prompt_file) or not os.path.exists(response_file):
            print(f"  Skipping prompt {i}: file not found")
            continue

        prompt = open(prompt_file).read().strip()
        response = open(response_file).read().strip()

        print(f"\n--- Prompt {i} ---")
        print(f"  Prompt:   {prompt[:80]}...")
        print(f"  Response: {response[:80]}...")

        result = compute_perplexity(model, prompt, response)
        result["prompt_id"] = i
        results.append(result)
        print(f"  PPL = {result['ppl']:.4f}")

    # Average
    if results:
        avg_ppl = sum(r["ppl"] for r in results) / len(results)
        print(f"\n=== Average PPL: {avg_ppl:.4f} ===")
    else:
        avg_ppl = None

    # Compare against reference and compute utility
    utilities = []
    if args.reference_dir and results:
        print(f"\n--- Utility & Delta vs reference ({args.reference_dir}) ---")
        for i in range(1, args.num_prompts + 1):
            prompt_file = os.path.join(args.prompts_dir, f"prompt_{i}.txt")
            ref_file = os.path.join(args.reference_dir, f"response_{i}.txt")
            if not os.path.exists(ref_file):
                continue

            prompt = open(prompt_file).read().strip()
            ref_response = open(ref_file).read().strip()

            print(f"\n  Prompt {i} (reference):")
            ref_result = compute_perplexity(model, prompt, ref_response)

            comp_result = next((r for r in results if r["prompt_id"] == i), None)
            if comp_result:
                # Compute utility: A(eta) ∈ [0, 1]
                util = compute_utility(ref_result, comp_result)
                utilities.append(util["utility"])

                delta_ppl = comp_result["ppl"] - ref_result["ppl"]
                print(f"  ref_PPL={ref_result['ppl']:.4f}  "
                      f"comp_PPL={comp_result['ppl']:.4f}  "
                      f"delta={delta_ppl:+.4f}")
                print(f"  Utility A(eta) = {util['utility']:.4f}  "
                      f"(1.0=perfect, 0=degraded)")

        if utilities:
            avg_utility = sum(utilities) / len(utilities)
            print(f"\n=== Avg Utility A(eta): {avg_utility:.4f} ===")
    else:
        avg_utility = None

    # Save
    output = {
        "weights": args.weights_dir,
        "responses_dir": args.responses_dir,
        "reference_dir": args.reference_dir,
        "per_prompt": [],
        "avg_ppl": round(avg_ppl, 4) if avg_ppl else None,
        "avg_utility": round(avg_utility, 4) if avg_utility is not None else None,
    }
    # Strip raw tensors from results before saving, add utility per prompt
    for idx, r in enumerate(results):
        r_clean = {k: v for k, v in r.items() if k != "token_log_probs"}
        if idx < len(utilities):
            r_clean["utility"] = round(utilities[idx], 4)
        output["per_prompt"].append(r_clean)

    out_file = os.path.join(args.responses_dir, "ppl_results.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()