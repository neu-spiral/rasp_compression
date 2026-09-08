"""
train_acc_model.py — Fit the accuracy surrogate A(eta) from an OFFLINE prompt set
==================================================================================

Instead of hand-written (X, y) arrays, this builds the training set by actually
running an offline prompt corpus (``prompts_offline.txt``) through the full Gemma
pipeline at many per-link compression configurations and measuring the resulting
utility A(eta) ∈ [0, 1] against an uncompressed reference — the same
``compute_utility`` used by measure_ppl.py.

Pipeline per config eta:
  1. (once) generate a reference response per prompt with NO compression, and its
     per-token reference log-probs.
  2. for each eta vector, teacher-force that same reference response through the
     model with compression applied at each inter-node link (exactly the
     deployment codec via tools/network_utils), and read the compressed log-probs.
  3. utility_prompt = mean_i clip(exp(logp_cmp_i - logp_ref_i), 0, 1)
     y(eta)         = mean over prompts;   X row = eta vector.
Then fit SurrogateAccuracyModel and save to outputs/accuracy_model.pkl.

The eta vector has one entry PER MACHINE CUT-POINT (not per layer). Cut-points are
derived from --nodes exactly as in controller.py: the last layer of each pipeline
machine whose output crosses to a different machine, excluding the return hop to
the UE. eta[j] compresses the output of cutpoint_layers[j]. For scripts/nodes.txt
this is 6 links (layers 3,5,8,10,12,15).

IMPORTANT: --nodes and --method must match your no_csi deployment, since the
controller calls model.predict(eta) with a vector of that length (= number of
cut-points), and A(eta) depends on the codec.

Usage:
    python train_acc_model.py --weights_dir weights/1.1-2b-it \
        --nodes scripts/nodes.txt --prompts_file prompts_offline.txt --method topk \
        --output outputs/accuracy_model.pkl
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tools"))  # so network_utils' `from compressors import` resolves

from measure_ppl import FullPipelineModel, compute_perplexity, compute_utility
from accuracy_model import SurrogateAccuracyModel
import tools.network_utils as nu

# Same chat template the deployment / batch_inference uses.
USER_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'


class CompressedPipelineModel(FullPipelineModel):
    """FullPipelineModel that applies the deployment codec at MACHINE CUT-POINTS only.

    Compression is a round-trip through tools/network_utils (serialize+deserialize),
    identical to what happens on the wire between nodes, so the accuracy effect
    matches the real system. Compression is applied only at the cut-point layers
    (the last layer of each pipeline machine, from nodes.txt): eta[j] compresses
    the output of cutpoint_layers[j], matching the controller's per-machine links.
    """

    def __init__(self, weights_dir, variant="2b", device="cpu", cutpoint_layers=None):
        super().__init__(weights_dir, variant, device)
        # Cut-point layers: model-layer indices whose OUTPUT crosses to another
        # machine (from nodes.txt). Compression is applied only there. Fallback
        # (no topology given): every inter-layer boundary except the last.
        if not cutpoint_layers:
            cutpoint_layers = list(range(1, self.total_layers - 1))
        self.cutpoint_layers = list(cutpoint_layers)
        self.link_of_layer = {lay: idx for idx, lay in enumerate(self.cutpoint_layers)}
        self.num_links = len(self.cutpoint_layers)
        self._compress = False
        self._eta_vec = np.ones(self.num_links)
        self._method = "topk"
        self._params = None

    def set_config(self, method, eta_vec, llmint8_params=None):
        self._method = method
        self._eta_vec = np.asarray(eta_vec, dtype=float).reshape(-1)
        self._params = llmint8_params
        self._compress = True

    def disable_compression(self):
        self._compress = False

    def _roundtrip(self, hidden, eta):
        """Apply the active codec at ratio `eta` to `hidden` (lossy for eta<1)."""
        nu.set_compression(self._method, float(eta), llmint8_params=self._params)
        payload = nu.TorchTensorSerializer.serialize({"h": hidden})
        out = nu.TorchTensorSerializer.deserialize(payload)["h"]
        return out.to(hidden.dtype).to(self.device)

    @torch.no_grad()
    def forward_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full pipeline forward, inserting per-link compression at node outputs."""
        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]

        hidden = self.embed(token_ids)
        positions = torch.arange(seq_len, device=self.device)
        freqs = self.freqs_cis.index_select(0, positions)
        mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38,
                          dtype=torch.float, device=self.device)
        mask = torch.triu(mask, diagonal=1)
        dtype = self.config.get_dtype()

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
            # Compress the output of layer i ONLY if it is a machine cut-point
            # (its activation crosses to another machine). eta index = link_of_layer[i].
            if self._compress and i in self.link_of_layer:
                eta_i = self._eta_vec[self.link_of_layer[i]]
                if eta_i < 1.0 - 1e-9:
                    hidden = self._roundtrip(hidden, eta_i)

        if hidden.shape[-1] == self.config.hidden_size:
            logits = torch.matmul(hidden.float(), self.embedding_weights.float().T)
        else:
            logits = hidden
        return logits

    @torch.no_grad()
    def greedy_generate(self, prompt: str, out_len: int, eos_id: int) -> str:
        """Greedy decode a reference continuation (compression state as currently set)."""
        ids = self.tokenize(prompt)
        generated = []
        for _ in range(out_len):
            logits = self.forward_logits(ids)
            nxt = int(torch.argmax(logits[0, -1]).item())
            if nxt == eos_id:
                break
            generated.append(nxt)
            ids = torch.cat(
                [ids, torch.tensor([[nxt]], dtype=torch.long, device=self.device)], dim=1)
        return self.detokenize(generated)


def parse_args():
    p = argparse.ArgumentParser(description="Fit A(eta) from an offline prompt set")
    p.add_argument("--weights_dir", type=str, default="weights/1.1-2b-it")
    p.add_argument("--variant", type=str, default="2b")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--prompts_file", type=str, default="prompts_offline.txt")
    p.add_argument("--num_prompts", type=int, default=0,
                   help="Limit number of offline prompts (0 = all)")
    p.add_argument("--nodes", type=str, default="scripts/nodes.txt",
                   help="Topology file (same as the controller's --nodes). Compression "
                        "links = machine cut-points derived from it (eta-vector length).")
    p.add_argument("--method", type=str, default="topk",
                   help="Codec the model is trained for: topk | quantization | llmint8. "
                        "MUST match the no_csi run's --method.")
    p.add_argument("--llmint8_outlier_ratio", type=float, default=0.05)
    p.add_argument("--llmint8_outlier_prec", type=str, default="fp16")
    p.add_argument("--llmint8_regular_prec", type=str, default="int8")
    p.add_argument("--out_len", type=int, default=20,
                   help="Reference generation length (tokens) per prompt")
    p.add_argument("--eta_min", type=float, default=0.1)
    p.add_argument("--n_uniform", type=int, default=10,
                   help="Number of uniform-eta anchor configs across [eta_min, 1]")
    p.add_argument("--n_random", type=int, default=40,
                   help="Number of random per-link eta configs (needed so features vary; "
                        "uniform-only data is degenerate for a per-link model)")
    p.add_argument("--model_type", type=str, default="poly2",
                   help="Surrogate backend: linear_monotonic|poly2|poly3|gbm|rf|mlp|mlp_small")
    p.add_argument("--output", type=str, default="outputs/accuracy_model.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cutpoints_from_nodes(path):
    """Cut-point model-layer indices from nodes.txt (same rule as controller.py):
    a pipeline layer whose next layer (wrapping to UE=0) is on a different machine,
    excluding the return hop to the UE (next layer 0)."""
    layer_to_ip = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        parts = line.split("-")
        ip = parts[1].strip()
        for l in parts[2].split(","):
            layer_to_ip[int(l.strip())] = ip
    all_layers = sorted(layer_to_ip)
    max_layer = all_layers[-1]
    cut = []
    for l in [x for x in all_layers if x > 0]:
        nxt = (l + 1) % (max_layer + 1)          # last pipeline layer wraps to 0 (UE)
        if nxt != 0 and layer_to_ip[l] != layer_to_ip.get(nxt):
            cut.append(l)
    return cut


def main():
    args = parse_args()

    cutpoints = cutpoints_from_nodes(args.nodes)
    print(f"Cut-point layers from {args.nodes}: {cutpoints}")

    print("Loading offline pipeline (all layers on one machine)...")
    model = CompressedPipelineModel(args.weights_dir, args.variant, args.device,
                                    cutpoint_layers=cutpoints)
    n_links = model.num_links
    print(f"n_links = {n_links} (compression links = machine cut-points), "
          f"method = {args.method}")

    llmint8_params = None
    if args.method.lower() in ("llmint8", "llm_int8", "llm.int8"):
        llmint8_params = [args.llmint8_outlier_ratio,
                          args.llmint8_outlier_prec,
                          args.llmint8_regular_prec]

    eos_id = model.tokenizer.eos_id() if hasattr(model.tokenizer, "eos_id") else 1

    # Load offline prompts, wrap in the same chat template as deployment.
    raw = [l.strip() for l in open(args.prompts_file) if l.strip()]
    if args.num_prompts > 0:
        raw = raw[:args.num_prompts]
    prompts = [USER_TEMPLATE.format(prompt=r) for r in raw]
    print(f"Loaded {len(prompts)} offline prompts from {args.prompts_file}")

    # ---- Phase 1: uncompressed reference response + reference log-probs per prompt
    print("\n=== Phase 1: generating uncompressed references ===")
    model.disable_compression()
    refs = []  # (prompt, ref_response, ref_result)
    for idx, p in enumerate(prompts):
        ref_resp = model.greedy_generate(p, args.out_len, eos_id)
        if not ref_resp.strip():
            print(f"  prompt {idx+1}: empty reference, skipping")
            continue
        ref_result = compute_perplexity(model, p, ref_resp)
        refs.append((p, ref_resp, ref_result))
        print(f"  prompt {idx+1}: ref_ppl={ref_result['ppl']:.3f} "
              f"resp='{ref_resp[:50]}...'")
    if not refs:
        raise SystemExit("No usable reference responses generated; aborting.")

    # ---- Phase 2: sweep eta configs, measure utility A(eta)
    print("\n=== Phase 2: measuring A(eta) over configs ===")
    rng = np.random.default_rng(args.seed)
    configs = [np.full(n_links, float(e))
               for e in np.linspace(args.eta_min, 1.0, args.n_uniform)]
    configs += [rng.uniform(args.eta_min, 1.0, size=n_links)
                for _ in range(args.n_random)]

    X, y = [], []
    for c_idx, cfg in enumerate(configs):
        model.set_config(args.method, cfg, llmint8_params)
        utils = []
        for (p, ref_resp, ref_result) in refs:
            cmp_result = compute_perplexity(model, p, ref_resp)
            utils.append(compute_utility(ref_result, cmp_result)["utility"])
        A = float(np.mean(utils))
        X.append(np.asarray(cfg, dtype=float))
        y.append(A)
        print(f"  config {c_idx+1}/{len(configs)}: mean_eta={np.mean(cfg):.3f} -> A={A:.4f}")

    X = np.vstack(X)
    y = np.asarray(y)
    print(f"\nBuilt dataset: X={X.shape}, y={y.shape}, "
          f"A range [{y.min():.3f}, {y.max():.3f}]")

    # ---- Phase 3: fit + sanity check + save
    print("\n=== Phase 3: fitting surrogate ===")
    surrogate = SurrogateAccuracyModel(model_type=args.model_type)
    surrogate.fit(X, y)
    print(f"Fit metrics: {surrogate.fit_metrics}")
    print(f"A(ones)      = {surrogate.predict(np.ones(n_links)):.4f}   (expect ~1.0)")
    print(f"A(0.5 * ones)= {surrogate.predict(np.full(n_links, 0.5)):.4f}")
    print(f"A(eta_min)   = {surrogate.predict(np.full(n_links, args.eta_min)):.4f}")

    out = Path(args.output)
    surrogate.save(out)
    print(f"\nSaved accuracy model -> {out}")
    print("Point the controller at it:  python controller.py ... "
          f"--algorithm no_csi --accuracy_model {out}")


if __name__ == "__main__":
    main()
