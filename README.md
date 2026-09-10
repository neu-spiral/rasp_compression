# Distributed LLM Inference with Activation Compression on Raspberry Pis

This repository is the Raspberry Pi, autoregressive-LLM (Gemma-2 2B) deployment of the work:

**Communication-Aware Model Distributed Inference via Latent Representation Compression** (MobiHoc 2026)

The optimizer and offline simulation code (dependency) is available [here](https://github.com/neu-spiral/communication-aware-inference).

Gemma-2 2B is split layer by layer and run as a pipeline across a cluster of Raspberry Pis. Between machines, the intermediate **activations** are compressed by a factor η ∈ (0, 1] to meet a target token throughput over a fluctuating WiFi link — trading a little accuracy for lower communication delay. A central controller measures each link and picks the per-cut-point compression rate every time slot, using either a channel-state-aware (CSI) closed-form allocation or a channel-oblivious (no-CSI) stochastic dual-descent policy.

This repository contains the deployment code, the controller, and the offline tools for training the accuracy surrogate and measuring text quality.

## Splitting the LLM

To partition the weights of the LLM:

1. **Run the partition notebook.** Open and execute `PartitionGemma.ipynb`. You will need a Kaggle account and permission to download the Gemma weights. It offers:
   - **Full Model Split** — partitions the model layer by layer; this is what the deployment uses.
   - **First Layer and Remaining Layers Split** — separates the first layer from the rest.

2. **Select the model variant.** The notebook partitions Gemma 2B by default. For a 7B variant, change the `VARIANT` variable in the second cell (see [Kaggle: Gemma Models](https://www.kaggle.com/models/google/gemma)).

3. **Move the partitions.** After partitioning, transfer the model files (including the tokenizer, which may be saved to a separate directory) to where they will run.

Alternatively, download the pre-split 2B weights from this [link](https://drive.google.com/file/d/1H7-7gW61q17H974-noDhz9-h5xQECKRb/view?usp=sharing).

## Deploying on Raspberry Pis

1. **Install on each node.** On every Pi participating in inference, clone the repo and set up a virtual environment:
   ```bash
   python -m venv venv
   source ./venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Place the weights.** Put the model weights (including the tokenizer) in `rasp_compression/weights/$VARIANT$` on each node.

3. **End-user (UE) node.** Ensure the UE's copy of the repo is at `$HOME/RESEARCH/rasp_compression`.

4. **Create the node allocation file.** Create `nodes.txt` **on the UE** in `rasp_compression/scripts`, specifying where each layer runs:
   ```plaintext
   username1-ip1-layer_num1, layer_num2, layer_num3
   username2-ip2-layer_num4, layer_num5
   username3-ip3-layer_num6
   ...
   ```

5. **Launch the nodes.**
   ```bash
   cd scripts
   bash init_demo.sh nodes.txt
   ```

6. **Stop the nodes** when done:
   ```bash
   bash kill_scripts.sh nodes.txt
   ```

## Compression at Machine Cut-Points

While the model runs, each board passes intermediate numbers (**activations**) to the next board. Before sending, activations are **compressed** by a factor η (`eta`) ∈ (0, 1] — the fraction of data kept.

Compression is applied **only at machine cut-points** — the last layer of each machine, where the activation actually crosses the network to a *different* machine. Layers that sit on the same machine hand off locally and are sent raw. The controller determines these cut-points automatically, so only the inter-machine links carry a compression rate.

A central `controller.py` collects per-node metrics over UDP and chooses η per cut-point each time slot, according to `--algorithm`:

| `--algorithm` | Description |
|---|---|
| `uniform` | Fixed η everywhere (`--default_eta`) |
| `uniform_csi` | Single η, bottlenecked by the worst link |
| `csi_aware` | Per-link `η = min(1, c/(R·a))` (closed form) |
| `no_csi` | Estimated stochastic dual descent, maximizes `A(η) − μ·λ·D̂` |
| `no_csi_myopic` / `no_csi_conservative` / `no_csi_moving_avg` | No-CSI baselines with different channel estimates |

The accuracy surrogate `A(η)` (used only by `no_csi`) is trained offline with `train_acc_model.py` on a held-out prompt set (`prompts_offline.txt`) and is codec-specific — train one `.pkl` per compression method and pass the matching one with `--accuracy_model`.

## Running an Experiment & Collecting Data

A run has three parts, started in this order on the same WiFi:

1. **Launch the Pi nodes** (SSHes to each Pi and starts one `resilientNode.py` per layer, pointing them at the controller):
   ```bash
   cd scripts && bash init_demo.sh nodes.txt
   ```

2. **Start the controller** on the UE. It assigns η per cut-point, collects reports, and computes excess delay `ΔD = D_act − 1/R` each slot:
   ```bash
   python controller.py --nodes scripts/nodes.txt --algorithm uniform \
       --method topk --R_t 5.0 --default_eta 1.0 --output_dir results/run1
   ```

3. **Start inference** on the UE. It feeds prompts through the pipeline and saves the generated text:
   ```bash
   python batch_inference.py --host <UE_ip> --port 12346 \
       --next_layers <layer1_ip:port> <layer2_ip:port> ... \
       --prompts_file prompts.txt --output_dir results/run1 --max_tokens 50
   ```

`batch_inference.py` **exits on its own** once all prompts finish. Then stop the `controller` and it writes the data files. Everything is saved to `--output_dir`:

| File | Written by | Contents |
|---|---|---|
| `per_slot_delays.json` / `.csv` | controller | Per slot: `step, D_act, excess_delay, eta[...], t_0` |
| `controller_summary.json` | controller | Run averages: `avg_D_act`, `avg_excess_delay`, `deadline_met` |
| `metrics_log.json` | controller | Raw per-node reports (compute time τ, throughput c, bytes) |
| `prompt_N.txt` / `response_N.txt` | batch_inference | Prompt sent and text generated |
| `timing_N.json` | batch_inference | Per-token arrival times at the UE |

Stop the Pis afterwards with `cd scripts && bash kill_scripts.sh nodes.txt`.

## Prompt Data

Create two prompt files yourself (they are **not** shipped with the repo), placed at the repo root, with **one prompt per line**:

- **`prompts.txt`** — the prompts sent through the pipeline at deployment (`batch_inference.py --prompts_file`).
- **`prompts_offline.txt`** — a separate held-out set used to train the accuracy surrogate (`train_acc_model.py`).

Each line is a single user prompt, for example:

```plaintext
Explain how cross-device ad tracking works.
Write a one-day pescatarian meal plan around 1900 kcal.
Summarize the plot of a short horror story about a water tower.
```

We selected ours from the **ShareGPT** dataset (real user/assistant chats): pick a handful of conversations and keep each user turn as one line. Source: [ShareGPT_Vicuna_unfiltered on Hugging Face](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).

## Measuring Text Quality

To quantify the accuracy cost of compression, `measure_ppl.py` compares each response against the no-compression reference and reports a perplexity / utility score ∈ [0, 1] per prompt (saved to `ppl_results.json`):

```bash
python measure_ppl.py --weights_dir weights/1.1-2b-it \
    --prompts_dir results/run1 --responses_dir results/run1 \
    --reference_dir results/no_compression --num_prompts 7
```
