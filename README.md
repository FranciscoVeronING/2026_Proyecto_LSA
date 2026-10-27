# Semantic branch — LSA glosses to Spanish

Fine-tuning of small LLMs that translate Argentine Sign Language glosses into one Rioplatense Spanish sentence. Training exports GGUF Q4_K_M and scores that binary. The fixed comparison runs on `dataset_trial.json`.

Run the scripts from `src/semantic` (modules are imported as sibling files).

## Files

| Path | Role |
|------|------|
| `train.py` | 4-bit LoRA, GGUF export, and 15% holdout evaluation |
| `eval_trial.py` | Comparison of trained checkpoints on `dataset_trial.json` |
| `models.py` | Catalog, paths, and generation limits |
| `glosses.py` | Fingerspelling, numbers, and accuracy normalization |
| `inference.py` | GGUF load and translation |
| `metrics.py` | BLEU, ROUGE-L, METEOR, BERTScore, COMET, and BLEURT |
| `dataset_glosses.json` | Gloss-to-Spanish pairs used for training |
| `dataset_trial.json` | Fixed set, with several valid translations per item |
| `prompts/sys_prompt.txt` | System prompt for train and eval |
| `requirements.txt` | Pip dependencies, without Unsloth or the COMET package |
| `outputs/eval_trial/` | Saved comparisons, including reports from before this layout |
| `outputs/training_summary.json` | History of training runs |

Weights, GGUF files, and logs stay in `outputs/` and `logs/`. Git ignores them, except the training summary and `outputs/eval_trial/`.

## Environment

NVIDIA GPU with CUDA. Unsloth does not train on CPU. 0.5B–1.7B in 4-bit need about 8 GB of VRAM; the 3B needs about 12 GB. 16 GB of RAM and about 10 GB of disk for bases, adapters, and GGUF.

```powershell
conda create -n lsa_semantic python=3.10 -y
conda activate lsa_semantic
cd src\semantic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The print must show a `+cu128` build and `True`. If it says `+cpu` or `False`, stop.

```powershell
pip install -r requirements.txt
pip install unsloth unsloth_zoo --no-deps
pip install unbabel-comet==2.2.7 --no-deps
python -c "import torch; from unsloth import FastLanguageModel; print(torch.__version__, torch.cuda.is_available())"
```

`unsloth` without `--no-deps` can replace the CUDA torch build. `unbabel-comet` without `--no-deps` downgrades `transformers` to the 4.x line and breaks Unsloth. The COMET and BLEURT code includes the patches needed to keep Transformers 5.

The first evaluation downloads `bert-base-multilingual-cased`, `Unbabel/wmt22-comet-da` (XLM-RoBERTa large), and `lucadiliello/BLEURT-20`.

## Train

```powershell
python train.py --continue-on-error
python train.py --models qwen2.5-0.5b llama-3.2-1b --continue-on-error
python train.py --epochs 5
python train.py --dataset .\dataset_glosses.json
```

Each model uses LoRA (`r=16`, `alpha=32`, dropout 0, attention and MLP modules). Training quantization is 4-bit. Load context is 2048. Training text is truncated to 512. Batch 4 and accumulation 4 (the 3B uses 2 and 8). Learning rate 5e-4, or 3e-4 for the 3B. 5 epochs, warmup 10, AdamW 8-bit, cosine, seed 3407. The internal split is 85/15 with seed 3407.

Before the prompt is built, loose letters are compacted (`M a i t e` -> `Maite`, at most two identical letters in a row) and loose digits are concatenated without truncation. The system prompt is the `system` message, glosses go in `Glosas: ...`, and the Spanish sentence is the `assistant` message. Only the response is trained. Both prompts stay in Spanish.

After training, the adapter is merged and exported as GGUF Q4_K_M. Holdout evaluation runs on that binary, not on the 4-bit model. Outputs:

```
outputs/unsloth_<Model>/          adapter and metrics.json
outputs/unsloth_<Model>_gguf/     *.Q4_K_M.gguf
outputs/training_summary.json     one entry per successful run
outputs/training_batch_<date>.json batch summary
logs/train_<date>.log
```

Training and evaluation use the Spanish system prompt and the user prefix `Glosas:`, the same pair the checkpoints already on disk were trained with.

## Evaluate `dataset_trial`

Uses the Q4_K_M GGUF when it exists. Otherwise it loads the Hugging Face checkpoint in the model folder (that path does not apply a loose LoRA adapter: it needs `config.json` from a full model).

```powershell
python eval_trial.py
python eval_trial.py --models qwen2.5-1.5b
python eval_trial.py --stop-on-error
```

Generation matches the training holdout: temperature 0, 24 new tokens, repetition penalty 1.05, stop on newline and on the Qwen and Llama end tokens. Trial glosses go through the same fingerspelling unification.

Each run writes a dated pair and overwrites `trial_comparison_latest.*`. Older reports stay in `outputs/eval_trial/`. Reports saved before this rename keep the `comparativa_trial_*` names.

## Metrics

All scores are on a 0–100 scale. With several valid references, ROUGE-L, METEOR, BERTScore, COMET, and BLEURT use the best reference for that sentence. BLEU is corpus BLEU over the reference list.

| Metric | What it measures |
|--------|------------------|
| Strict accuracy | The prediction is identical to some reference |
| Normalized accuracy | Same match, lowercased and without `¿?¡!.,;"` |
| BLEU | Surface n-grams |
| ROUGE-L | Longest common subsequence |
| METEOR | Unigrams with stemming and synonyms |
| BERTScore | F1 of `bert-base-multilingual-cased`, language `es` |
| COMET | `Unbabel/wmt22-comet-da`; the source is the gloss text |
| BLEURT | BLEURT-20, reference/prediction pair with its SentencePiece |
| Latency | Milliseconds of GGUF or HF generation, excluding the metrics |

The comparison rank sorts by normalized accuracy, then BLEU, ROUGE-L, METEOR, BERTScore, COMET, BLEURT, and finally lower latency.

## Catalog

| ID | Model |
|----|-------|
| `qwen2.5-0.5b` | `unsloth/Qwen2.5-0.5B-Instruct` |
| `qwen2.5-1.5b` | `unsloth/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-3b` | `unsloth/Qwen2.5-3B-Instruct` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct` |
| `smollm2-1.7b` | `unsloth/SmolLM2-1.7B-Instruct` |

To add a model, add one entry to `MODELS` in `models.py`: id, Hugging Face name, label, Unsloth chat template, and the instruction and response markers. Train and eval both read that list.

## Common failures

**`Unsloth cannot find any torch accelerator`.** Torch is the CPU build. Uninstall `torch` and `torchvision`, reinstall the `cu128` index, and install Unsloth again with `--no-deps`.

**`No GGUF or HF checkpoint`.** That id has not been trained yet, or the GGUF is not in `outputs/<folder>_gguf/`.

**COMET or BLEURT fail to import.** Check that `unbabel-comet` was installed with `--no-deps` and that `transformers` is still 5.x.
