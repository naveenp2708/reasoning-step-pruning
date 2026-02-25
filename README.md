# Reasoning Step Pruning via Attention Scores

Implementation of attention-based reasoning chain pruning for LLMs, inspired by:
1. **[Think Clearly: Improving Reasoning via Redundant Token Pruning](https://arxiv.org/abs/2507.08806)** (Choi et al., 2025)
2. **[TRAAC: Think Right with Adaptive, Attentive Compression](https://arxiv.org/abs/2510.01581)** (2025)

## Key Idea

Reasoning LLMs (like DeepSeek-R1) generate long chain-of-thought traces that contain significant redundancy. By analyzing attention patterns, we can identify which reasoning steps the model actually relies on when producing its final answer — and prune the rest.

**Core insight from "Think Clearly":** Steps that receive low attention from subsequent tokens are redundant. Removing them reduces distraction and can actually *improve* accuracy.

**Core insight from "TRAAC":** Self-attention over reasoning trajectories identifies important vs. expendable steps, enabling adaptive compression without retraining.

## Method

1. **Generate** a full reasoning chain for each math problem
2. **Segment** the chain into discrete reasoning steps (by paragraph breaks / reasoning markers)
3. **Score** each step by computing how much attention the answer-relevant tokens pay to it (averaged across all layers and heads)
4. **Prune** steps below a percentile threshold of importance
5. **Re-evaluate** the model with only the retained reasoning context
6. **Compare** accuracy and token efficiency across pruning thresholds

## Results Format

| Threshold | Accuracy | Steps Kept | Avg Length | Notes |
|-----------|----------|------------|------------|-------|
| 0.0 (baseline) | X% | 100% | N chars | Full reasoning chain |
| 0.1 | X% | ~90% | N chars | Light pruning |
| 0.2 | X% | ~80% | N chars | Moderate pruning |
| 0.3 | X% | ~70% | N chars | Moderate-aggressive |
| 0.4 | X% | ~60% | N chars | Aggressive pruning |
| 0.5 | X% | ~50% | N chars | Heavy pruning |

Run the experiment to fill in actual values.

## Setup

### Google Colab (Recommended)
1. Upload `Reasoning_Pruning_Experiment.ipynb` to Colab
2. Set runtime to **GPU (T4)**
3. Run all cells

### Kaggle
1. Create a new notebook, paste code from `reasoning_pruning_experiment.py`
2. Enable **GPU T4 x2** accelerator
3. Run

### Local
```bash
pip install transformers accelerate bitsandbytes torch sentencepiece datasets
python reasoning_pruning_experiment.py
```

## Model

- **DeepSeek-R1-Distill-Qwen-1.5B** (4-bit quantized via bitsandbytes)
- Uses `<think>...</think>` structured reasoning
- Small enough for free-tier Colab/Kaggle T4 GPU

## Dataset

- **AIME 2024** (American Invitational Mathematics Examination)
- 5 problems included in code (expand to full 30 for comprehensive eval)
- Integer answers (0-999), straightforward to verify

## Files

```
├── README.md
├── reasoning_pruning_experiment.py    # Full Python script
├── Reasoning_Pruning_Experiment.ipynb # Colab notebook
└── aime24_pruning_results.json        # Results (generated after running)
```

## How It Works (Technical Details)

### Attention Scoring
For each reasoning step, we compute importance as:

```
importance(step_i) = Σ attention(last_token → token_j) for all token_j in step_i
```

Where attention is averaged across all layers and heads. This captures how much the model's final output "looks back" at each reasoning step.

### Pruning Strategy
Given threshold `t` (0 to 1):
- Compute the `t`-th percentile of step importance scores
- Remove all steps below this cutoff
- Always keep at least the highest-importance step
- Reconstruct the reasoning chain from remaining steps

### Re-evaluation
After pruning, we feed the retained reasoning back to the model as context and ask it to produce a final answer. This simulates the effect of KV-cache pruning at inference time.

## References

```bibtex
@article{choi2025thinkclearly,
  title={Think Clearly: Improving Reasoning via Redundant Token Pruning},
  author={Choi, Daewon and Lee, Jimin and Tack, Jihoon and Song, Woomin and others},
  journal={arXiv preprint arXiv:2507.08806},
  year={2025}
}

@article{traac2025,
  title={Think Right with Adaptive, Attentive Compression},
  journal={arXiv preprint arXiv:2510.01581},
  year={2025}
}
```

## Author

Naveen Pasupuleti — Collaboration application for Sarvesh Gharat (IIT Bombay)
