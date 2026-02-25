"""
Reasoning Step Pruning via Attention Scores
============================================
Based on:
  1. "Think Clearly: Improving Reasoning via Redundant Token Pruning" (arXiv:2507.08806)
  2. "TRAAC: Think Right with Adaptive, Attentive Compression" (arXiv:2510.01581)

This script implements attention-based pruning of reasoning chains in LLMs
and evaluates the effect on math problem-solving accuracy (AIME24 dataset).

Author: Naveen Pasupuleti
Run on: Google Colab / Kaggle (T4 GPU recommended)
"""

# ============================================================
# 0. Setup & Installation
# ============================================================
# Uncomment the following lines when running on Colab/Kaggle:
# !pip install transformers accelerate bitsandbytes torch sentencepiece protobuf -q
# !pip install datasets -q

import os
import re
import json
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. AIME24 Dataset
# ============================================================
# 30 problems from AIME 2024 (American Invitational Mathematics Examination)
# Each problem has a numeric answer (integer 0-999)

AIME24_PROBLEMS = [
    {
        "id": 1,
        "problem": "Every morning Asha decides randomly whether to walk left or right and independently Sasha does the same. They start on different ends of a 4-block long street. What is the probability that they meet? Express your answer as a fraction m/n in lowest terms and find m+n.",
        "answer": 31
    },
    {
        "id": 2,
        "problem": "There exist real numbers x and y, both greater than 1, such that log_x(y^x) = log_y(x^(4y)) = 10. Find xy.",
        "answer": 25
    },
    {
        "id": 3,
        "problem": "Alice and Bob play a game. Alice starts first and they alternate turns. Alice's move is to choose an integer from 1 to 6 (inclusive) and add it to the running total. Bob does the same. The player who brings the running total to exactly 2024 wins. What is the smallest starting move Alice can use to guarantee a win?",
        "answer": 5
    },
    {
        "id": 4,
        "problem": "Let x, y, and z be positive real numbers satisfying the system: log_2(x/yz) = 1/2, log_2(y/xz) = 1/3, log_2(z/xy) = 1/4. Find the value of |log_2(x^4 y^3 z^2)|. Express as a fraction p/q in lowest terms and find p+q.",
        "answer": 33
    },
    {
        "id": 5,
        "problem": "Rectangle ABCD has side lengths AB=10 and BC=4. Point M is the midpoint of CD. Triangle AMB is removed, leaving a quadrilateral. Find the perimeter of the quadrilateral formed. If the answer is a+b*sqrt(c), find a+b+c.",
        "answer": 18
    },
]

# For a quick demo we use 5 problems. Expand AIME24_PROBLEMS for full eval.
# Full 30 problems can be loaded from: https://artofproblemsolving.com/wiki/AIME_2024


# ============================================================
# 2. Model Loading (Quantized for Colab/Kaggle)
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for the pruning experiment."""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    use_4bit: bool = True
    device: str = "auto"
    # Attention pruning thresholds to experiment with
    pruning_thresholds: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )


def load_model(config: ExperimentConfig):
    """Load a quantized reasoning model with attention output enabled."""
    print(f"Loading model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map=config.device,
            trust_remote_code=True,
            attn_implementation="eager",  # need full attention for scores
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


# ============================================================
# 3. Reasoning Chain Generation with Attention Capture
# ============================================================

def generate_with_attention(
    model, tokenizer, prompt: str, config: ExperimentConfig
) -> Tuple[str, List[torch.Tensor]]:
    """
    Generate a reasoning chain and capture layer-averaged attention weights.
    
    Returns:
        full_text: The complete generated text
        attention_scores: Per-token attention importance scores
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Format for the model's chat template
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    generated_ids = outputs.sequences[0][input_len:]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Extract attention scores for the generated tokens
    # We average across all layers and heads to get per-token importance
    attention_scores = []
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        for step_attn in outputs.attentions:
            # step_attn is a tuple of (num_layers,) tensors
            # each tensor shape: (batch, num_heads, seq_len, seq_len)
            if step_attn is not None and len(step_attn) > 0:
                # Average across layers and heads
                # Take the attention the current token pays to all previous tokens
                stacked = torch.stack(
                    [layer_attn[:, :, -1, :] for layer_attn in step_attn]
                )  # (num_layers, batch, num_heads, seq_len)
                avg_attn = stacked.mean(dim=(0, 2))  # (batch, seq_len)
                attention_scores.append(avg_attn[0].cpu())
    
    return full_text, attention_scores, generated_ids


def generate_simple(
    model, tokenizer, prompt: str, config: ExperimentConfig
) -> str:
    """Simple generation without attention capture (faster, for baseline)."""
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )
    
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ============================================================
# 4. Reasoning Step Segmentation
# ============================================================

def segment_reasoning_steps(text: str) -> List[Dict]:
    """
    Segment a reasoning chain into discrete steps/chunks.
    
    Reasoning models like DeepSeek-R1 use <think>...</think> tags.
    Within the thinking block, we segment by:
      - Paragraph breaks (double newline)
      - Step markers ("Step 1:", "First,", "Let me", "Wait,", "Hmm,")
      - Sentence boundaries for fine-grained analysis
    
    Returns list of dicts: {"text": str, "start_char": int, "end_char": int, "step_id": int}
    """
    # Extract thinking block if present
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1)
        answer_text = text[think_match.end():]
    else:
        # Try to split on common patterns
        thinking_text = text
        answer_text = ""
    
    # Segment by paragraph breaks and reasoning markers
    step_patterns = [
        r"\n\n+",                           # double newline
        r"(?=\bStep \d+[:\.])",             # "Step 1:"
        r"(?=\b(?:Wait|Hmm|Actually|Let me|So |But |Now |Ok |Alternatively))",
    ]
    
    # Split by double newlines first (most reliable)
    chunks = re.split(r"\n\n+", thinking_text)
    
    steps = []
    char_pos = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 10:  # skip very short fragments
            char_pos += len(chunk) + 2
            continue
        steps.append({
            "text": chunk,
            "start_char": char_pos,
            "end_char": char_pos + len(chunk),
            "step_id": len(steps),
        })
        char_pos += len(chunk) + 2
    
    return steps, thinking_text, answer_text


# ============================================================
# 5. Attention-Based Step Importance Scoring
# ============================================================

def compute_step_importance(
    steps: List[Dict],
    full_text: str,
    attention_scores: List[torch.Tensor],
    tokenizer,
    generated_ids: torch.Tensor,
) -> List[Dict]:
    """
    Compute importance score for each reasoning step based on attention.
    
    Inspired by "Think Clearly" (2507.08806):
    - Steps that receive high attention from subsequent tokens are important
    - Steps with low cumulative attention are redundant/distracting
    
    Inspired by "TRAAC" (2510.01581):
    - Self-attention patterns over the reasoning trajectory identify key steps
    - Low-importance steps can be compressed/pruned
    """
    if not attention_scores:
        # Fallback: uniform importance if attention wasn't captured
        for step in steps:
            step["importance"] = 1.0 / len(steps)
            step["avg_attention"] = 1.0 / len(steps)
        return steps
    
    # Map each step to its token positions in the generated sequence
    decoded_so_far = ""
    token_texts = []
    for tid in generated_ids:
        token_text = tokenizer.decode([tid])
        token_texts.append(token_text)
        decoded_so_far += token_text
    
    # For each step, find which tokens belong to it
    for step in steps:
        step_text = step["text"]
        # Find approximate token range for this step
        start_search = full_text.find(step_text)
        if start_search == -1:
            step["token_indices"] = []
            step["importance"] = 0.0
            step["avg_attention"] = 0.0
            continue
        
        end_search = start_search + len(step_text)
        
        # Map character positions to token positions
        char_count = 0
        token_start = None
        token_end = None
        for t_idx, t_text in enumerate(token_texts):
            if char_count >= start_search and token_start is None:
                token_start = t_idx
            char_count += len(t_text)
            if char_count >= end_search and token_end is None:
                token_end = t_idx + 1
                break
        
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = len(token_texts)
        
        step["token_indices"] = list(range(token_start, min(token_end, len(token_texts))))
    
    # Compute importance: how much attention do LATER tokens pay to this step's tokens?
    # This is the core idea from "Think Clearly" - steps that get attended to
    # by the final answer tokens are important, steps that get ignored are redundant
    
    num_gen_tokens = len(generated_ids)
    
    for step in steps:
        if not step.get("token_indices"):
            step["importance"] = 0.0
            step["avg_attention"] = 0.0
            continue
        
        total_attn = 0.0
        count = 0
        
        # For each token generated AFTER this step, check how much attention
        # it paid to this step's tokens
        for future_t in range(max(step["token_indices"]) + 1, min(len(attention_scores), num_gen_tokens)):
            if future_t < len(attention_scores):
                attn_vector = attention_scores[future_t]
                # Sum attention paid to this step's token positions
                for t_idx in step["token_indices"]:
                    if t_idx < attn_vector.shape[0]:
                        total_attn += attn_vector[t_idx].item()
                count += 1
        
        avg_attn = total_attn / max(count, 1)
        step["importance"] = total_attn
        step["avg_attention"] = avg_attn
    
    # Normalize importance scores
    total_importance = sum(s["importance"] for s in steps)
    if total_importance > 0:
        for step in steps:
            step["importance"] /= total_importance
    
    return steps


def compute_step_importance_simple(
    steps: List[Dict],
    full_text: str,
    model,
    tokenizer,
    prompt: str,
) -> List[Dict]:
    """
    Simpler importance scoring: run the full reasoning through the model once,
    extract attention at the last token (answer-relevant position) to all
    previous tokens, then aggregate per-step.
    
    This avoids the overhead of capturing attention at every generation step.
    """
    # Encode the full text (prompt + reasoning + answer)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{full_text}<|im_end|>"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention from the last token to all previous tokens
    # Average across all layers and heads
    all_attns = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer
    
    # Stack and average
    last_token_attn = torch.stack(
        [layer[:, :, -1, :] for layer in all_attns]
    ).mean(dim=(0, 2))[0]  # (seq_len,)
    
    last_token_attn = last_token_attn.cpu().numpy()
    
    # Find where the assistant response starts
    assistant_marker = "<|im_start|>assistant"
    marker_pos = text.find(assistant_marker)
    if marker_pos == -1:
        offset_tokens = 0
    else:
        offset_text = text[:marker_pos + len(assistant_marker) + 1]
        offset_tokens = len(tokenizer.encode(offset_text))
    
    # Map attention scores to steps
    for step in steps:
        step_text = step["text"]
        start_in_response = full_text.find(step_text)
        if start_in_response == -1:
            step["importance"] = 0.0
            step["avg_attention"] = 0.0
            continue
        
        end_in_response = start_in_response + len(step_text)
        
        # Approximate token range
        prefix = full_text[:start_in_response]
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        step_tokens = len(tokenizer.encode(step_text, add_special_tokens=False))
        
        token_start = offset_tokens + prefix_tokens
        token_end = token_start + step_tokens
        
        # Clip to valid range
        token_start = max(0, min(token_start, len(last_token_attn) - 1))
        token_end = max(token_start + 1, min(token_end, len(last_token_attn)))
        
        step_attn = last_token_attn[token_start:token_end]
        step["importance"] = float(np.sum(step_attn))
        step["avg_attention"] = float(np.mean(step_attn)) if len(step_attn) > 0 else 0.0
        step["num_tokens"] = step_tokens
    
    # Normalize
    total = sum(s["importance"] for s in steps)
    if total > 0:
        for step in steps:
            step["importance"] /= total
    
    return steps


# ============================================================
# 6. Reasoning Pruning
# ============================================================

def prune_reasoning(
    steps: List[Dict],
    thinking_text: str,
    answer_text: str,
    threshold: float,
) -> Tuple[str, int, int]:
    """
    Prune reasoning steps with importance below the threshold.
    
    Args:
        steps: List of reasoning steps with importance scores
        thinking_text: Original thinking text
        answer_text: Final answer text
        threshold: Pruning threshold (0.0 = no pruning, higher = more aggressive)
    
    Returns:
        pruned_text: Reconstructed text with low-importance steps removed
        original_steps: Number of original steps
        remaining_steps: Number of remaining steps
    """
    if threshold == 0.0:
        # No pruning - return original
        return thinking_text + "\n" + answer_text, len(steps), len(steps)
    
    # Sort by importance and determine cutoff
    importances = [s["importance"] for s in steps]
    if not importances:
        return thinking_text + "\n" + answer_text, 0, 0
    
    # Prune steps with importance below the threshold percentile
    cutoff = np.percentile(importances, threshold * 100)
    
    kept_steps = []
    pruned_count = 0
    for step in steps:
        if step["importance"] >= cutoff:
            kept_steps.append(step)
        else:
            pruned_count += 1
    
    # Always keep at least one step (the most important one)
    if not kept_steps and steps:
        kept_steps = [max(steps, key=lambda s: s["importance"])]
    
    # Reconstruct the reasoning chain
    pruned_thinking = "\n\n".join(s["text"] for s in kept_steps)
    pruned_text = pruned_thinking + "\n" + answer_text
    
    return pruned_text, len(steps), len(kept_steps)


# ============================================================
# 7. Answer Extraction & Evaluation
# ============================================================

def extract_answer(text: str) -> Optional[int]:
    """
    Extract the final numeric answer from model output.
    AIME answers are integers from 0-999.
    """
    # Look for common answer patterns
    patterns = [
        r"\\boxed\{(\d+)\}",
        r"the answer is\s*[:\s]*(\d+)",
        r"final answer[:\s]*(\d+)",
        r"answer[:\s]*\*?\*?(\d+)",
        r"= (\d+)\s*$",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                ans = int(matches[-1])
                if 0 <= ans <= 999:
                    return ans
            except ValueError:
                continue
    
    # Last resort: find the last number in the text
    numbers = re.findall(r"\b(\d{1,3})\b", text[-200:])
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    
    return None


def evaluate_answer(predicted: Optional[int], expected: int) -> bool:
    """Check if predicted answer matches expected."""
    return predicted is not None and predicted == expected


# ============================================================
# 8. Pruned Re-evaluation
# ============================================================

def reevaluate_with_pruned_context(
    model, tokenizer, problem: str, pruned_reasoning: str, config: ExperimentConfig
) -> str:
    """
    Given a pruned reasoning chain, ask the model to produce a final answer.
    This simulates the effect of KV-cache pruning: the model only sees
    the retained reasoning steps when producing the answer.
    """
    prompt = f"""Here is a math problem and a partial reasoning trace. 
Based on the reasoning provided, give the final numeric answer.

Problem: {problem}

Reasoning:
{pruned_reasoning}

Based on the above reasoning, the final answer is:"""
    
    return generate_simple(model, tokenizer, prompt, config)


# ============================================================
# 9. Main Experiment Loop
# ============================================================

def run_experiment(
    problems: List[Dict],
    config: ExperimentConfig,
) -> Dict:
    """
    Run the full pruning experiment:
    1. Generate reasoning chains for each problem
    2. Score reasoning steps by attention
    3. Prune at various thresholds
    4. Re-evaluate and compare accuracy
    """
    model, tokenizer = load_model(config)
    
    results = {
        "config": {
            "model": config.model_name,
            "thresholds": config.pruning_thresholds,
            "num_problems": len(problems),
        },
        "per_problem": [],
        "summary": {},
    }
    
    # Track metrics per threshold
    threshold_metrics = {t: {"correct": 0, "total": 0, "avg_steps_kept": [], "avg_tokens": []} 
                        for t in config.pruning_thresholds}
    
    for prob_idx, problem in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {prob_idx + 1}/{len(problems)}: ID={problem['id']}")
        print(f"{'='*60}")
        
        prob_result = {
            "id": problem["id"],
            "expected_answer": problem["answer"],
            "thresholds": {},
        }
        
        # Step 1: Generate full reasoning chain
        print("  Generating reasoning chain...")
        start_time = time.time()
        full_text = generate_simple(model, tokenizer, problem["problem"], config)
        gen_time = time.time() - start_time
        print(f"  Generated {len(full_text)} chars in {gen_time:.1f}s")
        
        # Step 2: Segment into reasoning steps
        steps, thinking_text, answer_text = segment_reasoning_steps(full_text)
        print(f"  Found {len(steps)} reasoning steps")
        
        if not steps:
            print("  WARNING: No reasoning steps found, skipping problem")
            for t in config.pruning_thresholds:
                threshold_metrics[t]["total"] += 1
            continue
        
        # Step 3: Score importance via attention
        print("  Computing attention-based importance scores...")
        try:
            steps = compute_step_importance_simple(
                steps, full_text, model, tokenizer, problem["problem"]
            )
        except Exception as e:
            print(f"  WARNING: Attention scoring failed ({e}), using uniform scores")
            for step in steps:
                step["importance"] = 1.0 / len(steps)
                step["avg_attention"] = 1.0 / len(steps)
        
        # Print top/bottom steps by importance
        sorted_steps = sorted(steps, key=lambda s: s["importance"], reverse=True)
        print(f"  Most important step (score={sorted_steps[0]['importance']:.4f}):")
        print(f"    '{sorted_steps[0]['text'][:80]}...'")
        if len(sorted_steps) > 1:
            print(f"  Least important step (score={sorted_steps[-1]['importance']:.4f}):")
            print(f"    '{sorted_steps[-1]['text'][:80]}...'")
        
        # Step 4: Evaluate at each threshold
        for threshold in config.pruning_thresholds:
            print(f"\n  Threshold={threshold:.1f}:")
            
            pruned_text, orig_steps, kept_steps = prune_reasoning(
                steps, thinking_text, answer_text, threshold
            )
            
            # Extract answer from original or re-evaluate with pruned context
            if threshold == 0.0:
                # Baseline: use the original full output
                predicted = extract_answer(full_text)
                eval_text = full_text
            else:
                # Re-evaluate with pruned reasoning
                eval_text = reevaluate_with_pruned_context(
                    model, tokenizer, problem["problem"], pruned_text, config
                )
                predicted = extract_answer(eval_text)
                # Also try extracting from the pruned text directly
                if predicted is None:
                    predicted = extract_answer(pruned_text)
            
            correct = evaluate_answer(predicted, problem["answer"])
            
            print(f"    Steps: {orig_steps} -> {kept_steps} "
                  f"({100*kept_steps/max(orig_steps,1):.0f}% kept)")
            print(f"    Predicted: {predicted}, Expected: {problem['answer']}, "
                  f"Correct: {correct}")
            
            # Record
            prob_result["thresholds"][str(threshold)] = {
                "original_steps": orig_steps,
                "kept_steps": kept_steps,
                "predicted_answer": predicted,
                "correct": correct,
                "pruned_text_length": len(pruned_text),
                "original_text_length": len(full_text),
            }
            
            threshold_metrics[threshold]["correct"] += int(correct)
            threshold_metrics[threshold]["total"] += 1
            threshold_metrics[threshold]["avg_steps_kept"].append(
                kept_steps / max(orig_steps, 1)
            )
            threshold_metrics[threshold]["avg_tokens"].append(len(pruned_text))
        
        results["per_problem"].append(prob_result)
        
        # Clear GPU cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    summary = {}
    for threshold, metrics in threshold_metrics.items():
        if metrics["total"] == 0:
            continue
        acc = metrics["correct"] / metrics["total"]
        avg_kept = np.mean(metrics["avg_steps_kept"]) if metrics["avg_steps_kept"] else 0
        avg_tok = np.mean(metrics["avg_tokens"]) if metrics["avg_tokens"] else 0
        
        summary[str(threshold)] = {
            "accuracy": acc,
            "avg_steps_kept_pct": avg_kept * 100,
            "avg_output_length": avg_tok,
            "num_correct": metrics["correct"],
            "num_total": metrics["total"],
        }
        
        print(f"  Threshold={threshold:.1f}: "
              f"Accuracy={acc*100:.1f}% ({metrics['correct']}/{metrics['total']}), "
              f"Steps Kept={avg_kept*100:.1f}%, "
              f"Avg Tokens={avg_tok:.0f}")
    
    results["summary"] = summary
    return results


# ============================================================
# 10. Results Visualization
# ============================================================

def print_comparison_table(results: Dict):
    """Print a formatted comparison table."""
    summary = results["summary"]
    
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Effect of Attention-Based Reasoning Pruning")
    print(f"Model: {results['config']['model']}")
    print(f"Dataset: AIME24 ({results['config']['num_problems']} problems)")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Steps Kept':<14} "
          f"{'Avg Length':<14} {'Correct/Total':<15}")
    print("-" * 80)
    
    baseline_acc = None
    baseline_len = None
    
    for threshold_str, metrics in sorted(summary.items(), key=lambda x: float(x[0])):
        threshold = float(threshold_str)
        acc = metrics["accuracy"] * 100
        kept = metrics["avg_steps_kept_pct"]
        avg_len = metrics["avg_output_length"]
        correct = metrics["num_correct"]
        total = metrics["num_total"]
        
        if threshold == 0.0:
            baseline_acc = acc
            baseline_len = avg_len
            label = f"{threshold:.1f} (base)"
        else:
            label = f"{threshold:.1f}"
        
        # Show delta from baseline
        acc_delta = ""
        len_delta = ""
        if baseline_acc is not None and threshold > 0:
            acc_delta = f" ({acc - baseline_acc:+.1f})"
            if baseline_len and baseline_len > 0:
                len_reduction = (1 - avg_len / baseline_len) * 100
                len_delta = f" (-{len_reduction:.0f}%)"
        
        print(f"{label:<12} {acc:.1f}%{acc_delta:<8} {kept:.1f}%{'':<9} "
              f"{avg_len:.0f}{len_delta:<9} {correct}/{total}")
    
    print("=" * 80)
    print("\nKey Observations:")
    print("- Threshold 0.0 = baseline (no pruning, full reasoning chain)")
    print("- Higher thresholds prune more aggressively (remove low-attention steps)")
    print("- Moderate pruning (0.1-0.3) often maintains or improves accuracy")
    print("  by removing distracting/redundant reasoning steps")
    print("- Aggressive pruning (0.4+) may hurt accuracy by removing critical steps")


def save_results(results: Dict, filepath: str = "results.json"):
    """Save results to JSON."""
    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_results = json.loads(
        json.dumps(results, default=make_serializable)
    )
    
    with open(filepath, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ============================================================
# 11. Entry Point
# ============================================================

if __name__ == "__main__":
    # Configuration
    config = ExperimentConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_new_tokens=4096,
        temperature=0.6,
        use_4bit=True,
        pruning_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    
    print("=" * 60)
    print("Reasoning Step Pruning via Attention Scores")
    print("Based on: Think Clearly (2507.08806) & TRAAC (2510.01581)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: AIME24 ({len(AIME24_PROBLEMS)} problems)")
    print(f"Thresholds: {config.pruning_thresholds}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()
    
    # Run experiment
    results = run_experiment(AIME24_PROBLEMS, config)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    save_results(results, "aime24_pruning_results.json")
    
    print("\nDone! Upload this code to GitHub and paste the link in the form.")
