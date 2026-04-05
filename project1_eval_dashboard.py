"""
PROJECT 1: AI Content Evaluation Dashboard
Tests prompt variations (zero-shot, few-shot, CoT) and scores outputs
with BLEU, ROUGE, and custom rubrics.
"""

import streamlit as st
import anthropic
import json
import time
import re
from dataclasses import dataclass, field
from typing import Optional
import math

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Prompt Evaluation Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

  .stApp { background: #0d0f14; color: #e8eaf0; font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #7cf0c8; }
  .metric-card {
    background: #1a1d27; border: 1px solid #2a2d3d; border-radius: 12px;
    padding: 20px; margin: 8px 0;
  }
  .score-high { color: #7cf0c8; font-weight: 700; }
  .score-mid  { color: #f0c87c; font-weight: 700; }
  .score-low  { color: #f07c7c; font-weight: 700; }
  .prompt-tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
  }
  .tag-zeroshot { background: #1e3a4a; color: #7cd4f0; border: 1px solid #7cd4f0; }
  .tag-fewshot  { background: #2a3a1e; color: #a0f07c; border: 1px solid #a0f07c; }
  .tag-cot      { background: #3a1e3a; color: #e07cf0; border: 1px solid #e07cf0; }
  .output-box {
    background: #111320; border: 1px solid #2a2d3d; border-radius: 8px;
    padding: 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; line-height: 1.6; white-space: pre-wrap;
    max-height: 300px; overflow-y: auto;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7cf0c8, #5abfa0);
    color: #0d0f14; font-weight: 700; border: none;
    font-family: 'JetBrains Mono', monospace;
  }
  .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Metrics (pure Python, no NLTK) ─────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        ref_ng = ngrams(ref_tokens, n)
        hyp_ng = ngrams(hyp_tokens, n)
        if not hyp_ng:
            scores.append(0.0)
            continue
        ref_count = {}
        for ng in ref_ng:
            ref_count[ng] = ref_count.get(ng, 0) + 1
        matches = 0
        for ng in hyp_ng:
            if ref_count.get(ng, 0) > 0:
                matches += 1
                ref_count[ng] -= 1
        scores.append(matches / len(hyp_ng))

    if not scores or all(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores if s > 0) / len(scores)

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else \
         math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    return round(bp * math.exp(log_avg) * 100, 2)


def compute_rouge_n(reference: str, hypothesis: str, n: int = 1) -> dict:
    ref_ng = ngrams(tokenize(reference), n)
    hyp_ng = ngrams(tokenize(hypothesis), n)
    if not ref_ng or not hyp_ng:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ref_set = {}
    for ng in ref_ng:
        ref_set[ng] = ref_set.get(ng, 0) + 1
    matches = 0
    for ng in hyp_ng:
        if ref_set.get(ng, 0) > 0:
            matches += 1
            ref_set[ng] -= 1
    precision = matches / len(hyp_ng)
    recall    = matches / len(ref_ng)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2)
    }


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """LCS-based ROUGE-L."""
    ref_t = tokenize(reference)
    hyp_t = tokenize(hypothesis)
    if not ref_t or not hyp_t:
        return 0.0
    m, n = len(ref_t), len(hyp_t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_t[i - 1] == hyp_t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n
    recall    = lcs / m
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return round(f1 * 100, 2)


def custom_rubric_score(output: str, task: str, reference: str) -> dict:
    """Custom rule-based rubric scoring."""
    scores = {}
    words = tokenize(output)

    # Fluency: avg word length as proxy (real = language model perplexity)
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    scores["fluency"] = min(100, round(50 + avg_len * 4, 1))

    # Conciseness: penalise very long or very short responses
    ref_words = len(tokenize(reference))
    out_words = len(words)
    ratio = out_words / max(ref_words, 1)
    scores["conciseness"] = round(max(0, 100 - abs(ratio - 1.0) * 60), 1)

    # Relevance: keyword overlap with reference
    ref_set = set(tokenize(reference))
    out_set = set(words)
    overlap = len(ref_set & out_set) / max(len(ref_set), 1)
    scores["relevance"] = round(overlap * 100, 1)

    # Completeness: sentence count
    sentences = re.split(r'[.!?]+', output)
    sentences = [s.strip() for s in sentences if s.strip()]
    scores["completeness"] = min(100, round(len(sentences) * 12.5, 1))

    scores["overall"] = round(
        0.3 * scores["fluency"] +
        0.2 * scores["conciseness"] +
        0.3 * scores["relevance"] +
        0.2 * scores["completeness"], 1
    )
    return scores


# ─── Prompt Builders ─────────────────────────────────────────────────────────
def build_prompts(task: str, input_text: str) -> dict[str, str]:
    if task == "Summarization":
        return {
            "zero_shot": f"Summarize the following text:\n\n{input_text}",
            "few_shot": (
                "Here are some examples of good summaries:\n\n"
                "TEXT: The Great Barrier Reef is the world's largest coral reef system, "
                "stretching over 2,300 kilometres.\n"
                "SUMMARY: The Great Barrier Reef is the world's largest coral system, spanning 2,300 km.\n\n"
                "TEXT: Machine learning is a subset of artificial intelligence that enables systems "
                "to learn from data.\n"
                "SUMMARY: Machine learning is an AI subfield focused on data-driven learning.\n\n"
                f"Now summarize:\nTEXT: {input_text}\nSUMMARY:"
            ),
            "chain_of_thought": (
                "Let's summarize this text step by step:\n"
                "1. First identify the main topic\n"
                "2. Find the key supporting points\n"
                "3. Note any important numbers or names\n"
                "4. Write a concise summary in 1-2 sentences\n\n"
                f"Text to summarize: {input_text}\n\nNow work through each step:"
            ),
        }
    elif task == "Classification":
        return {
            "zero_shot": (
                f"Classify the sentiment of this text as Positive, Negative, or Neutral:\n\n{input_text}"
            ),
            "few_shot": (
                "Examples:\n"
                "TEXT: I love this product! → Positive\n"
                "TEXT: This is terrible. → Negative\n"
                "TEXT: The package arrived. → Neutral\n\n"
                f"Classify: {input_text} →"
            ),
            "chain_of_thought": (
                "Analyze the sentiment step by step:\n"
                "1. Identify emotional words\n"
                "2. Assess overall tone\n"
                "3. Consider context and nuance\n"
                "4. Give final classification: Positive / Negative / Neutral\n\n"
                f"Text: {input_text}"
            ),
        }
    else:  # Q&A
        return {
            "zero_shot": f"Answer this question clearly and concisely:\n\n{input_text}",
            "few_shot": (
                "Q: What is the capital of France? A: Paris.\n"
                "Q: What is photosynthesis? A: The process by which plants convert sunlight to food.\n\n"
                f"Q: {input_text} A:"
            ),
            "chain_of_thought": (
                "Answer step by step:\n"
                "1. Understand what's being asked\n"
                "2. Recall relevant knowledge\n"
                "3. Structure a clear answer\n\n"
                f"Question: {input_text}\n\nWorking through it:"
            ),
        }


# ─── Claude API ───────────────────────────────────────────────────────────────
def call_claude(prompt: str, max_tokens: int = 400) -> tuple[str, float]:
    client = anthropic.Anthropic()
    t0 = time.time()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = round(time.time() - t0, 2)
    return msg.content[0].text, latency


def score_color(val: float) -> str:
    if val >= 65:  return "score-high"
    if val >= 35:  return "score-mid"
    return "score-low"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    task = st.selectbox("Task Type", ["Summarization", "Classification", "Q&A"])
    max_tokens = st.slider("Max Tokens", 100, 800, 300)

    st.markdown("---")
    st.markdown("### 📐 Metrics to Compute")
    use_bleu   = st.checkbox("BLEU Score",   value=True)
    use_rouge1  = st.checkbox("ROUGE-1",      value=True)
    use_rouge2  = st.checkbox("ROUGE-2",      value=True)
    use_rougel  = st.checkbox("ROUGE-L",      value=True)
    use_rubric  = st.checkbox("Custom Rubric",value=True)

    st.markdown("---")
    st.markdown("### 🧩 Prompt Variants")
    run_zs  = st.checkbox("Zero-Shot",        value=True)
    run_fs  = st.checkbox("Few-Shot",         value=True)
    run_cot = st.checkbox("Chain-of-Thought", value=True)


# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("# 🧪 AI Prompt Evaluation Dashboard")
st.markdown(
    "Compare **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought** prompts "
    "on real tasks using BLEU, ROUGE, and custom rubrics."
)

col1, col2 = st.columns([3, 2])

with col1:
    if task == "Summarization":
        default_input = (
            "Artificial intelligence (AI) is transforming every sector of the economy. "
            "From healthcare to finance, AI systems are automating decisions, accelerating "
            "drug discovery, detecting fraud, and personalizing customer experiences at scale. "
            "Large language models in particular have emerged as general-purpose tools capable "
            "of writing code, drafting legal documents, tutoring students, and generating creative "
            "content—all from natural language instructions. Despite rapid progress, AI systems "
            "still face significant limitations including hallucination, bias, and lack of "
            "common-sense reasoning, which researchers are actively working to address."
        )
        default_ref = (
            "AI is revolutionizing industries through automation and large language models, "
            "though challenges like hallucination and bias remain under active research."
        )
    elif task == "Classification":
        default_input = (
            "I've been using this software for three months and it's completely changed how "
            "I work. The interface is intuitive, support is responsive, and the new features "
            "in the latest update are exactly what I needed!"
        )
        default_ref = "Positive"
    else:
        default_input  = "What are the key differences between supervised and unsupervised machine learning?"
        default_ref    = (
            "Supervised learning uses labelled data to train models for classification or "
            "regression tasks, while unsupervised learning finds patterns in unlabelled data "
            "through clustering or dimensionality reduction."
        )

    input_text = st.text_area("📄 Input Text", value=default_input, height=160)

with col2:
    reference = st.text_area("✅ Reference / Ground Truth", value=default_ref, height=160)
    run_btn   = st.button("▶ Run Evaluation", use_container_width=True)


# ─── Run ─────────────────────────────────────────────────────────────────────
if run_btn:
    prompts = build_prompts(task, input_text)
    variant_map = {
        "zero_shot": ("Zero-Shot",         "tag-zeroshot", run_zs),
        "few_shot":  ("Few-Shot",          "tag-fewshot",  run_fs),
        "chain_of_thought": ("Chain-of-Thought", "tag-cot", run_cot),
    }

    results = {}
    active = [(k, v) for k, v in variant_map.items() if v[2]]

    if not active:
        st.warning("Select at least one prompt variant in the sidebar.")
        st.stop()

    progress = st.progress(0, text="Running evaluations…")

    for i, (key, (label, tag_cls, _)) in enumerate(active):
        with st.spinner(f"Calling Claude with {label} prompt…"):
            output, latency = call_claude(prompts[key], max_tokens)

        metrics: dict = {}
        if use_bleu:
            metrics["BLEU"]    = compute_bleu(reference, output)
        if use_rouge1:
            metrics["ROUGE-1"] = compute_rouge_n(reference, output, 1)["f1"]
        if use_rouge2:
            metrics["ROUGE-2"] = compute_rouge_n(reference, output, 2)["f1"]
        if use_rougel:
            metrics["ROUGE-L"] = compute_rouge_l(reference, output)
        if use_rubric:
            rub = custom_rubric_score(output, task, reference)
            metrics["Rubric Overall"] = rub["overall"]
            metrics["_rubric_detail"] = rub

        results[key] = {
            "label":   label,
            "tag_cls": tag_cls,
            "output":  output,
            "latency": latency,
            "metrics": metrics,
        }
        progress.progress((i + 1) / len(active))

    progress.empty()
    st.success("✅ Evaluation complete!")

    # ── Results Grid ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Results")

    cols = st.columns(len(results))
    for col, (key, res) in zip(cols, results.items()):
        with col:
            st.markdown(
                f'<span class="prompt-tag {res["tag_cls"]}">{res["label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"**⏱ Latency:** `{res['latency']}s`")

            for metric, val in res["metrics"].items():
                if metric.startswith("_"):
                    continue
                cls = score_color(val)
                st.markdown(
                    f'<div class="metric-card"><b>{metric}</b><br>'
                    f'<span class="{cls}" style="font-size:1.6rem">{val}</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("**Output:**")
            st.markdown(f'<div class="output-box">{res["output"]}</div>',
                        unsafe_allow_html=True)

    # ── Rubric Breakdown ──────────────────────────────────────────────────────
    if use_rubric:
        st.markdown("---")
        st.markdown("## 🎯 Custom Rubric Breakdown")
        rubric_cols = st.columns(len(results))
        for col, (key, res) in zip(rubric_cols, results.items()):
            with col:
                st.markdown(f"**{res['label']}**")
                detail = res["metrics"].get("_rubric_detail", {})
                for dim, score in detail.items():
                    if dim == "overall":
                        continue
                    bar_val = int(score)
                    st.markdown(f"`{dim.capitalize()}`")
                    st.progress(bar_val / 100, text=f"{score}%")

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Comparison Table")
    header = "| Metric | " + " | ".join(r["label"] for r in results.values()) + " |"
    sep    = "|--------|" + "|".join(["-----:"] * len(results)) + "|"
    all_metrics = [m for m in list(results.values())[0]["metrics"] if not m.startswith("_")]
    rows   = []
    for m in all_metrics:
        row = f"| **{m}** |"
        for res in results.values():
            row += f" {res['metrics'].get(m, '—')} |"
        rows.append(row)
    st.markdown("\n".join([header, sep] + rows))

    # ── Best Performer ────────────────────────────────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("## 🏆 Best Performing Variant")
        avg_scores = {}
        for key, res in results.items():
            vals = [v for k, v in res["metrics"].items()
                    if not k.startswith("_") and isinstance(v, float)]
            avg_scores[key] = sum(vals) / len(vals) if vals else 0
        best_key = max(avg_scores, key=avg_scores.get)
        best     = results[best_key]
        st.success(
            f"🥇 **{best['label']}** achieved the highest average score: "
            f"**{avg_scores[best_key]:.2f}**\n\n"
            "This variant produced the most consistent, high-quality output "
            "across all evaluated metrics."
        )
