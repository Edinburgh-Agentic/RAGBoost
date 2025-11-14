# RAGBoost Batch Inference Examples

This directory contains examples for optimized batch inference with RAGBoost.

## Overview

RAGBoost reorders retrieved contexts to maximize prefix sharing, which significantly improves KV cache efficiency during batch inference.

## Files

- **`prepare_batch.py`** - Optimize and group retrieval results for batch inference
- **`sglang_inference.py`** - Run batch inference with SGLang using the prepared batch
- **`analyze_results.py`** - Analyze batch preparation output and generate statistics (optional)
- **`sample_data.jsonl`** - Example input data for testing

## Quick Start

### Step 1: Prepare Batch

```bash
python prepare_batch.py \
    --context_path sample_data.jsonl \
    --output_path batch_output.jsonl
```

**Options:**
- `--context_path`: Path to retrieval results (JSONL format)
- `--output_path`: Where to save the prepared batch
- `--use_gpu`: Use GPU for distance computation (optional)
- `--linkage_method`: Clustering method - `average`, `complete`, or `single` (default: `average`)
- `--alpha`: Weight for position differences in distance calculation (default: 0.005)

### Step 2: Run Inference

```bash
# Start SGLang server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --port 30000 \
    --tp-size 1 \
    --enable-metrics

# Run batch inference
python sglang_inference.py \
    --model Qwen/Qwen3-4B-Instruct \
    --batch_path batch_output.jsonl \
    --corpus_path <path-to-corpus.jsonl>
```

**Options:**
- `--model`: Model name/path
- `--batch_path`: Path to prepared batch from Step 1
- `--corpus_path`: Path to corpus JSONL with document texts

### Step 3: Analyze Results (Optional)

```bash
python analyze_results.py batch_output.jsonl
```

## Input Data Format

The input JSONL file should have this format:

```json
{
    "qid": 0,
    "text": "What is the capital of France?",
    "answer": ["Paris"],
    "top_k_doc_id": [123, 456, 789]
}
```

**Required fields:**
- `qid`: Unique query identifier
- `text`: Query text
- `answer`: List of acceptable answers
- `top_k_doc_id`: List of retrieved document IDs

## Output Format

The prepared batch contains grouped queries optimized for inference:

```json
{
    "group_id": 0,
    "group_size": 5,
    "group_score": 0.85,
    "items": [
        {
            "qid": 1,
            "text": "Query text",
            "answer": ["Answer"],
            "top_k_doc_id": [101, 102, 103],
            "orig_top_k_doc_id": [103, 101, 102]
        }
    ]
}
```

## What It Does

1. **Intra-context reordering**: Reorders documents within each query for better relevance
2. **Inter-context clustering**: Groups similar queries using hierarchical clustering
3. **Tree-based scheduling**: Organizes execution order to maximize KV cache hits
4. **Optimized batching**: Creates batches that share maximum prefix tokens

## Performance

Typical improvements with RAGBoost:
- 2-5x reduction in KV cache usage
- 1.5-3x faster inference throughput
- Up to 80% token reuse across queries

## Example Workflow

```bash
# 1. Prepare batch from your retrieval results
python prepare_batch.py \
    --context_path /path/to/retrieval_results.jsonl \
    --output_path optimized_batch.jsonl \
    --use_gpu

# 2. Start inference server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct \
    --port 30000 \
    --tp-size 4 \
    --enable-metrics \
    --schedule-policy lpm

# 3. Run optimized batch inference
python sglang_inference.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --batch_path optimized_batch.jsonl \
    --corpus_path /path/to/corpus.jsonl

# 4. (Optional) Analyze the optimization
python analyze_results.py optimized_batch.jsonl
```

## See Also

- Main README: `../../README.md`
- Context Index: `../../ragboost/context_index/`
- Context Ordering: `../../ragboost/context_ordering/`