# RAGBoost Examples

This directory contains practical code examples for using RAGBoost.

## Directory Structure

```
examples/
├── README.md                                      # This file
├── pipeline_examples.py                           # Basic Pipeline API examples
├── concurrent_first_then_multiturn_example.py     # Multi-turn with deduplication
├── batch_inference/                               # Low-level batch processing
│   ├── prepare_batch.py                          # Batch optimization
│   ├── sglang_inference.py                       # Inference with SGLang
│   └── analyze_results.py                        # Performance analysis
└── construct_rag_data/                           # Data retrieval
    ├── multihopRAG_faiss.py                      # FAISS vector search
    └── multihopRAG_bm25.py                       # BM25 lexical search
```

## Available Examples

### pipeline_examples.py

Basic Pipeline API examples:

1. Simple RAGBoost - Minimal example
2. With vs Without RAGBoost - Comparison
3. FAISS + RAGBoost - Vector search
4. Advanced Configuration
5. Custom Retriever
6. Existing Retrieval Results

**Run:**
```bash
python examples/pipeline_examples.py
```

### concurrent_first_then_multiturn_example.py

**Production multi-turn workflow with concurrent users:**

- **Phase 1**: Multiple users start conversations → Context optimization
- **Phase 2**: Users continue with follow-ups concurrently → Per-conversation deduplication
- Includes LLM response generation

**Run:**
```bash
# Start inference server first
python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --port 30000

# Run example
python examples/concurrent_first_then_multiturn_example.py
```

### batch_inference/

Low-level batch processing examples for production workflows:

- **prepare_batch.py** - Optimize retrieval results into batches
- **sglang_inference.py** - Run inference with SGLang server
- **analyze_results.py** - Analyze cache efficiency and performance

**Run:**
```bash
# Step 1: Optimize batch
python -m examples.batch_inference.prepare_batch \
 --context_path retrieval_results.jsonl \
 --output_path optimized_batch.jsonl

# Step 2: Run inference (after starting SGLang server)
python -m examples.batch_inference.sglang_inference \
 --model Qwen/Qwen2.5-32B-Instruct \
 --batch_path optimized_batch.jsonl \
 --corpus_path corpus.jsonl

# Step 3: Analyze results
python -m examples.batch_inference.analyze_results \
 --result_path results.jsonl \
 --output_dir analysis/
```

### construct_rag_data/

Data retrieval examples:

- **multihopRAG_bm25.py** - BM25 lexical search with Elasticsearch
- **multihopRAG_faiss.py** - FAISS vector search with embeddings

**Run:**
```bash
# BM25 retrieval
python -m examples.construct_rag_data.multihopRAG_bm25 \
 --query_path queries.jsonl \
 --corpus_path corpus.jsonl \
 --output_path retrieval_results.jsonl

# FAISS retrieval (requires embedding server running)
python -m examples.construct_rag_data.multihopRAG_faiss \
 --corpus_path corpus.jsonl \
 --query_path queries.jsonl \
 --output_path retrieval_results.jsonl
```

## Data Formats

All examples expect JSONL format:

**Queries:**
```json
{"qid": 0, "question": "What is machine learning?", "answers": ["ML is..."]}
```

**Corpus:**
```json
{"chunk_id": 0, "text": "Machine learning is a subset of AI...", "title": "ML"}
```

**Retrieval Results:**
```json
{"text": "What is machine learning?", "top_k_doc_id": ["42", "15", "8"]}
```

## Running Examples

From project root:
```bash
# Run pipeline examples
python examples/pipeline_examples.py

# Run batch processing (module syntax)
python -m examples.batch_inference.prepare_batch --help

# Run retrieval
python -m examples.construct_rag_data.multihopRAG_bm25 --help
```

## Requirements

Install RAGBoost first:
```bash
pip install -e .
```

**Optional (depending on example):**
- Elasticsearch (for BM25)
- FAISS: `pip install faiss-cpu` or `faiss-gpu`
- SGLang or vLLM (for inference)

## Troubleshooting

**Import Error:**
```bash
# Install RAGBoost first
cd /path/to/RAGBoost
pip install -e .
```

**Elasticsearch Connection Failed:**
```bash
# Start Elasticsearch with Docker
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.x
```

**FAISS Index Not Found:**
Build the index first or provide `corpus_data` to auto-create it.

---

For detailed guides, see [Documentation](../docs/README.md).
