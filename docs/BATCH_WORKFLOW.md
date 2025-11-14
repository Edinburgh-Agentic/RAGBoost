# Batch Processing Workflow

This guide explains the complete production workflow for RAGBoost's offline batch processing, from data retrieval to performance analysis.

## Overview

The workflow consists of three main stages:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Retrieval  │  ──▶ │ Optimization │  ──▶ │  Analysis   │
│             │      │  & Inference │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
```

## Stage 1: Data Retrieval

Retrieve relevant documents for your queries using BM25 or vector search.

### Option A: BM25 (Elasticsearch)

BM25 provides lexical matching based on term frequency and document frequency:

```bash
python -m examples.retrieval.run_bm25 \
  --query_path queries.jsonl \
  --corpus_path corpus.jsonl \
  --output_path retrieval_results.jsonl \
  --top_k 10
```

**Parameters:**
- `--query_path`: Path to JSONL file with queries (each line: `{"qid": 0, "text": "question"}`)
- `--corpus_path`: Path to JSONL file with documents (each line: `{"doc_id": 0, "text": "content"}`)
- `--output_path`: Where to save retrieval results
- `--top_k`: Number of documents to retrieve per query (default: 10)

**Output Format:**
```json
{
    "qid": 0,
    "text": "What is AI?",
    "top_k_doc_id": [42, 15, 8, 99, 7, ...]
}
```

### Option B: Vector Search (FAISS)

FAISS provides semantic similarity search using dense embeddings:

**Step 1: Start Embedding Server**
```bash
python -m sglang.launch_server \
  --model-path Alibaba-NLP/gte-Qwen2-7B-instruct \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

**Step 2: Build Index and Retrieve**
```bash
python -m examples.construct_rag_data.multihopRAG_faiss \
  --corpus_path corpus.jsonl \
  --query_path queries.jsonl \
  --output_path retrieval_results.jsonl \
  --index_path faiss.index \
  --embedding_model Alibaba-NLP/gte-Qwen2-7B-instruct \
  --top_k 10
```

**Parameters:**
- `--index_path`: Where to save/load FAISS index
- `--embedding_model`: Model for generating embeddings
- `--hnsw_m`: HNSW graph connectivity (default: 32)
- `--ef_construction`: Index build quality (default: 200)
- `--ef_search`: Search quality vs speed tradeoff (default: 128)

See `examples/construct_rag_data/multihopRAG_faiss.py` for full implementation.

## Stage 2: Batch Optimization and Inference

After retrieval, optimize contexts and run batch inference.

### Step 2a: Prepare Optimized Batch

Transform retrieval results into optimized batches for inference:

```bash
python -m examples.batch_inference.prepare_batch \
  --context_path retrieval_results.jsonl \
  --output_path optimized_batch.jsonl \
  --use_gpu \
  --alpha 0.001 \
  --linkage_method average
```

**Parameters:**
- `--context_path`: Path to retrieval results from Stage 1
- `--output_path`: Where to save optimized batch
- `--use_gpu`: Enable GPU acceleration for distance computation (recommended for >128 contexts)
- `--alpha`: Balance between similarity and diversity (default: 0.001, recommended: 0.01-0.0001)
- `--linkage_method`: Clustering method - `average`, `complete`, or `single` (default: `average`)

**What This Step Does:**

1. **Intra-Context Reordering**: Reorders documents within each query to maximize prefix overlap
   - Groups similar documents together
   - Reduces redundant KV cache computation within a single query

2. **Hierarchical Clustering**: Groups similar queries using document overlap similarity
   - Measures Jaccard similarity between retrieved document sets
   - Creates tree structure for efficient batching

3. **Inter-Context Scheduling**: Orders queries to maximize KV cache reuse across the batch
   - Traverses clustering tree to find optimal execution order
   - Ensures queries with similar contexts are processed consecutively

4. **Group Optimization**: Creates groups that share common document prefixes
   - Each group has a shared prefix of documents
   - Minimizes redundant prefix computation

**Output Format:**
```json
{
    "group_id": 0,
    "items": [
        {
            "qid": 42,
            "text": "What is AI?",
            "top_k_doc_id": [15, 8, 42, ...],
            "answer": ["AI is..."]
        }
    ],
    "group_score": 0.85
}
```

### Step 2b: Run Inference

Execute optimized batch with prefix caching enabled:

**Start SGLang Server:**
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B-Instruct \
  --port 30000 \
  --tp-size 4 \
  --enable-metrics \
  --schedule-policy lpm \
  --disable-radix-cache  # Use LPM cache policy for better batch processing
```

**Run Batch Inference:**
```bash
python -m examples.batch_inference.sglang_inference \
  --model Qwen/Qwen2.5-32B-Instruct \
  --batch_path optimized_batch.jsonl \
  --corpus_path corpus.jsonl \
  --output_path inference_results.jsonl \
  --api_url http://localhost:30000
```

**Parameters:**
- `--model`: Model identifier (must match server)
- `--batch_path`: Path to optimized batch from Step 2a
- `--corpus_path`: Path to original corpus with document texts
- `--output_path`: Where to save inference results
- `--api_url`: SGLang server endpoint

**Prompt Format:**

RAGBoost uses a structured prompt format that includes importance ranking:

```
With provided related documents:
<documents_section>
[Doc_1] Document content for doc_id 15...
[Doc_2] Document content for doc_id 8...
[Doc_3] Document content for doc_id 42...
</documents_section>

Answer the question:
<question_section>
What is artificial intelligence?
</question_section>

Please read the documents in the following ranking and answer the question:
<importance_ranking_section>
[Doc_1] > [Doc_2] > [Doc_3]
</importance_ranking_section>

Please prioritize information from higher-ranked documents to answer the question.
```

The importance ranking section preserves the original retrieval order while documents are physically reordered for cache efficiency.

**How It Works:**
1. Documents are reordered to maximize prefix sharing
2. Importance ranking hints guide the model to prioritize based on original retrieval scores
3. Prefix caching saves computation on shared document sequences
4. Accuracy is maintained through explicit ranking guidance

## Stage 3: Performance Analysis

Analyze inference metrics and cache efficiency:

```bash
python -m examples.batch_inference.analyze_results \
  --result_path inference_results.jsonl \
  --output_dir analysis_output/ \
  --compare_baseline  # Optional: compare with baseline without RAGBoost
```

**Generated Reports:**

1. **Cache Hit Rate Analysis**
   - Per-query and per-group hit rates
   - Prefix length distribution
   - Cache reuse patterns

2. **Latency Reduction**
   - Prefill latency comparison (RAGBoost vs baseline)
   - Time-to-first-token improvements
   - Total inference time savings

3. **Throughput Improvements**
   - Queries per second
   - Tokens per second
   - Effective batch size utilization

4. **Accuracy Metrics**
   - Answer quality comparison
   - Impact of reordering on correctness
   - Statistical significance tests

**Output Files:**
- `cache_analysis.json`: Detailed cache statistics
- `latency_report.json`: Timing measurements
- `accuracy_comparison.json`: Quality metrics
- `summary.txt`: Human-readable summary

## Data Formats

### Input: Queries (queries.jsonl)
```json
{"qid": 0, "text": "What is machine learning?"}
{"qid": 1, "text": "How does neural network work?"}
```

### Input: Corpus (corpus.jsonl)
```json
{"doc_id": 0, "text": "Machine learning is a subset of AI..."}
{"doc_id": 1, "text": "Neural networks are computing systems..."}
```

### Intermediate: Retrieval Results
```json
{
    "qid": 0,
    "text": "What is machine learning?",
    "top_k_doc_id": [42, 15, 8, 99, 7]
}
```

### Intermediate: Optimized Batch
```json
{
    "group_id": 0,
    "items": [
        {
            "qid": 0,
            "text": "What is machine learning?",
            "top_k_doc_id": [15, 8, 42, 99, 7],
            "answer": ["ML is..."]
        }
    ],
    "group_score": 0.85
}
```

### Output: Inference Results
```json
{
    "qid": 0,
    "question": "What is machine learning?",
    "prediction": "Machine learning is...",
    "ground_truth": ["ML is..."],
    "cache_hit_rate": 0.73,
    "prefill_latency_ms": 145.2
}
```

## Advanced Configuration

### GPU vs CPU Distance Computation

RAGBoost supports both GPU and CPU for similarity computation:

**When to use GPU:**
- Processing ≥ 128 contexts (17-72x speedup)
- Batch processing workloads
- High-throughput production systems

**When to use CPU:**
- Processing < 128 contexts
- GPU resources unavailable
- Latency requirements not critical

See [GPU vs CPU Benchmarking Results](../README.md#gpu-vs-cpu-benchmarking-results) for detailed performance comparison.

### Clustering Parameters

**Alpha (α)**: Controls similarity vs diversity tradeoff
- `α = 0.0001`: Very position-sensitive (maximum cache reuse)
- `α = 0.001`: Default value (balanced approach)
- `α = 0.01`: More diversity
- **Recommended range**: `0.01` to `0.0001` depending on your workload

**Linkage Method**: Determines how cluster distances are computed
- `average`: Average distance between all pairs (recommended)
- `complete`: Maximum distance between any pair (conservative)
- `single`: Minimum distance between any pair (aggressive)

### Custom Retrieval

Integrate your own retrieval system:

```python
from ragboost.retriever.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, queries, top_k=10):
        # Your retrieval logic
        results = []
        for qid, query in enumerate(queries):
            doc_ids = your_retrieval_function(query, top_k)
            results.append({
                "qid": qid,
                "text": query,
                "top_k_doc_id": doc_ids
            })
        return results
```

Then use in pipeline:
```python
from ragboost.pipeline import RAGPipeline

pipeline = RAGPipeline(
    retriever=CustomRetriever(),
    corpus_data=corpus
)
```

## Troubleshooting

### Common Issues

**1. Elasticsearch Connection Failed**
```
Error: Could not connect to Elasticsearch at localhost:9200
```
**Solution**: Start Elasticsearch service or check connection parameters.

**2. Out of Memory (GPU)**
```
CUDA out of memory
```
**Solution**: 
- Reduce batch size
- Use CPU for distance computation
- Use smaller model or quantization

**3. Low Cache Hit Rate**
```
Warning: Cache hit rate below 30%
```
**Solution**:
- Verify retrieval quality (documents should overlap across queries)
- Adjust alpha parameter (try lower values)
- Check that prefix caching is enabled in inference engine

**4. Slow Index Construction**
```
Index construction taking too long
```
**Solution**:
- Enable GPU acceleration (`--use_gpu`)
- Reduce number of queries for testing
- Check GPU utilization with `nvidia-smi`

## Performance Tips

1. **Batch Size**: Larger batches = better cache reuse (but more memory)
2. **Model Selection**: Smaller models = faster inference, similar cache benefits
3. **Top-K**: More documents per query = better potential for reordering benefits
4. **Clustering Quality**: Higher quality clustering = better cache efficiency
5. **Hardware**: GPU-enabled distance computation dramatically speeds up optimization

## Next Steps

- Try the Pipeline API for simplified workflow (see examples/pipeline_examples.py)
- Explore [Advanced Examples](../examples/pipeline_examples.py)

## See Also

- [Quick Start Guide](QUICK_START.md)
- [Pipeline API Reference](PIPELINE_API.md)
- [Examples Directory](../examples/)
