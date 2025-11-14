# RAGBoost Pipeline - Quick Start Guide

This guide shows you how to use RAGBoost with just a few lines of code using the new Pipeline API.

## Installation

```bash
git clone https://github.com/SecretSettler/RAGBoost.git
cd RAGBoost
pip install -e .
```

## Simplest Example

```python
from ragboost.pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    model="Qwen/Qwen2.5-7B-Instruct"
)

# Run on your queries
results = pipeline.run(queries=[
    "What is artificial intelligence?",
    "What is machine learning?"
])

# Save optimized batch
pipeline.save_results(results, "optimized_batch.jsonl")
```

That's it! The pipeline automatically:
1. ✅ Sets up BM25 retrieval with Elasticsearch
2. ✅ Retrieves relevant documents for your queries
3. ✅ Applies RAGBoost optimization (context reordering + scheduling)
4. ✅ Groups queries for maximum cache efficiency

## Without RAGBoost (Standard RAG)

Want to compare? Just disable RAGBoost:

```python
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_ragboost=False  # Standard RAG, no optimization
)

results = pipeline.run(queries=["What is AI?"])
```

## Using FAISS for Semantic Search

```python
pipeline = RAGPipeline(
    retriever="faiss",
    corpus_path="corpus.jsonl",
    index_path="faiss_index.faiss",
    embedding_model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    embedding_base_url="http://localhost:30000"
)

results = pipeline.run(queries=["Explain quantum computing"])
```

## Step-by-Step Control

Need more control? Run each step separately:

```python
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl"
)

# Step 1: Retrieve documents
retrieval_results = pipeline.retrieve(
    queries=["What is AI?"],
    top_k=20
)

# Step 2: Optimize with RAGBoost
optimized = pipeline.optimize(retrieval_results)

# Step 3: Save for inference
pipeline.save_results(
    {"optimized_batch": optimized["groups"]},
    "batch.jsonl"
)

# Step 4: Run inference (using existing scripts)
# python examples/batch_inference/sglang_inference.py \
#   --batch_path batch.jsonl \
#   --corpus_path corpus.jsonl
```

## Advanced Configuration

For full control, use configuration objects:

```python
from ragboost.pipeline import (
    RAGPipeline,
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig
)

pipeline = RAGPipeline(
    retriever=RetrieverConfig(
        retriever_type="bm25",
        top_k=20,
        corpus_path="corpus.jsonl"
    ),
    optimizer=OptimizerConfig(
        enabled=True,
        use_gpu=True,
        linkage_method="average"
    ),
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        temperature=0.1,
        max_tokens=512
    )
)
```

## Query Formats

The pipeline is flexible with query formats:

```python
# Simple string
pipeline.run(queries="What is AI?")

# List of strings
pipeline.run(queries=["What is AI?", "What is ML?"])

# Dictionaries with metadata
pipeline.run(queries=[
    {
        "qid": 1,
        "text": "What is AI?",
        "answer": ["Artificial Intelligence"]
    }
])
```

## Output Format

Results are organized into optimized groups:

```json
{
  "optimized_batch": [
    {
      "group_id": 0,
      "group_size": 3,
      "group_score": 0.85,
      "items": [
        {
          "qid": 1,
          "text": "What is AI?",
          "top_k_doc_id": [5, 12, 3],
          "orig_top_k_doc_id": [12, 5, 3]
        }
      ]
    }
  ],
  "metadata": {
    "num_queries": 15,
    "num_groups": 5,
    "total_time": 2.34
  }
}
```

## Complete Example

```python
from ragboost.pipeline import RAGPipeline

# Setup pipeline
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="data/corpus.jsonl",
    use_ragboost=True,
    use_gpu=True,
    top_k=20
)

# Your queries
queries = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is deep learning?"
]

# Run pipeline
results = pipeline.run(queries)

# Save results
pipeline.save_results(results, "output/batch.jsonl")

# Check statistics
print(f"✅ Processed {results['metadata']['num_queries']} queries")
print(f"📊 Created {results['metadata']['num_groups']} groups")
print(f"⏱️  Total time: {results['metadata']['total_time']:.2f}s")
```

## Next Steps

- 📖 Read the [full API documentation](PIPELINE_API.md)
- 🎯 See [complete examples](../examples/pipeline_examples.py)
- 🔧 Check out [batch workflow guide](BATCH_WORKFLOW.md)
- 📚 Review the [main README](../README.md) for architecture details

## Key Benefits

✅ **Simple API**: 3-5 lines of code for complete RAG pipeline  
✅ **Flexible**: Toggle RAGBoost on/off, swap retrievers easily  
✅ **Modular**: Run full pipeline or individual steps  
✅ **Compatible**: Integrates with existing RAGBoost workflows  
✅ **Fast**: GPU acceleration, efficient caching, optimal scheduling  

## Questions?

- Open an [issue](https://github.com/SecretSettler/RAGBoost/issues)
- Review [examples](../examples/)
