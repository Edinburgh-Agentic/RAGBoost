# Multi-Turn RAG Conversations with Context Deduplication

RAGBoost provides efficient multi-turn conversation support with automatic context deduplication. This feature eliminates redundant document prefills by tracking retrieved documents across conversation turns and replacing duplicates with location hints.

## Key Features

- **Automatic Context Deduplication**: Identifies and removes duplicate documents across turns
- **Location Hints**: Guides LLM to previously seen content without re-prefilling
- **O(N) Algorithm**: Efficient set-based overlap detection
- **Per-Conversation State**: Maintains separate history for concurrent conversations
- **Comprehensive Statistics**: Track deduplication effectiveness in real-time

## Quick Start

### Basic Multi-Turn Conversation

```python
from ragboost.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_ragboost=True
)

# Process conversation turns
conversation_id = "user_session_123"

# Turn 1
result1 = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="What is machine learning?",
    top_k=5,
    enable_deduplication=True
)

# Turn 2 - deduplication will automatically kick in
result2 = pipeline.process_conversation_turn(
    conversation_id=conversation_id,
    query="How does it differ from deep learning?",
    top_k=5,
    enable_deduplication=True
)

# Check deduplication stats
print(f"Deduplicated: {result2['deduplication_stats']['num_deduplicated']} docs")
```

### With LLM Generation

```python
from ragboost.pipeline import RAGPipeline, InferenceConfig

pipeline = RAGPipeline(
    retriever="bm25",
    corpus_path="corpus.jsonl",
    use_ragboost=True,
    inference=InferenceConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="sglang",
        base_url="http://localhost:30000",
        max_tokens=200
    )
)

# Generate responses for each turn
result = pipeline.process_conversation_turn(
    conversation_id="conv_1",
    query="What are the best practices for ML?",
    top_k=5,
    enable_deduplication=True,
    generate_response=True  # Enable generation
)

print(f"Response: {result['response']}")
print(f"Generation time: {result['generation_metadata']['latency']}s")
```

## How It Works

### Algorithm Overview

1. **Retrieve Documents**: Get top-k relevant documents for current query
2. **Identify Overlaps**: Compare with conversation's document history using set intersection (O(N))
3. **Deduplicate**: Separate documents into:
   - **Novel**: Not seen before, include full content
   - **Deduplicated**: Already in history, use location hints
4. **Build Context**: Construct prompt with hints for duplicates, full content for novel docs
5. **Update History**: Add only novel documents to conversation history

### Example Flow

**Turn 1:**
```
Retrieved: {1, 2, 4}
History: ∅ (empty)
Novel: {1, 2, 4}

Context:
[Doc_1]: <full content>
[Doc_2]: <full content>
[Doc_4]: <full content>
```

**Turn 2:**
```
Retrieved: {1, 5, 2}
History: {1, 2, 4}
Overlap: {1, 2}
Novel: {5}

Context:
Please refer to [Doc_1] in the previous conversation.
[Doc_5]: <full content>
Please refer to [Doc_2] in the previous conversation.
```

**Turn 3:**
```
Retrieved: {3, 5, 6}
History: {1, 2, 4, 5}
Overlap: {5}
Novel: {3, 6}

Context:
[Doc_3]: <full content>
Please refer to [Doc_5] in the previous conversation.
[Doc_6]: <full content>
```

## API Reference

### `RAGPipeline.process_conversation_turn()`

Process a single conversation turn with context deduplication.

**Parameters:**
- `conversation_id` (str): Unique conversation identifier
- `query` (str or dict): User query
- `top_k` (int, optional): Number of documents to retrieve
- `enable_deduplication` (bool, default=True): Whether to apply deduplication
- `generate_response` (bool, default=False): Whether to generate LLM response
- `max_tokens` (int, optional): Maximum tokens for generation

**Returns:**
Dictionary with:
- `query`: Original query text
- `retrieved_docs`: List of all retrieved document IDs
- `novel_docs`: List of novel document IDs (not in history)
- `deduplicated_docs`: List of deduplicated document IDs
- `context`: Formatted context string with hints/content
- `deduplication_stats`: Statistics for this turn
  - `num_retrieved`: Total documents retrieved
  - `num_novel`: Documents with full content
  - `num_deduplicated`: Documents replaced with hints
  - `deduplication_rate`: Percentage deduplicated
- `conversation_stats`: Cumulative conversation statistics
- `response` (if generate_response=True): Generated text
- `generation_metadata` (if generate_response=True): Generation metrics

### `RAGPipeline.get_conversation_stats()`

Get deduplication statistics for conversations.

**Parameters:**
- `conversation_id` (str, optional): Specific conversation, or None for all

**Returns:**
Dictionary with conversation statistics:
- `conversation_id`: Conversation identifier
- `turn_count`: Number of turns processed
- `total_retrieved`: Total documents retrieved across all turns
- `total_novel`: Total novel documents processed
- `total_deduplicated`: Total documents deduplicated
- `deduplication_rate`: Overall deduplication percentage

### `RAGPipeline.reset_conversation()`

Reset a specific conversation's history.

**Parameters:**
- `conversation_id` (str): Conversation to reset

### `RAGPipeline.reset_all_conversations()`

Reset all conversation histories.

## Advanced Usage

### Baseline Comparison

Compare performance with and without deduplication:

```python
# With deduplication
result_dedup = pipeline.process_conversation_turn(
    conversation_id="conv_dedup",
    query="What is AI?",
    enable_deduplication=True
)

# Without deduplication (baseline)
result_baseline = pipeline.process_conversation_turn(
    conversation_id="conv_baseline",
    query="What is AI?",
    enable_deduplication=False  # No deduplication
)

# Compare
stats_dedup = pipeline.get_conversation_stats("conv_dedup")
stats_baseline = pipeline.get_conversation_stats("conv_baseline")

print(f"Deduplication saved: {stats_baseline['total_novel'] - stats_dedup['total_novel']} docs")
```

### Multiple Concurrent Conversations

Handle multiple users/sessions simultaneously:

```python
# User 1
result1 = pipeline.process_conversation_turn(
    conversation_id="user_1_session",
    query="Tell me about Python"
)

# User 2 (independent conversation)
result2 = pipeline.process_conversation_turn(
    conversation_id="user_2_session",
    query="Tell me about JavaScript"
)

# Get stats for all conversations
all_stats = pipeline.get_conversation_stats()
print(f"Total conversations: {all_stats['total_conversations']}")
print(f"Average deduplication: {all_stats['average_deduplication_rate']:.1%}")
```

### Custom Context Building

Access the `MultiTurnManager` directly for fine-grained control:

```python
from ragboost.pipeline import MultiTurnManager

manager = MultiTurnManager()

# Manual deduplication
context_str, novel_ids, stats = manager.deduplicate_context(
    conversation_id="conv_1",
    retrieved_doc_ids=[1, 2, 3],
    corpus_map=corpus_dict,
    enable_deduplication=True
)

# Get specific conversation stats
conv_stats = manager.get_conversation_stats("conv_1")
```

## Performance Benefits

### Computation Savings

Context deduplication provides:
- **Reduced Prefill**: Only novel documents need KV cache generation
- **Lower Latency**: Fewer tokens to process per turn
- **Memory Efficiency**: Smaller KV cache size
- **Cumulative Gains**: Savings increase with conversation length

### Typical Results

In multi-turn RAG benchmarks:
- **30-60% deduplication rate** in typical conversations
- **40-50% prefill reduction** over 5+ turns
- **Minimal overhead**: O(N) set operations add <1ms per turn

### Example Metrics

```
Conversation Statistics (5 turns):
- Total documents retrieved: 25 (5 per turn)
- Novel documents processed: 12
- Documents deduplicated: 13
- Deduplication rate: 52%
- Effective savings: 52% fewer prefills
```

## Quality Preservation

Location hints maintain answer quality by:
1. **Explicit References**: LLM knows where to find prior information
2. **Context Continuity**: Conversation flow preserved
3. **No Information Loss**: All content accessible via hints

Example prompt with hints:
```
With the chat history and the following context, answer the question:

Please refer to [Doc_1] in the previous conversation.

[Doc_5]: New document content here...

Please refer to [Doc_2] in the previous conversation.

Question: How does this relate to the previous information?
```

## Best Practices

1. **Use Consistent Conversation IDs**: Maintain same ID across related turns
2. **Enable Deduplication by Default**: Use baseline mode only for comparison
3. **Monitor Statistics**: Track deduplication rate to verify effectiveness
4. **Reset When Appropriate**: Call `reset_conversation()` when context changes
5. **Tune top_k**: Smaller top_k values may reduce deduplication opportunities

## Troubleshooting

### Low Deduplication Rate

If deduplication rate is unexpectedly low:
- Check if queries are diverse (may retrieve different documents)
- Verify conversation_id is consistent across turns
- Ensure corpus has sufficient document overlap

### Performance Issues

If experiencing slowness:
- The O(N) algorithm is very fast; issues likely elsewhere
- Check retrieval performance (BM25/FAISS latency)
- Monitor LLM inference time if using generation

### Reset Not Working

If history persists after reset:
- Verify correct conversation_id passed to reset
- Check that new turns use enable_deduplication=True
- Ensure pipeline instance is reused (not recreated)

## Examples

See `examples/multi_turn_example.py` for complete working examples:
- Basic multi-turn conversation
- Multi-turn with LLM generation
- Baseline vs. deduplication comparison
- Multiple concurrent conversations

## References

For implementation details, see:
- `ragboost/pipeline/multi_turn.py`: Core deduplication logic
- `ragboost/pipeline/rag_pipeline.py`: Pipeline integration
- `examples/multi_turn_example.py`: Usage examples
