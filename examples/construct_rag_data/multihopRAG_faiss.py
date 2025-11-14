from ragboost.retriever import FAISSRetriever
from ragboost.utils.tools import chunk_documents
from ragboost.offline.corpus_analyer import CorpusTokenAnalyzer
from datasets import load_dataset
import asyncio
import argparse
import json


parser = argparse.ArgumentParser(description="Process queries and build index")
parser.add_argument("--gen_model_path", type=str, default="Qwen/Qwen3-32B", help="Path to the model")
parser.add_argument("--embedding_model_path", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct", help="Path to the model")
parser.add_argument("--corpus_path", type=str, default="mulhoprag_corpus.jsonl", help="Path to the corpus you want to store, ending with .jsonl")
parser.add_argument("--index_path", type=str, default="mulhoprag_corpus_index.faiss", help="Path to save or load the index")
parser.add_argument("--query_path", type=str, default="mulhoprag_queries.jsonl", help="Path to the queries you want to store, ending with .jsonl")
parser.add_argument("--output_path", type=str, default="mulhoprag_faiss_results_top20.jsonl", help="Path to the output results")
parser.add_argument("--port", type=int, default=30000, help="Port for the API")
parser.add_argument("--topk", type=int, default=20, help="Number of top documents to retrieve")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
parser.add_argument("--batch_delay", type=float, default=5.0, help="Delay between batches in seconds")
args = parser.parse_args()
base_url = f"http://localhost:{args.port}/v1"


corpus_origin = load_dataset("yixuantt/MultiHopRAG", "corpus")
qa = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

# === Step 2: Create corpus.jsonl with title included in text ===
corpus = []
chunk_id = 0
seen_paragraph_prefixes = set()

for entry in corpus_origin:
    for p in corpus_origin[entry]:
        paragraph_text = p["body"]
        
        # Calculate the first 50% of characters
        prefix_length = len(paragraph_text) // 2
        paragraph_prefix = paragraph_text[:prefix_length]
        
        # Skip if we've seen this prefix before
        if paragraph_prefix in seen_paragraph_prefixes:
            continue
        
        # Add this prefix to our seen set
        seen_paragraph_prefixes.add(paragraph_prefix)

all_chunks = chunk_documents(list(seen_paragraph_prefixes), out_file=args.corpus_path)
        
# === Step 3: Create queries.jsonl ===
queries = []
query_id = 0
for entry in qa:
    for q in qa[entry]:
        queries.append({
            "id": query_id,
            "question": q["query"],
            "answers": [q["answer"]] + q.get("answer_aliases", [])
        })
        query_id += 1

with open(args.query_path, "w") as f:
    for query in queries:
        f.write(json.dumps(query) + "\n")

# === Step 4: Analyze corpus and save context lengths ===
analyzer = CorpusTokenAnalyzer(
    model_name_or_tokenizer=args.gen_model_path,
    corpus_data=all_chunks,
    output_corpus_file=f"{args.corpus_path[:-6]}_with_context_lengths.jsonl"
)

detailed_chunks = analyzer.chunks_data

retriever = FAISSRetriever(
    model_path=args.embedding_model_path,
    base_url=base_url,
    index_path=args.index_path
)

# Example usage:
# `corpus_path` is checked for existence to determine if indexing should run.
# If the corpus file exists, it will be indexed. If not, the program assumes
# a pre-built index exists at `index_path` and proceeds to search.

asyncio.run(retriever.run_retrieval(
    corpus_file=f"{args.corpus_path[:-6]}_with_context_lengths.jsonl",
    queries_file=args.query_path,
    output_file=args.output_path,
    top_k=args.topk,
    batch_size=args.batch_size,
    batch_delay=args.batch_delay
))