"""
OpenAI API based RAG Evaluation Script for InfiniteBench

Metrics:
- Accuracy: Exact Match (EM), F1 Score, Precision, Recall
- Performance: Latency (TTFT, total), Throughput (tokens/sec), Cost estimation

Prompt Types:
- baseline: Original document order, no importance ranking (uses BASELINE_PROMPT)
- optimized: Reordered documents + importance ranking (uses PROMPT_TEMPLATE)
- optimized_no_ranking: Reordered documents, no ranking (uses PROMPT_TEMPLATE_NO_RANKING)
- default: Simple built-in prompt (no special formatting)

Usage:
    # Baseline evaluation (original retrieval order)
    python eval_openai.py --retrieval_results infinibench_bm25_results_top15.jsonl \
                          --corpus infinibench_corpus.jsonl \
                          --prompt_type baseline \
                          --base_url http://localhost:8000/v1 \
                          --model your-model

    # Optimized with importance ranking
    python eval_openai.py --optimized_results optimized.json \
                          --corpus infinibench_corpus.jsonl \
                          --prompt_type optimized \
                          --base_url http://localhost:8000/v1 \
                          --model your-model

    # Optimized without ranking (ablation)
    python eval_openai.py --optimized_results optimized.json \
                          --corpus infinibench_corpus.jsonl \
                          --prompt_type optimized_no_ranking \
                          --base_url http://localhost:8000/v1 \
                          --model your-model
"""

import json
import time
import argparse
import re
import string
from collections import Counter
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

# Add ragboost to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ragboost.utils.prompt_generator import (
    prompt_generator,
    prompt_generator_baseline,
    prompt_generator_optimized_no_ranking
)


# ==================== Evaluation Metrics ====================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Calculate F1, Precision, and Recall scores."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    if not prediction_tokens or not ground_truth_tokens:
        return ZERO_METRIC
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_answer(prediction: str, gold_answers: List[str]) -> Dict[str, float]:
    """Evaluate prediction against multiple gold answers, taking best scores."""
    max_em, max_f1, max_prec, max_recall = 0.0, 0.0, 0.0, 0.0

    for gold in gold_answers:
        em = float(exact_match_score(prediction, gold))
        f1, prec, recall = f1_score(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    return {
        'em': max_em,
        'f1': max_f1,
        'precision': max_prec,
        'recall': max_recall
    }


# ==================== Data Loading ====================

def load_corpus(corpus_path: str) -> Dict[int, str]:
    """Load corpus and create chunk_id -> text mapping."""
    corpus_map = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            chunk_id = doc.get('chunk_id', doc.get('document_id', doc.get('_id')))
            text = doc.get('text', doc.get('content', ''))
            corpus_map[chunk_id] = text
    return corpus_map


def load_retrieval_results(results_path: str) -> List[Dict[str, Any]]:
    """Load retrieval results with queries and retrieved doc IDs."""
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            results.append(data)
    return results


def load_optimized_results(optimized_path: str) -> List[Dict[str, Any]]:
    """Load optimized_results from RAGPipeline.optimize() output.
    
    Supports two formats:
    1. JSONL: One group per line (e.g., planning_output.jsonl)
    2. JSON: Single object with {"groups": [...]} structure
    """
    results = []
    
    # Try JSONL format first (one group per line)
    with open(optimized_path, 'r') as f:
        first_line = f.readline().strip()
        
    # Check if it's JSONL (each line is a complete JSON object with "items")
    try:
        first_obj = json.loads(first_line)
        if "items" in first_obj:
            # JSONL format: each line is a group with "items"
            with open(optimized_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    group = json.loads(line)
                    for item in group.get("items", []):
                        results.append(item)
            return results
    except json.JSONDecodeError:
        pass
    
    # Fall back to single JSON file with {"groups": [...]}
    with open(optimized_path, 'r') as f:
        optimized = json.load(f)
    
    for group in optimized.get("groups", []):
        for item in group.get("items", []):
            results.append(item)
    
    return results


# ==================== Prompt Generation ====================

def build_rag_prompt(question: str, doc_texts: List[str], max_context_chars: int = 50000) -> str:
    """Build a RAG prompt with retrieved documents as context."""
    # Combine documents with truncation if needed
    context_parts = []
    total_chars = 0
    
    for i, text in enumerate(doc_texts):
        doc_header = f"[Document {i+1}]\n{text}\n"
        if total_chars + len(doc_header) > max_context_chars:
            break
        context_parts.append(doc_header)
        total_chars += len(doc_header)
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on the following documents, answer the question concisely.

{context}

Question: {question}

Answer:"""
    
    return prompt


# ==================== OpenAI API Inference ====================

class OpenAIEvaluator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url
        )
        
        # Cost per 1M tokens (approximate, update as needed)
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
    
    def generate(self, prompt: str, max_tokens: int = 32768) -> Dict[str, Any]:
        """Generate response and collect timing metrics."""
        start_time = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions concisely based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0
            )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            print(response)
            
            generated_text = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return {
                "success": True,
                "generated_text": generated_text,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "success": False,
                "error": str(e),
                "latency": end_time - start_time,
                "generated_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = self.pricing.get(self.model, {"input": 1.0, "output": 3.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


# ==================== Main Evaluation ====================

def run_evaluation(
    retrieval_results: List[Dict[str, Any]],
    corpus_map: Dict[int, str],
    evaluator: OpenAIEvaluator,
    max_samples: int = None,
    max_workers: int = 16,
    max_tokens: int = 32768,
    output_path: str = None,
    prompt_type: str = "default"
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Args:
        retrieval_results: List of query results with doc IDs
        corpus_map: Mapping from chunk_id to document text
        evaluator: OpenAI evaluator instance
        max_samples: Maximum samples to evaluate
        max_workers: Number of parallel workers
        max_tokens: Max tokens to generate per response
        output_path: Path to save results
        prompt_type: One of 'baseline', 'optimized', 'optimized_no_ranking', or 'default'
    """
    
    if max_samples:
        retrieval_results = retrieval_results[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(retrieval_results)} samples")
    print(f"Model: {evaluator.model}")
    print(f"Prompt type: {prompt_type}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # Generate prompts based on prompt_type
    if prompt_type in ['baseline', 'optimized', 'optimized_no_ranking']:
        # Use the prompt generators from ragboost
        if prompt_type == 'baseline':
            prompts, qids, answers_list = prompt_generator_baseline(
                chunk_id_text_dict=corpus_map,
                inputs=retrieval_results,
                apply_template=False
            )
        elif prompt_type == 'optimized':
            prompts, qids, answers_list = prompt_generator(
                chunk_id_text_dict=corpus_map,
                reordered_inputs=retrieval_results,
                apply_template=False
            )
        elif prompt_type == 'optimized_no_ranking':
            prompts, qids, answers_list = prompt_generator_optimized_no_ranking(
                chunk_id_text_dict=corpus_map,
                reordered_inputs=retrieval_results,
                apply_template=False
            )
        
        # Build eval_items from generator output
        eval_items = []
        for i, (prompt, qid, answer) in enumerate(zip(prompts, qids, answers_list)):
            # Handle string answers
            if isinstance(answer, str):
                answer = [answer]
            
            item = retrieval_results[i]
            question = item.get('text', item.get('question', ''))
            doc_ids = item.get('top_k_doc_id', [])
            # print(prompt)
            # exit()
            
            eval_items.append({
                'qid': qid,
                'question': question,
                'prompt': prompt,
                'gold_answers': answer,
                'doc_ids': doc_ids
            })
    else:
        # Default: use built-in build_rag_prompt
        eval_items = []
        for item in retrieval_results:
            qid = item.get('qid', item.get('id', 0))
            question = item.get('text', item.get('question', ''))
            doc_ids = item.get('top_k_doc_id', [])
            answers = item.get('answer', item.get('answers', []))
            
            # Handle string answers
            if isinstance(answers, str):
                answers = [answers]
            
            # Get document texts
            doc_texts = []
            for doc_id in doc_ids:
                if doc_id in corpus_map:
                    doc_texts.append(corpus_map[doc_id])
            
            prompt = build_rag_prompt(question, doc_texts)
            
            eval_items.append({
                'qid': qid,
                'question': question,
                'prompt': prompt,
                'gold_answers': answers,
                'doc_ids': doc_ids
            })
    
    # Run inference
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    
    start_time = time.perf_counter()
    
    def process_item(item):
        response = evaluator.generate(item['prompt'], max_tokens=max_tokens)
        return {**item, **response}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in eval_items}
        
        for future in tqdm(as_completed(futures), total=len(eval_items), desc="Evaluating"):
            result = future.result()
            results.append(result)
            
            if result['success']:
                total_input_tokens += result['input_tokens']
                total_output_tokens += result['output_tokens']
                total_latency += result['latency']
    
    end_time = time.perf_counter()
    wall_clock_time = end_time - start_time
    
    # Calculate accuracy metrics
    accuracy_metrics = {'em': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    successful_count = 0
    
    detailed_results = []
    
    for result in results:
        if result['success'] and result['gold_answers']:
            # Extract answer after </think>\n\n if present (for reasoning models)
            generated_text = result['generated_text']
            if '</think>\n\n' in generated_text:
                generated_text = generated_text.split('</think>\n\n')[-1]
            scores = evaluate_answer(generated_text, result['gold_answers'])
            accuracy_metrics['em'] += scores['em']
            accuracy_metrics['f1'] += scores['f1']
            accuracy_metrics['precision'] += scores['precision']
            accuracy_metrics['recall'] += scores['recall']
            successful_count += 1
            
            detailed_results.append({
                'qid': result['qid'],
                'question': result['question'],
                'prediction': generated_text,
                'gold_answers': result['gold_answers'],
                'em': scores['em'],
                'f1': scores['f1'],
                'latency': result['latency'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens']
            })
    
    # Normalize accuracy metrics
    if successful_count > 0:
        for key in accuracy_metrics:
            accuracy_metrics[key] = accuracy_metrics[key] / successful_count * 100  # Convert to percentage
    
    # Calculate performance metrics
    performance_metrics = {
        'total_samples': len(retrieval_results),
        'successful_samples': successful_count,
        'failed_samples': len(retrieval_results) - successful_count,
        'wall_clock_time_sec': wall_clock_time,
        'total_latency_sec': total_latency,
        'avg_latency_sec': total_latency / successful_count if successful_count > 0 else 0,
        'throughput_samples_per_sec': successful_count / wall_clock_time if wall_clock_time > 0 else 0,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'output_tokens_per_sec': total_output_tokens / total_latency if total_latency > 0 else 0,
        'estimated_cost_usd': evaluator.estimate_cost(total_input_tokens, total_output_tokens)
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print("\n📊 ACCURACY METRICS:")
    print(f"  Exact Match (EM):  {accuracy_metrics['em']:.2f}%")
    print(f"  F1 Score:          {accuracy_metrics['f1']:.2f}%")
    print(f"  Precision:         {accuracy_metrics['precision']:.2f}%")
    print(f"  Recall:            {accuracy_metrics['recall']:.2f}%")
    
    print("\n⚡ PERFORMANCE METRICS:")
    print(f"  Total samples:     {performance_metrics['total_samples']}")
    print(f"  Successful:        {performance_metrics['successful_samples']}")
    print(f"  Failed:            {performance_metrics['failed_samples']}")
    print(f"  Wall clock time:   {performance_metrics['wall_clock_time_sec']:.2f}s")
    print(f"  Avg latency:       {performance_metrics['avg_latency_sec']:.3f}s")
    print(f"  Throughput:        {performance_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  Output tokens/sec: {performance_metrics['output_tokens_per_sec']:.2f}")
    
    print("\n💰 TOKEN & COST METRICS:")
    print(f"  Input tokens:      {performance_metrics['total_input_tokens']:,}")
    print(f"  Output tokens:     {performance_metrics['total_output_tokens']:,}")
    print(f"  Total tokens:      {performance_metrics['total_tokens']:,}")
    print(f"  Estimated cost:    ${performance_metrics['estimated_cost_usd']:.4f}")
    
    print(f"\n{'='*60}\n")
    
    # Save detailed results
    if output_path:
        output_data = {
            'accuracy_metrics': accuracy_metrics,
            'performance_metrics': performance_metrics,
            'model': evaluator.model,
            'detailed_results': detailed_results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return {
        'accuracy': accuracy_metrics,
        'performance': performance_metrics,
        'detailed_results': detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI API based RAG Evaluation")
    parser.add_argument("--retrieval_results", type=str, default=None,
                        help="Path to retrieval results JSONL file")
    parser.add_argument("--optimized_results", type=str, default=None,
                        help="Path to optimized_results JSON file (from RAGPipeline.optimize())")
    parser.add_argument("--corpus", type=str, default="infinibench_corpus.jsonl",
                        help="Path to corpus JSONL file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name to use")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Custom API base URL (for compatible APIs like vLLM, SGLang, etc.)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel API requests")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum tokens to generate per response")
    parser.add_argument("--output", type=str, default="eval_results_reorder.json",
                        help="Path to save detailed results")
    parser.add_argument("--prompt_type", type=str, default="default",
                        choices=["baseline", "optimized", "optimized_no_ranking", "default"],
                        help="Prompt generation strategy: "
                             "baseline (original doc order, no ranking), "
                             "optimized (reordered + importance ranking), "
                             "optimized_no_ranking (reordered, no ranking), "
                             "default (simple built-in prompt)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.retrieval_results and not args.optimized_results:
        parser.error("Either --retrieval_results or --optimized_results must be provided")
    
    # Validate prompt_type compatibility
    if args.prompt_type in ['optimized', 'optimized_no_ranking'] and not args.optimized_results:
        parser.error(f"--prompt_type={args.prompt_type} requires --optimized_results input")
    if args.prompt_type == 'baseline' and args.optimized_results and not args.retrieval_results:
        print("Warning: baseline prompt uses original retrieval results format. "
              "Using optimized_results but treating top_k_doc_id as original order.")
    
    # Load data
    print("Loading corpus...")
    corpus_map = load_corpus(args.corpus)
    print(f"Loaded {len(corpus_map)} documents")
    
    print("Loading queries...")
    if args.optimized_results:
        retrieval_results = load_optimized_results(args.optimized_results)
        print(f"Loaded {len(retrieval_results)} queries from optimized_results")
    else:
        retrieval_results = load_retrieval_results(args.retrieval_results)
        print(f"Loaded {len(retrieval_results)} queries from retrieval results")
    
    # Initialize evaluator
    evaluator = OpenAIEvaluator(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Run evaluation
    results = run_evaluation(
        retrieval_results=retrieval_results,
        corpus_map=corpus_map,
        evaluator=evaluator,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        output_path=args.output,
        prompt_type=args.prompt_type
    )
    
    return results


if __name__ == "__main__":
    main()
