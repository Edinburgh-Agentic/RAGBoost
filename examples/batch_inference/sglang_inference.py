from ragboost.utils.eval_metrics import eval_answer
from ragboost.utils.prompt_generator import prompt_generator
from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text
from transformers import AutoTokenizer

import json
import argparse
import ast
import re
import time
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference with optimized context ordering.")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-32B")
    parser.add_argument('--batch_path', type=str, required=True, help='Path to the JSONL file containing prepared batch.')
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the JSONL file containing corpus.')
    parser.add_argument('--result_file', type=str, default='results.jsonl', help='Path to save inference results.')
    args = add_common_sglang_args_and_parse(parser)
    return args

args = parse_args()

with open(args.batch_path, 'r') as f:
    group_inputs = [json.loads(line) for line in f]

with open(args.corpus_path, 'r') as f:
    chunks = [json.loads(line) for line in f]

chunk_id_text_dict = {chunk['chunk_id']: chunk['text'] for chunk in chunks}
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Generate prompts for all groups
all_prompts = []
all_qids = []
all_answers = []

for group_input in group_inputs:
    prompts, qids, answers = prompt_generator(chunk_id_text_dict, group_input['items'])
    all_prompts.extend(prompts)
    all_qids.extend(qids)
    all_answers.extend(answers)

INVALID = -9999999

def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def evaluate(prompts, answers):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Construct arguments
    arguments = []
    for prompt in prompts:
        arguments.append({"prompt": prompt})

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def multi_document_qa(s, prompt):
        s += "You are a helpful assistant that answers questions using your own knowledge and relevant documents.\n"
        s += sgl.user_begin()
        s += prompt
        s += sgl.user_end()
        s += "/no_think"
        s += sgl.assistant(sgl.gen("answer", max_tokens=32))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    em = []
    f1 = []
    precision = []
    recall = []

    tic = time.perf_counter()
    states = multi_document_qa.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True
    )
    latency = time.perf_counter() - tic
    # print(states)
    # print([s["answer"] for s in states])
    for s, label in zip(states, answers):
        try:
            answer = s["answer"].split('\n\n')[-1]
            print(f"Label: {label}")
            print(f"Answer: {answer}")
            curr_em = []
            curr_f1 = []
            curr_precision = []
            curr_recall = []
            for lab in label:
                em_score, f1_score, prec, rec = eval_answer(answer, lab)
                curr_em.append(em_score)
                curr_f1.append(f1_score)
                curr_precision.append(prec)
                curr_recall.append(rec)
            em.append(max(curr_em))
            f1.append(max(curr_f1))
            precision.append(max(curr_precision))
            recall.append(max(curr_recall))
        except Exception as e:
            continue
    
    final_em = np.mean(em)
    final_f1 = np.mean(f1) 
    final_precision = np.mean(precision)
    final_recall = np.mean(recall)

    # Compute speed
    # num_output_tokens = sum(
    #     s.get_meta_info("answer")["completion_tokens"] for s in states
    # )
    # output_throughput = num_output_tokens / latency
    print(f"Exact Match: {final_em:.3f}")
    print(f"F1 Score: {final_f1:.3f}")
    print(f"Precision: {final_precision:.3f}")
    print(f"Recall: {final_recall:.3f}")
    print(f"Latency: {latency:.3f} s")
    # print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "Exact Match": round(final_em, 3),
            "F1 Score": round(final_f1, 3),
            "Precision": round(final_precision, 3),
            "Recall": round(final_recall, 3),
            # "output_throughput": round(output_throughput, 3),
            # "num_requests": args.num_questions,
            "other": {
                # "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")

evaluate(all_prompts, all_answers)