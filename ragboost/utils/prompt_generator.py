PROMPT_TEMPLATE = '''
**Note**:
1. Please answer within 10 words or fewer. If the answer is Yes or No, just say Yes or No
2. Do not include any special characters like <answer> or <answer/>."

With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>

Please read the documents in the following ranking and answer the question:
<importance_ranking_section>
{importance_ranking}
</importance_ranking_section>

Please prioritize information from higher-ranked documents to answer the question.
'''

BASELINE_PROMPT='''
**Note**:
1. Please answer within 10 words or fewer. If the answer is Yes or No, just say Yes or No
2. Do not include any special characters like <answer> or <answer/>."

With provided related documents:
<documents_section>
{docs_section}
</documents_section>

Answer the question:
<question_section>
{question}
</question_section>
'''

def prompt_generator(chunk_id_text_dict, reordered_inputs):

    def format_docs(reordered_doc_ids):
        """Format documents section using doc IDs"""
        docs_section = ""
        for doc_id in reordered_doc_ids:
            # Try both the original doc_id and string version to support both int and string IDs
            if doc_id in chunk_id_text_dict:
                content = chunk_id_text_dict[doc_id]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            elif str(doc_id) in chunk_id_text_dict:
                content = chunk_id_text_dict[str(doc_id)]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            else:
                print(f"Warning: Doc_{doc_id} not found")
        return docs_section.strip()
        
    def format_importance(original_doc_order):
        """Format importance ranking"""
        return " > ".join([f"[Doc_{doc_id}]" for doc_id in original_doc_order])
    
    prompts = []
    qids = [i["qid"] for i in reordered_inputs]
    answers = [i["answer"] for i in reordered_inputs]
    

    for reordered_input in reordered_inputs:
        reordered_doc_ids = reordered_input['top_k_doc_id']
        original_doc_order = reordered_input['orig_top_k_doc_id']
        question = reordered_input['question']

        docs_section = format_docs(reordered_doc_ids)
        importance_ranking = format_importance(original_doc_order)

        prompt = PROMPT_TEMPLATE.format(
            docs_section=docs_section,
            question=question,
            importance_ranking=importance_ranking
        )

        prompts.append(prompt)
        
    return prompts, qids, answers

def prompt_generator_baseline(chunk_id_text_dict, inputs):

    def format_docs(doc_ids):
        """Format documents section using doc IDs"""
        docs_section = ""
        for doc_id in doc_ids:
            if doc_id in chunk_id_text_dict:
                content = chunk_id_text_dict[doc_id]
                docs_section += f"[Doc_{doc_id}] {content}\n\n"
            else:
                print(f"Warning: Doc_{doc_id} not found")
        return docs_section.strip()
    
    prompts = []
    qids = [i["qid"] for i in inputs]
    answers = [i["answer"] for i in inputs]
    

    for _input in inputs:
        original_doc_order = _input['top_k_doc_id']
        question = _input['text']

        docs_section = format_docs(original_doc_order)
        # importance_ranking = format_importance(original_doc_order)

        prompt = BASELINE_PROMPT.format(
            docs_section=docs_section,
            question=question
        )

        prompts.append(prompt)
        
    return prompts, qids, answers