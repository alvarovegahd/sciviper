import json
from copy import deepcopy
from constants import REASONING_RESP_INST, REASONING_GRADING_PREFIX, \
                REASONING_GRADING_INST
import os

def get_reasoning_result_gpt(client, prompt, max_retries=10):
    curr_retries = 0
    max_tokens = 256
    while curr_retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4o-2024-05-13",
                response_format={"type": "json_object"},
                n=1,
                max_tokens=max_tokens,
                temperature=0,
                top_p=1,
                seed=42,
            ).choices[0].message.content
            content = json.loads(response)
            ext, scr = content['extracted_answer'], content['score']
            break
        except Exception as e:
            print(f"Error: {e}")
            # increase the max_tokens if the response is too long
            if 'Unterminated string starting at' in str(e):
                if max_tokens >= 1024:
                    print(f"Failed to get response for prompt: {prompt}")
                    ext, scr = 'Failed to parse response', -1
                    break
                else:
                    max_tokens = min(1024, max_tokens * 2) # double the max_tokens
                    print(f"Retrying with max_tokens: {max_tokens}")
            # otherwise, retry the request
            curr_retries += 1
    # if failed to get response, return dummy data
    if curr_retries == max_retries:
        print(f"Failed to get response for prompt: {prompt}")
        ext, scr = 'Failed to parse response', -1
    return ext, scr

def get_number_instruction(answer):
    base = answer.split('.')
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst

def build_reasoning_grading_queries(input, resp):
    queries = {}
    for _, data in input.items():
        figure_id = str(data['figure_id'])
        # question without instruction, response
        query, response = resp[figure_id]['raw_question'], resp[figure_id]['response']
        # get query for answer type (inst_category), then
        # populate the query with the question, ground truth, and response
        grading_query = REASONING_GRADING_PREFIX + deepcopy(\
            REASONING_GRADING_INST[data['inst_category']])\
            .replace("<|question|>", query)\
            .replace("<|ground_truth|>", data['answer'])\
            .replace("<|response|>", str(response))
        query = {
            'figure_id': figure_id,
            'grading_query': grading_query,
        }
        queries[figure_id] = query
    return queries

def build_reasoning_queries(data, image_dir):
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        inst_category = d['inst_category']
        # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d['query'])
        # 4: number-in-general -> need to specify the number of decimal places
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(d['query'], \
                                        get_number_instruction(d['answer']))
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")
        query = {
            'figure_id': d['figure_id'], # figure_id
            'figure_path': figure_path, # figure_path
            'inst_category': inst_category, # instruction category
            'raw_question': d['query'], # question @@@ without @@@ instruction
            'question': question, # question with instruction
        }
        queries[d['figure_id']] = query
    return queries

def build_reasoning_queries_with_feedbacks(data, image_dir, include_code=False):
    import glob
    FOLDER = '/scratch/cse692w25_class_root/cse692w25_class/jhsansom/results/reas_val'
    search_pattern = os.path.join(FOLDER, "gen-*.json")
    gen_file = glob.glob(search_pattern)[0]
    with open(gen_file, 'r') as fp:
        gen_data = json.load(fp)

    if include_code:
        feedback_template = (
            "Below is the previous code and the corresponding error message. Please review each class's attributes and properties carefully. "
            "Do **not** generate the same code again.\n\n"
            "— Previous Code —\n"
            "{code}\n\n"
            "— Error Message —\n"
            "{full_return}"
            "This error indicates that your solution is trying to use features or attributes that don't exist in the provided classes.\n"
            "Please:\n1. Carefully re-examine the class documentation at the beginning of the prompt\n2. Pay special attention to the available attributes and methods of each class\n3. Do NOT repeat the same approach that caused the error\n4. Develop a new solution that only uses documented attributes and methods\n5. Test your logic against the error message to ensure you've resolved the issue\n"
            "Remember that the classes may have different capabilities than you initially assumed. Your new solution should work within the constraints of what's actually available in the API."
        )

    else:
        feedback_template = (
            "Below is the error message from a previous run. Please carefully re-examine the class documentation at the beginning of the prompt. Test your logic against the error message to ensure you've resolved the issue.\n"
            "— Error Message —\n"
            "{full_return}"
        )


    code = None
    full_return = None

    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        inst_category = d['inst_category']
        # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d['query'])
        # 4: number-in-general -> need to specify the number of decimal places
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(d['query'], \
                                        get_number_instruction(d['answer']))
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")

        image_q_id = f"{d['figure_id']}"
        gen_dp = gen_data[image_q_id]
        full_return = gen_dp['response']
        if full_return is None or 'Exception' not in str(full_return):
            continue

        if include_code:
            code_filepath = os.path.join(FOLDER, f'code_{image_q_id}.txt')
            with open(code_filepath, 'r') as fp:
                code = fp.read()

        query = {
            'figure_id': d['figure_id'], # figure_id
            'figure_path': figure_path, # figure_path
            'inst_category': inst_category, # instruction category
            'raw_question': d['query'], # question @@@ without @@@ instruction
            'question': question, # question with instruction
            'feedback': feedback_template.format(code=code, full_return=full_return) if include_code else feedback_template.format(full_return=full_return), # feedback from previous run
        }
        queries[d['figure_id']] = query
    return queries
