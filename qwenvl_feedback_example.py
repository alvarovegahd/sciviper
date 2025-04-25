from configs import config
import json

import os
import glob
FOLDER = '/scratch/cse692w25_class_root/cse692w25_class/jhsansom/results/desc_qwen72b'
search_pattern = os.path.join(FOLDER, "gen-*.json")
gen_file = glob.glob(search_pattern)[0]
with open(gen_file, 'r') as fp:
    gen_data = json.load(fp)


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

image_q_id = "36_1"
prompt = """For the subplot at row 0 and column 0, what is the rightmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""" # TODO: make this dynamic

gen_dp = gen_data[image_q_id]
full_return = gen_dp['response']

code_filepath = os.path.join(FOLDER, f'code_{image_q_id}.txt')
with open(code_filepath, 'r') as fp:
    code = fp.read()

with open(config.codex.prompt) as f:
    base_prompt = f.read().strip()
extra_context = feedback_template.format(code=code, full_return=full_return)
# extra_context = ""

input_type = "image"
extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", prompt).replace('INSERT_TYPE_HERE', input_type).replace('EXTRA_CONTEXT_HERE', extra_context)

######################################################################################

import base64
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://34.31.189.7:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# # Uncomment this if you want to give the image input
# image_path = "images/36.jpg"
# with open(image_path, "rb") as f:
#     encoded_image = base64.b64encode(f.read())
# encoded_image_text = encoded_image.decode("utf-8")
# base64_qwen = f"data:image;base64,{encoded_image_text}"

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                # # Uncomment this if you want to give the image input
                # { "type": "image_url", "image_url": {"url": base64_qwen}},
                {"type": "text", "text": extended_prompt},
            ],
        },
    ],
)
print("Chat response:", chat_response.choices[0].message.content)
