# Add these paths
results_folder = 'results2/results_qwen72b'
image_q_pair = '88_3'
output_filepath = f'analysis_{image_q_pair}.ipynb'

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# Load the code
code_path = f'{results_folder}/code_{image_q_pair}.txt'
with open(code_path, 'r') as fp:
    code = fp.read()

# Load the image
image_num = image_q_pair.split('_')[0]
image_path = f'images/{image_num}.jpg'

# Set up some basic cells
cell1 = f'''
from main_simple_lib import *
code = """
{code}
"""

code_for_syntax = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0)
code = ast.unparse(ast.parse(code))
code_for_syntax_2 = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=True, start_line=0)

code = (code, syntax_2)

im = load_image("{image_path}")
execute_code(code, im, show_intermediate_steps=True)
'''

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add a code cell
nb['cells'] = [nbf.v4.new_code_cell(cell1)]

# Execute the notebook
try:
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
except Exception as e:
    print(e)

# Save the notebook
nbf.write(nb, output_filepath)