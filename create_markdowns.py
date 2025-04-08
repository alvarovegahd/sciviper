import os
import glob
import json

# Set this to the folder with the results you want to analyze
FOLDER = './results/results_desc_val'
GROUND_TRUTH_JSON = './data/descriptive_val.json' # Either descriptive or reasoning based on your purpose
IMAGE_DATA_FOLDER = '../../images' # Relative to FOLDER

markdown_template = '''
# Image {image_q_id}

![Test]({full_image_path})

# Question and Results

- **Question:** {question}
- **Correct Answer:** {correct_answer}
- **Full ViperGPT Return Value:** {return_value}
- **Extracted Return Value:** {extracted_value}
- **Correctness:** {correctness}

# Code

{code}
'''.strip()

# Taken from src/constants.py
DESCRIPTIVE_GRADING_QMAP = {
    1: "What is the title of the plot?",
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?",
    10: "How many lines are there?",
    11: "Do any lines intersect?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?",
    17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?",
    19: "What is the number of subplots?",
}

# Get relevant results json files
search_pattern = os.path.join(FOLDER, "gen-*.json")
gen_file = glob.glob(search_pattern)[0]
with open(gen_file, 'r') as fp:
    gen_data = json.load(fp)
search_pattern = os.path.join(FOLDER, "scores-*.json")
scores_file = glob.glob(search_pattern)[0]
with open(scores_file, 'r') as fp:
    scores_data = json.load(fp)
is_descriptive = ('descriptive' in scores_file)
with open(GROUND_TRUTH_JSON, 'r') as fp:
    gt_data = json.load(fp)

# Loop through each item in scores
for image_q_id, data in scores_data.items():
    # Get the question and answer
    gen_dp = gen_data[image_q_id]
    if is_descriptive:
        image_id, q_id = image_q_id.split('_')
        subplot_loc = gt_data[image_id]['subplot_loc']
        if isinstance(subplot_loc, list):
            [M, N] = gt_data[image_id]['subplot_loc']
            question = f'For the subplot at row {M} and column {N}, '
        else:
            question = f'For the {subplot_loc}, '
        question += DESCRIPTIVE_GRADING_QMAP[gen_dp['qid']]
        correct_answer = gt_data[image_id]['answers'][int(q_id)]
    else:
        image_id = image_q_id
        question = gen_dp['raw_question']
        correct_answer = gt_data[image_id]['answer']

    # Get the image path
    full_image_path = os.path.join(IMAGE_DATA_FOLDER, f'{image_id}.jpg')

    # Get raw model response
    return_value = gen_dp['response']

    # Get the extracted value
    extracted_value = data['extracted_answer']
    if extracted_value is None:
        extracted_value = 'None'
    elif extracted_value == '':
        extracted_value = 'Empty string: ""'

    # Get the correctness
    score = data['score']
    if score == 1:
        correctness = 'Correct'
    else:
        correctness = 'Wrong'

    # Get the code
    code_filepath = os.path.join(FOLDER, f'code_{image_q_id}.txt')
    with open(code_filepath, 'r') as fp:
        code = fp.read()

    # Get the full md_text
    md_text = markdown_template.format(
        image_q_id = image_q_id,
        full_image_path = full_image_path,
        question = question,
        correct_answer = correct_answer,
        return_value = return_value,
        extracted_value = extracted_value,
        correctness = correctness,
        code = code
    )

    # Write to a file
    save_to = os.path.join(FOLDER, f'{image_q_id}.md')
    with open(save_to, 'w') as fp:
        fp.write(md_text)