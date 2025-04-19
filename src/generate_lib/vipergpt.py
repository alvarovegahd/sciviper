from tqdm import tqdm
from PIL import Image
from main_simple_lib import get_code, execute_code

def generate_response(queries, model_path):
    for k in tqdm(queries):
        query = queries[k]['question'] # This will be a single question with instructions
        feedback = None
        if 'feedback' in queries[k]:
            feedback = queries[k]['feedback'] # This will be the feedback from the previous run
        image_path = queries[k]["figure_path"] # This will be the path to the figure associated with the above query

        code = get_code(query, f'results/code_{k}', extra_context=feedback, verbose=False)
        image = Image.open(image_path)
        response = execute_code(code, image, show_intermediate_steps=False)

        # Save the response
        queries[k]['response'] = response