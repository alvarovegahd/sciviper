import torch
from datasets import load_dataset
import requests
import json

def query_ollama(model_name, prompt, origin="http://localhost:11434"):
    # Set up the API endpoint
    url = "http://localhost:11434/v1/chat/completions"
    
    # Configure the headers
    headers = {
        "Content-Type": "application/json",
        "Origin": origin
    }
    
    # Prepare the request payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()['choices'][0]['message']['content']

# Loops through charxiv data and tests model on each one
def test_on_charxiv():
    ds = load_dataset("princeton-nlp/CharXiv")
    for i, example in enumerate(ds['validation']):

        image = example['image']
        question = example['reasoning_q']
        answer = example['reasoning_a']

        print(f'Running example {i}')
        print(f'Question: {question}')
        print(f'Answer: {answer}')

        result = test_on_image(image, question)
        print(f'Model result = {result}')
        raise Exception

# Tests the model on one single (image, query) input
def test_on_image(im, query):
    from main_simple_lib import load_image, get_code, execute_code

    #show_single_image(im)
    code = get_code(query)
    result = execute_code(code, im, show_intermediate_steps=False)
    return result


if __name__ == '__main__':
    print("PyTorch CUDA version:", torch.version.cuda)
    print(f'PyTorch device count: {torch.cuda.device_count()}')

    # Test code
    #query = 'How many muffins can each kid have for it to be fair?'
    #image_path = 'kids_muffins.jpg'
    #print(test_on_image(image_path, query))
    #test_on_charxiv()

    response = query_ollama('qwen2.5-coder', 'Why is the sky blue?')
    print(response)