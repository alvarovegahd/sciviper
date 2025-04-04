import torch
from datasets import load_dataset

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


def test_on_image(im, query):
    from main_simple_lib import load_image, get_code, execute_code

    #show_single_image(im)
    code = get_code(query)
    return execute_code(code, im, show_intermediate_steps=False)


if __name__ == '__main__':
    print("PyTorch CUDA version:", torch.version.cuda)
    print(f'PyTorch device count: {torch.cuda.device_count()}')

    # Test code
    #query = 'How many muffins can each kid have for it to be fair?'
    #image_path = 'kids_muffins.jpg'
    #print(test_on_image(image_path, query))
    test_on_charxiv()