import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from vision_models import QwenVLModel

def test_qwen_vl_model():
    """Test the QwenVL model with a local image and a question."""

    # Path to your test image
    image_path = "charxiv_12.jpg"  # Update this path to your local image file

    # Question to ask about the image
    question = "What is the Adversarial Accuracy for method A at Epoch 60?"

    # Load and preprocess the image
    if os.path.exists(image_path):
        print(f"Loading image from {image_path}")
        image = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image)

        print("Initializing QwenVL model...")
        qwen_model = QwenVLModel()

        print(f"Asking question: '{question}'")
        answer = qwen_model.forward(
            image=image_tensor,
            question=question,
            task='qa'
        )

        print(f"Answer: {answer}")

        caption = qwen_model.forward(
            image=image_tensor,
            task='caption'
        )

        print(f"Caption: {caption}")
    else:
        print(f"Error: Image file not found at {image_path}")

if __name__ == "__main__":
    test_qwen_vl_model()