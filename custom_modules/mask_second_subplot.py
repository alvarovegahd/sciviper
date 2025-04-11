from PIL import Image, ImageDraw

# Load the image
image = Image.open('scimodules/charxiv_0.jpg')

# Create a drawing context
draw = ImageDraw.Draw(image)
# Define the box coordinates
box = [(500, 0), (1030, 470)]
# Draw a white rectangle to mask the second subplot
draw.rectangle(box, fill='white')
# Save the masked image
image.save('custom_modules/images/masked_image.png')