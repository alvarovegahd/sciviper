import json
import shutil

# Parameters to set
NUM = 100 # Number of images you want in the new file
FILENAME = './descriptive_test.json' # Filename you want to trim

# Create a backup
shutil.copy(FILENAME, FILENAME+'.bak')

# Load the data
with open(FILENAME, 'r') as fp:
    data = json.load(fp)

# Get NUM elements
new_data = {}
for i, (key, val) in enumerate(data.items()):
    if i >= NUM: break
    new_data[key] = val

# Save the data back into the original filepath
with open(FILENAME, 'w') as fp:
    json.dump(new_data, fp, indent=4)