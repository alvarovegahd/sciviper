#!/usr/bin/env python3
import os
import csv
import torch
import pandas as pd
from tqdm import tqdm

import sys
import os

os.chdir("../..")
sys.path.append(os.getcwd())

from main_simple_lib import load_image, get_code, execute_code

# define environ variable for the output directory
output_dir = "/scratch/cse692w25_class_root/cse692w25_class/jhsansom/results/specs"
code_output_dir = os.path.join(output_dir, "code")
spec_output_dir = os.path.join(output_dir, "spectrograms")
os.environ["SPECS_OUTPUT_DIR"] = spec_output_dir
os.makedirs(code_output_dir, exist_ok=True)
os.makedirs(spec_output_dir, exist_ok=True)

output_csv = os.path.join("experiments/spectrograms/results/sciviper_specs_results.csv")

def main():
    print("PyTorch CUDA version:", torch.version.cuda)

    spectrogram_jpg_dirpath = "/scratch/cse692w25_class_root/cse692w25_class/jhsansom/spectrograms"
    annotatedspecs_csv_path = "data/AnnotatedSpectrograms.csv"

    # Load CSV
    df = pd.read_csv(annotatedspecs_csv_path)
    df["path"] = [
        os.path.join(spectrogram_jpg_dirpath, fname) 
        for fname in df["specs"]
    ]

    # Filter to rows we want if needed
    df_nonan = df[df["n_syllables"].notna()].copy()

    # If you want to store new columns in the original DataFrame eventually
    # we can store them in parallel arrays first or use a dictionary approach.
    # But here we do immediate row-by-row appending, so let's do it in an append-based approach.

    # Write header row to the output CSV (overwrites if existing):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # include columns from the metadata that you want
        # plus columns for your new results
        writer.writerow(["index", "specs", "peak_freq_pred", "n_syllables_pred"])

    # Loop over each spectrogram
    for i in tqdm(range(len(df_nonan))):
        row = df_nonan.iloc[i]
        idx = row.name  # original index from the dataframe
        specs_name = row["specs"]
        img_path = row["path"]

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            # Append a row with None for results
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, specs_name, None, None])
            continue

        # Load the image
        im = load_image(img_path)

        # Query #1: peak frequency
        query1 = "What is the peak frequency indicated by the red dashed line in this spectrogram?"
        code1 = get_code(query1)
        try:
            result1 = execute_code(code1, im, show_intermediate_steps=False)
        except Exception as e:
            print(f"Error code1: {e}")
            result1 = None

        # Query #2: number of syllables
        query2 = "How many syllables are there visible in this spectrogram?"
        code2 = get_code(query2)
        try:
            result2 = execute_code(code2, im, show_intermediate_steps=False)
        except Exception as e:
            print(f"Error code2: {e}")
            result2 = None

        # Now append a line for the current row
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, specs_name, result1, result2])

        # save code as .txt file to output_dir/code dirpath
        with open(os.path.join(output_dir, "code", f"code1_{idx}.txt"), 'w') as f:
            f.write(code1[0])
        with open(os.path.join(output_dir, "code", f"code2_{idx}.txt"), 'w') as f:
            f.write(code2[0])

    print(f"\nAll done. Output appended to {output_csv}")

if __name__ == "__main__":
    main()
