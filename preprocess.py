"""
Preprocess the datasets.
This includes creating the ground-truth formal context (i.e. the conditional probability matrix) for the datasets.
Status: Only Animal-behavior dataset is supported.
"""


import torch
import pandas as pd
import argparse
from pathlib import Path
from utils import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess the datasets.')
    parser.add_argument('-i', '--input_dir', type=str, default='./data', help='Path to input data directory')
    parser.add_argument('-fn', '--file_name', type=str, default='animal_behavior_file.txt', help='Input file name')
    parser.add_argument('-o', '--output_dir', type=str, default='./cache', help='Path to output data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./cache/models--bert-base-uncased', help='Path to BERT model')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='GPU-ID')
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name()

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    cache_dir = Path(args.cache_dir)
    print("=============CONFIG==============")
    print(f"Using device: {device} ({device_name})")
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Model path:", model_path)
    print("Cache directory:", cache_dir)
    print("=================================\n")
    
    # Load data.
    animals, behaviors, popularity, animal2behaviors, behavior2animals, animal_behavior_file = load_data(input_dir, args.file_name)
    print(f"{len(animals)} animals:\n{animals}\n")
    print(f"{len(behaviors)} behaviors:\n{behaviors}\n")
    print(f"{len(popularity)} popularity:\n{popularity}\n")
    print(f"animal2behaviors: (length: {len(animal2behaviors)})\n{animal2behaviors}\n")
    print(f"behavior2animals: (length: {len(behavior2animals)})\n{behavior2animals}\n")

    # Create behavior_animal_file by swapping the first two columns of animal_behavior_file.
    # Write behavior_animal_file.txt to cache_dir.
    with open(f'{cache_dir}/behavior_animal_file.txt', 'w', encoding='iso-8859-1') as l:
        for line in animal_behavior_file:
            l.write(line.split(',')[1] + ',' + line.split(',')[0] + ',' + line.split(',')[2])

    # Write animals.txt to cache_dir.
    with open(f'{cache_dir}/animals.txt', 'w', encoding='iso-8859-1') as l:
        for animal in animals:
            l.write(animal + '\n')

    # Write behaviors.txt to cache_dir.
    with open(f'{cache_dir}/behaviors.txt', 'w', encoding='iso-8859-1') as l:
        for behavior in behaviors:
            l.write(behavior + '\n')

    # Write behavior2animal.csv to cache_dir.
    with open(f'{cache_dir}/behavior2animal.csv', 'w', encoding='iso-8859-1') as l:
        for behavior in behavior2animals:
            l.write(behavior + ',' + ','.join(behavior2animals[behavior]) + '\n')

    # Sort animals alphabetically as objects.
    objects = sorted(list(animals), key=lambda s: s.split('_', 1)[0])
    print(f'{len(objects)} objects:\n{objects}\n')

    # Sort behaviors alphabetically as attributes.
    attributes = sorted(list(behaviors), key=lambda s: s.split('_', 1)[0])
    print(f'{len(attributes)} attributes:\n{attributes}\n')

    # Create ground-truth formal context for animals-behaviors.
    fc_true = torch.zeros(len(objects),len(attributes)).long() 
    print(f"fc_true:\n{fc_true}\nshape: {fc_true.shape}")
    for i in range(len(objects)):
        for j in range(len(attributes)):
            if attributes[j] in animal2behaviors[objects[i]]:
                fc_true[i,j] = 1
    fc_animal_true_df = pd.DataFrame(fc_true, columns=attributes, index=pd.Index(objects))
    print(f"fc_animal_true_df:\n{fc_animal_true_df}\n")
    fc_animal_true_df.to_csv(f'{cache_dir}/fc_animal_true.csv')


if __name__ == "__main__":
    main()