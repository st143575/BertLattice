"""
Preprocess the datasets.
This includes creating the ground-truth formal context (i.e. the conditional probability matrix) for the datasets.
version: 1.0
author: Chong Shen
Status: Only Animal-behavior dataset is supported.
"""


import os
import re
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from utils import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run BERT conditional probability on Animal-behavior dataset.')
    parser.add_argument('-i', '--input_dir', type=str, default='/mount/studenten/arbeitsdaten-studenten1/shencg/BertLattice/Reimplement/data', help='Path to input data directory')
    parser.add_argument('-fn', '--file_name', type=str, default='animal_behavior.txt', help='Input file name')
    parser.add_argument('-o', '--output_dir', type=str, default='/mount/studenten/arbeitsdaten-studenten1/shencg/BertLattice/Reimplement/output', help='Path to output data directory')
    parser.add_argument('-m', '--model_path', type=str, default='/mount/studenten/arbeitsdaten-studenten1/shencg/BertLattice/cache/models--bert-base-uncased', help='BERT model name')
    parser.add_argument('--cache_dir', type=str, default='/mount/studenten/arbeitsdaten-studenten1/shencg/BertLattice/cache', help='Path to cache directory')
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
    animals, behaviors, popularity, animal_behavior, behavior_animal = load_data(input_dir, args.file_name)

    # Load popular animals.
    popular_animals = set()
    with open(f'{input_dir}/popular_animals.txt', 'r', encoding='iso-8859-1') as l:
        animal_lines = l.readlines()
        for line in animal_lines:
            popular_animals.add(line.strip())
    print(f'{len(popular_animals)} popular animals:\n{popular_animals}\n')

    # Sort popular animals alphabetically as objects.
    objects = sorted(list(popular_animals), key=lambda s: s.split('_', 1)[0])
    print(f'{len(objects)} objects:\n{objects}\n')

    # Load behaviors.
    behaviors = set()
    with open(f'{input_dir}/behaviors.txt', 'r', encoding='iso-8859-1') as l:
        behavior_lines = l.readlines()
        for line in behavior_lines:
            behaviors.add(line.strip())
    print(f'{len(behaviors)} behaviors:\n{behaviors}\n')

    # Sort behaviors alphabetically as attributes.
    attributes = sorted(list(behaviors), key=lambda s: s.split('_', 1)[0])
    print(f'{len(attributes)} attributes:\n{attributes}\n')

    true_fc = torch.zeros(len(objects),len(attributes)).long() 
    print(true_fc.shape)
    print(true_fc)
    for i in range(len(objects)):
        for j in range(len(attributes)):
            if attributes[j] in animal_behavior[objects[i]]:
                true_fc[i,j] = 1
    df = np.array(true_fc)
    df_true = pd.DataFrame(df,columns=attributes, index=pd.Index(objects))
    print(df_true)
    df_true.to_csv(f'{cache_dir}/fc_animal_true.csv')


if __name__ == "__main__":
    main()