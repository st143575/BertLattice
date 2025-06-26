"""
Perform formal concept analysis to reconstruct the concept lattice of Animal-behavior dataset from formal context.
version: 1.0
author: Chong Shen
Status: Only Animal-behavior dataset is supported.
"""


import pandas as pd
import torch
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from fcapy.visualizer import LineVizNx
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform formal concept analysis to reconstruct the concept lattice of Animal-behavior dataset from formal context.')
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

    # Load ground-truth data.
    with open(f'{cache_dir}/fc_animal_true.csv', 'r', encoding='iso-8859-1'):
        fc_true = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)
    print(fc_true)
    
    # Convert int64 values to bool.
    fc_true = fc_true.astype(bool)
    
    # Create formal context.
    fc = FormalContext.from_pandas(fc_true)
    print(fc)

    # Reconstruct concept lattice.
    lattice = ConceptLattice.from_context(fc)
    print(len(lattice))
    print(lattice.top, lattice.bottom)

    # Visualize the concept lattice.
    fig, ax = plt.subplots(figsize=(12, 8),dpi=200)
    vsl = LineVizNx(node_label_font_size=12)
    vsl.draw_concept_lattice(lattice, ax=ax, flg_drop_bottom_concept=False, flg_node_indices=False)
    ax.set_title('"Animal-habit" concept lattice', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{cache_dir}/animal_behavior_concept_lattice.png')


if __name__ == "__main__":
    main()