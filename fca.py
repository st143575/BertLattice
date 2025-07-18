"""
Perform formal concept analysis to reconstruct the concept lattice of Animal-behavior dataset from formal context.
Visualize the concept lattice.
Status: Only Animal-behavior dataset is supported.
"""


import time
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from fcapy.visualizer import LineVizNx
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

ms_color = ('live on land', 'hunt insects')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Perform formal concept analysis to reconstruct the concept lattice of Animal-behavior dataset from formal context.')
    parser.add_argument('-i', '--input_dir', type=str, default='/./data', help='Path to input data directory')
    parser.add_argument('-fn', '--file_name', type=str, default='animal_behavior_file.txt', help='Input file name')
    parser.add_argument('-o', '--output_dir', type=str, default='./cache', help='Path to output data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./cache/models--bert-base-uncased', help='Path to BERT model')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='GPU-ID')
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()

def node_clr_label_func(c_i, lattice, ms_color=ms_color):
    lbl = LineVizNx.concept_lattice_label_func(
        c_i, 
        lattice, 
        flg_new_extent_count_prefix=False, 
        flg_new_intent_count_prefix=False
    )
    for s in ms_color:
        lbl = lbl.replace(s, '')
    lbl = lbl.replace(',', '')
    if c_i not in [2, 9, 13, 18, 24, 7, 8, 10, 19, 22, 27, 28, 29] and 'Yak' not in lbl:
        lbl=''
    if c_i == 0:
        lbl= 'Top'
    if c_i == 63:
        lbl= 'Bottom'
    return lbl

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
    # with open(f'{cache_dir}/fc_animal_true.csv', 'r', encoding='iso-8859-1'):
    fc_true = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)
    print(f"fc_true:\n{fc_true}\n")
    
    # Convert int64 values to bool.
    fc_true = fc_true.astype(bool)
    print(f"fc_true after boolean conversion:\n{fc_true}\n")
    
    # Create formal context.
    fc = FormalContext.from_pandas(fc_true[:10])
    print(f"Formal context:\n{fc}\n")

    # Reconstruct concept lattice.
    print("Reconstructing concept lattice...")
    start_time = time.time()
    lattice = ConceptLattice.from_context(fc)
    end_time = time.time()
    print("Concept lattice reconstructed.")
    print(f"Length of concept lattice: {len(lattice)}")
    print(f"Top concept: {lattice.top}")
    print(f"Bottom concept: {lattice.bottom}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")

    # Visualize the concept lattice.
    fig, ax = plt.subplots(figsize=(12, 8),dpi=200)
    vsl = LineVizNx(node_label_font_size=12)
    vsl.draw_concept_lattice(lattice, ax=ax, flg_drop_bottom_concept=False, flg_node_indices=False)
    ax.set_title('"Animal-habit" concept lattice', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{cache_dir}/animal_behavior_concept_lattice.png')

    # ms_color = ('Toad', 'Goldfish')
    lattice_clr = deepcopy(lattice)
    clr_map = {frozenset(ms_color): 'khaki', frozenset({ms_color[0]}): 'navy', frozenset({ms_color[1]}): 'forestgreen',}
    node_color_legend = {
        clr_map[frozenset(ms_color)]: 'live on land and hunt insects',
        clr_map[frozenset({ms_color[0]})]: 'live on land',
        clr_map[frozenset({ms_color[1]})]: 'hunt insects',
    }
    viz = LineVizNx(node_label_font_size=15)
    node_color = [clr_map.get(frozenset(c.intent) & frozenset(ms_color), viz.node_color) for c in lattice_clr]
    
    fig, ax = plt.subplots(figsize=(10,7))
    viz.draw_concept_lattice(
        lattice_clr, 
        ax=ax,
        flg_node_indices=False,
        # flg_new_intent_count_prefix=False, 
        # flg_new_extent_count_prefix=False,
        flg_drop_bottom_concept=True,
        flg_drop_top_concept=True,
        node_color=node_color,
        node_color_legend=node_color_legend,
        node_label_func=node_clr_label_func,
    )

    leg = plt.legend(title='', title_fontproperties={'size': '14',}, fontsize=15, bbox_to_anchor=(0.12,0.08), loc='lower left')
    leg._legend_box.align = "left"
    plt.tight_layout()
    plt.savefig(f'{cache_dir}/animal_behavior_concept_lattice_clr.png', bbox_inches='tight', pad_inches=0, dpi=800)


if __name__ == "__main__":
    main()