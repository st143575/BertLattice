"""
Obtain the formal contexts (i.e. the conditional probabilities of attributes given objects)
version: 1.0
author: Chong Shen
Status: Only Animal-behavior dataset is supported.
"""


import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
from transformers import BertModel, AutoModel, AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, average_precision_score
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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

    # Load the ground-truth formal context.
    with open(f'{cache_dir}/fc_animal_true.csv', 'r', encoding='iso-8859-1'):
        fc_true = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)

    candidate_objects = fc_true.index.tolist()
    candidate_attributes = fc_true.columns.tolist()
    print(f"Number of objects: {len(candidate_objects)}")
    print(f"Number of attributes: {len(candidate_attributes)}")

    print(candidate_objects)
    print('\n\n')
    print(candidate_attributes)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)

    obj_token_ids = []
    for token in candidate_objects:
        obj_token_ids.append(tokenizer.convert_tokens_to_ids(token.lower()))
    print(obj_token_ids)

    probs_a = []
    for attr in candidate_attributes:
        sentence = f"The {tokenizer.mask_token} is an animal that can {attr}"
        token_ids = tokenizer.encode(sentence, return_tensors='pt')
        masked_index = token_ids[0].tolist().index(tokenizer.mask_token_id)

        with torch.no_grad():
            outputs = model(token_ids.to(device))
            prob = outputs.logits[0, masked_index][obj_token_ids].sigmoid()

        probs_a.append(prob)
    probs_a = torch.stack(probs_a)
    # print(probs_a)

    probs_b = []
    for attr in candidate_attributes:
        sentence = f"The {tokenizer.mask_token} is a type of animal that has the ability to {attr}"
        token_ids = tokenizer.encode(sentence, return_tensors='pt')
        masked_index = token_ids[0].tolist().index(tokenizer.mask_token_id)

        with torch.no_grad():
            outputs = model(token_ids.to(device))
            prob = outputs.logits[0, masked_index][obj_token_ids].sigmoid()

        probs_b.append(prob)

    probs_b = torch.stack(probs_b)
    # print(probs_b)

    probs_c = []
    for attr in candidate_attributes:
        sentence = f"An animal known as the {tokenizer.mask_token} has the ability to {attr}"
        token_ids = tokenizer.encode(sentence, return_tensors='pt')
        masked_index = token_ids[0].tolist().index(tokenizer.mask_token_id)

        with torch.no_grad():
            outputs = model(token_ids.to(device))
            prob = outputs.logits[0, masked_index][obj_token_ids].sigmoid()

        probs_c.append(prob)

    probs_c = torch.stack(probs_c)
    # print(probs_c)


    stacked_tensors = torch.stack([probs_a,probs_b,probs_c], dim=0)
    probs_max = torch.max(stacked_tensors, dim=0)[0]
    # print(probs_max)

    fc_pred_df_max = np.array(probs_max.t().cpu())
    fc_pred_df_max = pd.DataFrame(fc_pred_df_max,columns=candidate_attributes, index=pd.Index(candidate_objects))
    print(fc_pred_df_max)
    fc_pred_df_max.to_csv(f'{cache_dir}/fc_animal_pred_max.csv')


    animals, behaviors, popularity, animal_behavior, behavior_animal = load_data(input_dir, args.file_name)

    mrr_max = mean_rank(fc_pred_df_max, animal_behavior)
    h1_max = hit_k(fc_pred_df_max, animal_behavior, 1)
    h5_max = hit_k(fc_pred_df_max, animal_behavior, 5)
    h10_max = hit_k(fc_pred_df_max, animal_behavior, 10)
    print(f"MRR: {mrr_max}, Hit@1: {h1_max}, Hit@5: {h5_max}, Hit@10: {h10_max}")
    with open(f'{cache_dir}/metrics_animal_max.txt', 'w') as f:
        f.write(f"mrr: {mrr_max}\n")
        f.write(f"h1: {h1_max}\n")
        f.write(f"h5: {h5_max}\n")
        f.write(f"h10: {h10_max}\n")

    probs_mean = torch.mean(stacked_tensors, dim=0)
    # print(probs_mean)
    outmap_min = probs_mean.min(dim=0,keepdim=True)[0]
    outmap_max = probs_mean.max(dim=0,keepdim=True)[0]
    logprob_normalized = (probs_max - outmap_min) / (outmap_max - outmap_min)
    # print(logprob_normalized)

    fc_pred_df_normalized = np.array(logprob_normalized.t().cpu())
    fc_pred_df_normalized = pd.DataFrame(fc_pred_df_normalized,columns=candidate_attributes, index=pd.Index(candidate_objects))
    fc_pred_df_normalized = fc_pred_df_normalized.round(0).astype(float)
    # print(fc_pred_df_normalized)
    fc_pred_df_normalized.to_csv(f'{cache_dir}/fc_animal_pred_normalized.csv')

    f1 = f1_score(y_true=fc_true.astype(int).values, y_pred=fc_pred_df_normalized.astype(int).values,average="weighted" )
    mAP = average_precision_score(fc_true.astype(int).values, fc_pred_df_normalized.astype(int).values)
    print(f"F1: {f1}, mAP: {mAP}")
    with open(f'{cache_dir}/metrics_animal_normalized.txt', 'w') as f:
        f.write(f"f1: {f1}\n")
        f.write(f"mAP: {mAP}\n")


    probs_avg = torch.mean(stacked_tensors, dim=0)
    # print(probs_avg)
    fc_pred_df_avg = np.array(probs_avg.t().cpu())
    fc_pred_df_avg = pd.DataFrame(fc_pred_df_avg,columns=candidate_attributes, index=pd.Index(candidate_objects))
    fc_pred_df_avg.to_csv(f'{cache_dir}/fc_animal_pred_avg.csv')

    mrr_avg = mean_rank(fc_pred_df_avg, animal_behavior)
    h1_avg = hit_k(fc_pred_df_avg, animal_behavior, 1)
    h5_avg = hit_k(fc_pred_df_avg, animal_behavior, 5)
    h10_avg = hit_k(fc_pred_df_avg, animal_behavior, 10)
    print(f"MRR: {mrr_avg}, Hit@1: {h1_avg}, Hit@5: {h5_avg}, Hit@10: {h10_avg}")
    with open(f'{cache_dir}/metrics_animal_avg.txt', 'w') as f:
        f.write(f"mrr: {mrr_avg}\n")
        f.write(f"h1: {h1_avg}\n")
        f.write(f"h5: {h5_avg}\n")
        f.write(f"h10: {h10_avg}\n")


    plt.figure(figsize=(25, 35), dpi=200)

    meta_r = 0


    sns.set_theme(font_scale=1.)

    x_axis_labels = candidate_attributes
    y_axis_labels = candidate_objects


    ax = sns.heatmap(fc_pred_df_normalized, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cbar=False, linewidth=0.2, cmap='GnBu', square=False, annot= False )

    plt.tick_params(axis='y', which='major', colors='black', labelsize=24, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.tick_params(axis='x', which='major', colors='black', rotation=90, labelsize=24, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.savefig(f'{cache_dir}/Full-softmax-69-25-animal.pdf', bbox_inches='tight', pad_inches=0, dpi=1200)


    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(fc_pred_df_normalized.values)
    # print(X_embedded.shape)

    classes = ["Animals that do not fly", "Animals that fly"]
    scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=fc_pred_df_normalized['fly in the sky'])
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(f'{cache_dir}/animals_tsne_emb.pdf', bbox_inches='tight', pad_inches=0, dpi=800)


if __name__ == "__main__":
    main()