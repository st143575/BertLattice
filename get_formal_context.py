"""
Obtain the formal contexts (i.e. the conditional probability matrix of attributes given objects)
Status: Only Animal-behavior dataset is supported.
"""


import yaml
import torch
import random
import argparse
import numpy as np
import pandas as pd
from utils import *
from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
from transformers import BertModel, AutoModel, AutoTokenizer, AutoModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run BERT conditional probability on Animal-behavior dataset.')
    parser.add_argument('-i', '--input_dir', type=str, default='./data', help='Path to input data directory')
    parser.add_argument('-fn', '--file_name', type=str, default='animal_behavior_file.txt', help='Input file name')
    parser.add_argument('-o', '--output_dir', type=str, default='./cache', help='Path to output data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./cache/models--bert-base-uncased', help='Path to BERT model')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='GPU-ID')
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()


def encode_candidate_objs_attrs(candidate_objects, candidate_attributes, tokenizer, model, device):
    assert candidate_objects is not None and candidate_attributes is not None
    obj_embs = []
    for obj in candidate_objects:
        obj_embs.append(get_sentence_embedding(obj, tokenizer, model, device))
    obj_embs = torch.stack(obj_embs)
    
    attr_embs = []
    for attr in candidate_attributes:
        attr_embs.append(get_sentence_embedding(attr, tokenizer, model, device))
    attr_embs = torch.stack(attr_embs)
    
    return obj_embs, attr_embs


def predict_obj(candidate_objects: list[str], candidate_attributes: list[str], template: str, tokenizer, model, device):
    """
    Predict masked object given attribute using a template.
    """
    assert candidate_objects is not None
    assert candidate_attributes is not None
    assert len(template) > 0
    print(f"\nTemplate: {template}")
    candidate_ids = []
    for token in candidate_objects:
        candidate_ids.append(tokenizer.convert_tokens_to_ids(token.lower()))
    
    probs = []
    for attr in candidate_attributes:
        sentence = template.format(MASK=tokenizer.mask_token, ATTR=attr)
        print(f"Sentence: {sentence}")
        token_ids = tokenizer.encode(sentence, return_tensors='pt')
        masked_index = token_ids[0].tolist().index(tokenizer.mask_token_id)
        with torch.no_grad():
            outputs = model(token_ids.to(device))
            prob = outputs.logits[0, masked_index][candidate_ids].sigmoid()
        probs.append(prob)
    print("\n")
    return torch.stack(probs)


def minmax_normalize(probs_pooled, candidate_objects, candidate_attributes, pooling_strategy: str):
    outmap_minpool = torch.log(probs_pooled).min(dim=0,keepdim=True)[0]
    outmap_maxpool = torch.log(probs_pooled).max(dim=0,keepdim=True)[0]
    logprob_normalized = (torch.log(probs_pooled) - outmap_minpool) / (outmap_maxpool - outmap_minpool)
    print(f"Logprob normalized:\n{logprob_normalized}\n{logprob_normalized.shape}\n")
    fc_pred_df_pooled_normalized = np.array(logprob_normalized.t().cpu())
    fc_pred_df_pooled_normalized = pd.DataFrame(fc_pred_df_pooled_normalized,columns=candidate_attributes, index=pd.Index(candidate_objects))
    fc_pred_df_pooled_normalized_rounded = fc_pred_df_pooled_normalized.round(0).astype(float)
    if pooling_strategy == 'max':
        print(f"fc_pred_df_maxpool_normalized_rounded:\n{fc_pred_df_pooled_normalized_rounded}\n")
    elif pooling_strategy == 'avg':
        print(f"fc_pred_df_avgpool_normalized_rounded:\n{fc_pred_df_pooled_normalized_rounded}\n")
    return logprob_normalized, fc_pred_df_pooled_normalized, fc_pred_df_pooled_normalized_rounded


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
        fc_true_df = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)

    candidate_objects = fc_true_df.index.tolist()
    candidate_attributes = fc_true_df.columns.tolist()
    print(f"{len(candidate_objects)} candidate objects:\n{candidate_objects}\n")
    print(f"{len(candidate_attributes)} candidate attributes:\n{candidate_attributes}\n")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)
    # tokenizer = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)

    obj_embs, attr_embs = encode_candidate_objs_attrs(candidate_objects, candidate_attributes, tokenizer, model, device)
    
    neg_dist = -torch.cdist(obj_embs, attr_embs, p=2.0)
    df_neg = np.array(neg_dist.t().cpu())
    df_neg = pd.DataFrame(df_neg, columns=candidate_objects, index=pd.Index(candidate_attributes))
    df_neg.to_csv(f'{cache_dir}/fc_animal_pred_neg.csv')
    

    # Load templates.
    with open(f'{input_dir}/templates.yaml', 'r', encoding='iso-8859-1') as f:
        templates_yaml = yaml.safe_load(f)
    templates = templates_yaml['templates']
    
    stacked_tensors = []
    for template in templates:
        stacked_tensors.append(predict_obj(candidate_objects, candidate_attributes, template, tokenizer, model_mlm, device))
    stacked_tensors = torch.stack(stacked_tensors, dim=0)
    print(f"\nStacked tensors:\n{stacked_tensors}\n{stacked_tensors.shape}\n")

    # Get the 2-dimensional projection of the smoothed formal context (conceptual embedding) to approximate the probabilistic incidence matrix.
    # The approximation can be achieved by aggregating over multiple patterns, either through max pooling or average pooling.
    # Here we use max pooling.
    probs_maxpool = torch.max(stacked_tensors, dim=0)[0]
    print(f"\nProbs maxpool:\n{probs_maxpool}\n{probs_maxpool.shape}\n")

    # Binarization using min-max normalization.
    logprob_maxpool_normalized, fc_pred_df_maxpool_normalized, fc_pred_df_maxpool_normalized_rounded = minmax_normalize(
        probs_maxpool, 
        candidate_objects, 
        candidate_attributes, 
        'max'
    )
    np.save(f'{cache_dir}/logprob_maxpool_normalized.npy', logprob_maxpool_normalized.cpu())
    np.savetxt(f'{cache_dir}/logprob_maxpool_normalized.csv', logprob_maxpool_normalized.cpu(), delimiter=',', fmt='%f')
    fc_pred_df_maxpool_normalized.to_csv(f'{cache_dir}/fc_animal_pred_maxpool_normalized.csv')
    fc_pred_df_maxpool_normalized_rounded.to_csv(f'{cache_dir}/fc_animal_pred_maxpool_normalized_rounded.csv')


    # Get the 2-dimensional projection of the smoothed formal context (conceptual embedding) to approximate the probabilistic incidence matrix.
    # The approximation can be achieved by aggregating over multiple patterns, either through max pooling or average pooling.
    # Here we use average pooling.
    probs_avgpool = torch.mean(stacked_tensors, dim=0)
    print(f"Probs avgpool:\n{probs_avgpool}\n{probs_avgpool.shape}\n")

    # Binarization using min-max normalization.
    logprob_avgpool_normalized, fc_pred_df_avgpool_normalized, fc_pred_df_avgpool_normalized_rounded = minmax_normalize(
        probs_avgpool, 
        candidate_objects, 
        candidate_attributes, 
        'avg'
    )
    np.save(f'{cache_dir}/logprob_avgpool_normalized.npy', logprob_avgpool_normalized.cpu())
    np.savetxt(f'{cache_dir}/logprob_avgpool_normalized.csv', logprob_avgpool_normalized.cpu(), delimiter=',', fmt='%f')
    fc_pred_df_avgpool_normalized.to_csv(f'{cache_dir}/fc_animal_pred_avgpool_normalized.csv')
    fc_pred_df_avgpool_normalized_rounded.to_csv(f'{cache_dir}/fc_animal_pred_avgpool_normalized_rounded.csv')


if __name__ == "__main__":
    main()