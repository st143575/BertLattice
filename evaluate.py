import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score
from sklearn.manifold import TSNE


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

def calculate_metrics(
    fc_pred_df_pooled_normalized_rounded, 
    animal2behaviors, 
    y_true, 
    y_pred_pooled, 
    y_pred_scores_pooled
):
    mrr_pooled = mrr(fc_pred_df_pooled_normalized_rounded, animal2behaviors)
    h1_pooled = hit_k(fc_pred_df_pooled_normalized_rounded, animal2behaviors, 1)
    h5_pooled = hit_k(fc_pred_df_pooled_normalized_rounded, animal2behaviors, 5)
    h10_pooled = hit_k(fc_pred_df_pooled_normalized_rounded, animal2behaviors, 10)
    f1_pooled = f1_score(
        y_true=y_true, 
        y_pred=y_pred_pooled, 
        average="weighted"
    )
    mAP_pooled = average_precision_score(y_true, y_pred_scores_pooled)
    return mrr_pooled, h1_pooled, h5_pooled, h10_pooled, f1_pooled, mAP_pooled


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

    fc_animal_true = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)
    print(f"fc_animal_true:\n{fc_animal_true}\n")

    y_true = fc_animal_true.values.astype(int)
    print(f"y_true:\n{y_true}\nshape: {y_true.shape}\n")
    np.save(f'{cache_dir}/y_true.npy', y_true)
    np.savetxt(f'{cache_dir}/y_true.csv', y_true, delimiter=',', fmt='%d')

    # Data for MAXPOOL.
    fc_pred_df_maxpool_normalized = pd.read_csv(f'{cache_dir}/fc_animal_pred_maxpool_normalized.csv', index_col=0)
    print(f"fc_pred_df_maxpool_normalized:\n{fc_pred_df_maxpool_normalized}\n")

    fc_pred_df_maxpool_normalized_rounded = pd.read_csv(f'{cache_dir}/fc_animal_pred_maxpool_normalized_rounded.csv', index_col=0)
    print(f"fc_pred_df_maxpool_normalized_rounded:\n{fc_pred_df_maxpool_normalized_rounded}\n")

    logprob_maxpool_normalized = np.load(f'{cache_dir}/logprob_maxpool_normalized.npy')
    print(f"logprob_maxpool_normalized:\n{logprob_maxpool_normalized}\nshape: {logprob_maxpool_normalized.shape}\n")

    # Data for AVGPOOL.
    fc_pred_df_avgpool_normalized = pd.read_csv(f'{cache_dir}/fc_animal_pred_avgpool_normalized.csv', index_col=0)
    print(f"fc_pred_df_avgpool_normalized:\n{fc_pred_df_avgpool_normalized}\n")

    fc_pred_df_avgpool_normalized_rounded = pd.read_csv(f'{cache_dir}/fc_animal_pred_avgpool_normalized_rounded.csv', index_col=0)
    print(f"fc_pred_df_avgpool_normalized_rounded:\n{fc_pred_df_avgpool_normalized_rounded}\n")

    logprob_avgpool_normalized = np.load(f'{cache_dir}/logprob_avgpool_normalized.npy')
    print(f"logprob_avgpool_normalized:\n{logprob_avgpool_normalized}\nshape: {logprob_avgpool_normalized.shape}\n")


    # Calculate metrics for MAXPOOL.
    y_pred_maxpool = fc_pred_df_maxpool_normalized_rounded.values.astype(int)
    print(f"y_pred_maxpool:\n{y_pred_maxpool}\nshape: {y_pred_maxpool.shape}\n")
    np.save(f'{cache_dir}/y_pred_maxpool.npy', y_pred_maxpool)
    np.savetxt(f'{cache_dir}/y_pred_maxpool.csv', y_pred_maxpool, delimiter=',', fmt='%d')
    
    y_pred_scores_maxpool = fc_pred_df_maxpool_normalized.values
    print(f"y_pred_scores_maxpool:\n{y_pred_scores_maxpool}\nshape: {y_pred_scores_maxpool.shape}\n")
    np.save(f'{cache_dir}/y_pred_scores_maxpool.npy', y_pred_scores_maxpool)
    np.savetxt(f'{cache_dir}/y_pred_scores_maxpool.csv', y_pred_scores_maxpool, delimiter=',', fmt='%f')

    mrr_maxpool, h1_maxpool, h5_maxpool, h10_maxpool, f1_maxpool, mAP_maxpool = calculate_metrics(
        fc_pred_df_maxpool_normalized_rounded,
        animal2behaviors,
        y_true,
        y_pred_maxpool,
        y_pred_scores_maxpool,
    )
    print(f"MRR_maxpool: {mrr_maxpool}\nHit@1_maxpool: {h1_maxpool}\nHit@5_maxpool: {h5_maxpool}\nHit@10_maxpool: {h10_maxpool}\nf1_maxpool: {f1_maxpool}\nmAP_maxpool: {mAP_maxpool}\n")

    with open(f'{cache_dir}/metrics_animal_maxpool.txt', 'w') as f:
        f.write(f"mrr: {mrr_maxpool}\n")
        f.write(f"h1: {h1_maxpool}\n")
        f.write(f"h5: {h5_maxpool}\n")
        f.write(f"h10: {h10_maxpool}\n")
        f.write(f"f1: {f1_maxpool}\n")
        f.write(f"mAP: {mAP_maxpool}\n")

    # Calculate metrics for AVGPOOL.
    y_pred_avgpool = fc_pred_df_avgpool_normalized_rounded.values.astype(int)
    print(f"y_pred_avgpool:\n{y_pred_avgpool}\nshape: {y_pred_avgpool.shape}\n")
    np.save(f'{cache_dir}/y_pred_avgpool.npy', y_pred_avgpool)
    np.savetxt(f'{cache_dir}/y_pred_avgpool.csv', y_pred_avgpool, delimiter=',', fmt='%d')
    
    y_pred_scores_avgpool = fc_pred_df_avgpool_normalized.values
    print(f"y_pred_scores_avgpool:\n{y_pred_scores_avgpool}\nshape: {y_pred_scores_avgpool.shape}\n")
    np.save(f'{cache_dir}/y_pred_scores_avgpool.npy', y_pred_scores_avgpool)
    np.savetxt(f'{cache_dir}/y_pred_scores_avgpool.csv', y_pred_scores_avgpool, delimiter=',', fmt='%f')

    mrr_avgpool, h1_avgpool, h5_avgpool, h10_avgpool, f1_avgpool, mAP_avgpool = calculate_metrics(
        fc_pred_df_avgpool_normalized_rounded,
        animal2behaviors,
        y_true,
        y_pred_avgpool,
        y_pred_scores_avgpool,
    )
    print(f"MRR_avgpool: {mrr_avgpool}\nHit@1_avgpool: {h1_avgpool}\nHit@5_avgpool: {h5_avgpool}\nHit@10_avgpool: {h10_avgpool}\nf1_avgpool: {f1_avgpool}\nmAP_avgpool: {mAP_avgpool}\n")

    with open(f'{cache_dir}/metrics_animal_avgpool.txt', 'w') as f:
        f.write(f"mrr: {mrr_avgpool}\n")
        f.write(f"h1: {h1_avgpool}\n")
        f.write(f"h5: {h5_avgpool}\n")
        f.write(f"h10: {h10_avgpool}\n")
        f.write(f"f1: {f1_avgpool}\n")
        f.write(f"mAP: {mAP_avgpool}\n")

    # Visualize the formal context.
    plt.figure(figsize=(25, 35), dpi=200)
    sns.set_theme(style="white", font_scale=1.)
    x_axis_labels = list(behaviors)
    y_axis_labels = list(animals)
    fc_animal_behavior_maxpool = pd.DataFrame(logprob_maxpool_normalized.T, columns=pd.Index(x_axis_labels), index=pd.Index(y_axis_labels))
    print(f"fc_animal_behavior_maxpool:\n{fc_animal_behavior_maxpool}\n")
    ax = sns.heatmap(
        fc_animal_behavior_maxpool,
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
        cbar=False,
        linewidth=0.2,
        cmap='GnBu',
        square=False,
        annot=False,
    )
    # ax.set_title('"Animal-Behavior" concept lattice', fontsize=18)
    plt.tight_layout()

    plt.tick_params(axis='y', which='major', colors='black', labelsize=24, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.tick_params(axis='x', which='major', colors='black', rotation=90, labelsize=24, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.savefig(f'{cache_dir}/Full-softmax-97-25-animal.pdf', bbox_inches='tight', pad_inches=0, dpi=1200)
    # plt.show()
    fc_animal_behavior_maxpool.to_csv(f'{cache_dir}/fc_animal_behavior_maxpool.csv', index=True)

    # T-SNE visualization.
    fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=200)
    X_embedded = TSNE(
        n_components=2, 
        learning_rate='auto', 
        init='random', 
        perplexity=3,
        random_state=42,
    ).fit_transform(fc_pred_df_maxpool_normalized_rounded.iloc[:, 1:].values)
    print(f"X_embedded shape: {X_embedded.shape}")

    labels = fc_pred_df_maxpool_normalized_rounded['fly in the sky'].astype(int).values
    plt.scatter(X_embedded[labels==0, 0], X_embedded[labels==0, 1], color='tab:blue', label='Animals that do not fly')
    plt.scatter(X_embedded[labels==1, 0], X_embedded[labels==1, 1], color='tab:orange', label='Animals that fly')
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig(f'{cache_dir}/animals_tsne_emb.pdf', bbox_inches='tight', pad_inches=0, dpi=800)
    print(fc_pred_df_maxpool_normalized_rounded.shape)


if __name__ == '__main__':
    main()