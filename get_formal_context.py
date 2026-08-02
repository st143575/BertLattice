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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

OBJECT_SLOT = "{MASK}"
ATTRIBUTE_SLOT = "{ATTR}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run BERT conditional probability on Animal-behavior dataset.')
    parser.add_argument('-i', '--input_dir', type=str, default='./data', help='Path to input data directory')
    parser.add_argument('-fn', '--file_name', type=str, default='animal_behavior_file.txt', help='Input file name')
    parser.add_argument('-o', '--output_dir', type=str, default='./cache', help='Path to output data directory')
    parser.add_argument('-m', '--model_path', type=str, default='./cache/models--bert-base-uncased', help='Path to BERT model')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='GPU-ID')
    # Arguments for Definition 9: optional open-vocabulary Gibbs sampling.
    parser.add_argument(
        "--run_gibbs", 
        action="store_true", 
        help="Run Gibbs sampling in addition to the original closed-set reconstruction."
    )
    parser.add_argument(
        "--gibbs_only", 
        action="store_true", 
        help=(
            "Run only the open-vocabulary setting as described by Definition 9 in the paper. "
            "This skips loading the ground-truth candidate object/attribute sets used by the original closed-set evaluation."
        )
    )
    parser.add_argument(
        "--gibbs_steps", 
        type=int, 
        default=500, 
        help="Number of Gibbs transitions after the initial (g0, m0) state."
    )
    parser.add_argument(
        "--gibbs_burn_in", 
        type=int, 
        default=100, 
        help="Discard states whose step index is smaller than this value."
    )
    parser.add_argument(
        "--gibbs_thinning",
        type=int,
        default=5,
        help="Retain one state every N transitions after burn-in."
    )
    parser.add_argument(
        "--gibbs_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Values below 1 make sampling more concentrated."
    )
    parser.add_argument(
        "--gibbs_top_k",
        type=int,
        default=100,
        help="Sample from the top-k valid vocabulary tokens; use 0 for all valid tokens."
    )
    parser.add_argument(
        "--gibbs_max_objects",
        type=int,
        default=100,
        help="Maximum number of unique sampled object tokens retained in the reconstructed matrix."
    )
    parser.add_argument(
        "--gibbs_max_attributes",
        type=int,
        default=100,
        help="Maximum number of unique sampled attribute tokens retained in the reconstructed matrix."
    )
    parser.add_argument(
        "--gibbs_threshold",
        type=float,
        default=0.5,
        help="Threshold applied after global log-probability min-max normalization."
    )
    parser.add_argument(
        "--gibbs_sequential_update",
        action="store_true",
        help=(
            "Use standard sequential Gibbs updates m_t ~ p(m | g_t). "
            "By default, the code follows Definition 9 literally: m_t ~ p(m | g_{t-1})."
        )
    )
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
    Predict masked object given each pre-defined attribute using a template.
    This is the closed-set setting, where the sets of candidate objects & attributes are fixed.
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
    """
    Binarization using minmax-normalization for lattice construction (cf. Equation 7 in the paper).
    Motivation: MLM outputs are softmax probabilities, but FCA requires a binary input.
    """
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


# ---------------------------------------------------------------------------
# Definition 9: formal-context generation via Gibbs sampling
# ---------------------------------------------------------------------------

def _validate_gibbs_template(template: str) -> None:
    """Require the supplied reproduction's two-slot template convention."""
    if template.count(OBJECT_SLOT) != 1 or template.count(ATTRIBUTE_SLOT) != 1:
        raise ValueError(
            "A Gibbs template must contain exactly one {MASK} object slot and one {ATTR} attribute slot. "
            f"Received: {template!r}"
        )


def _render_pattern(template: str, tokenizer, object_value: Optional[str], attribute_value: Optional[str]) -> str:
    """Fill either slot with a token or leave it masked."""
    _validate_gibbs_template(template)
    return template.format(
        MASK=tokenizer.mask_token if object_value is None else object_value,
        ATTR=tokenizer.mask_token if attribute_value is None else attribute_value,
    )


def _slot_mask_index(template: str, input_ids: torch.Tensor, tokenizer, slot: str) -> int:
    """
    Locate the object or attribute mask.

    When both slots are masked, their order in the tokenized input is inferred from
    the order of {MASK} and {ATTR} in the original template.
    """
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).flatten().tolist()
    if not mask_positions:
        raise ValueError("The rendered Gibbs pattern contains no mask token.")
    if len(mask_positions) == 1:
        return int(mask_positions[0])
    if len(mask_positions) != 2:
        raise ValueError(f"Expected one or two mask tokens, found {len(mask_positions)}.")

    slot_order = [
        name
        for _, name in sorted(
            [(template.index(OBJECT_SLOT), "object"), (template.index(ATTRIBUTE_SLOT), "attribute")]
        )
    ]
    if slot not in slot_order:
        raise ValueError(f"Unknown slot: {slot!r}")
    return int(mask_positions[slot_order.index(slot)])


def _masked_slot_logits(
    template: str, 
    tokenizer, 
    model, 
    device, 
    slot: str, 
    object_value: Optional[str], 
    attribute_value: Optional[str]
) -> torch.Tensor:
    """Return full-vocabulary logits at one masked slot."""
    sentence = _render_pattern(template, tokenizer, object_value, attribute_value)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    mask_index = _slot_mask_index(template, inputs["input_ids"], tokenizer, slot)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0, mask_index]


def _valid_open_vocabulary_ids(tokenizer, device) -> torch.Tensor:
    """
    Build a conservative vocabulary for open-vocabulary sampling.

    Definition 9 samples tokens from the MLM vocabulary. For a usable chain, this
    implementation excludes special tokens, WordPiece continuations (``##...``),
    punctuation, digits, and other non-alphabetic entries. Consequently each Gibbs
    state is a single lexical token; multi-token phrases require a blocked/iterative
    sampler and are deliberately not pretended to be supported here.
    """
    special_ids = set(tokenizer.all_special_ids)
    ids = []
    for token, token_id in tokenizer.get_vocab().items():
        if token_id in special_ids:
            continue
        if token.startswith("##"):
            continue
        if not token.isalpha():
            continue
        ids.append(token_id)
    if not ids:
        raise ValueError("No valid lexical tokens remain after vocabulary filtering.")
    return torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)


def _sample_token(
    logits: torch.Tensor,
    allowed_ids: torch.Tensor,
    tokenizer,
    temperature: float,
    top_k: int,
) -> tuple[int, str, float]:
    """Sample one valid vocabulary token and return id, text, and log probability."""
    if temperature <= 0:
        raise ValueError("gibbs_temperature must be greater than zero.")
    if top_k < 0:
        raise ValueError("gibbs_top_k must be non-negative.")

    allowed_logits = logits.index_select(0, allowed_ids) / temperature
    if top_k > 0 and top_k < allowed_logits.numel():
        selected_logits, selected_positions = torch.topk(allowed_logits, k=top_k)
        selected_ids = allowed_ids.index_select(0, selected_positions)
    else:
        selected_logits = allowed_logits
        selected_ids = allowed_ids

    probabilities = torch.softmax(selected_logits, dim=0)
    sampled_position = torch.multinomial(probabilities, num_samples=1).item()
    token_id = int(selected_ids[sampled_position].item())
    token = tokenizer.convert_ids_to_tokens(token_id)
    token_text = tokenizer.convert_tokens_to_string([token]).strip()
    log_probability = float(torch.log(probabilities[sampled_position].clamp_min(1e-30)).item())
    return token_id, token_text, log_probability


def gibbs_sample_object_attribute_pairs(
    template: str,
    tokenizer,
    model,
    device,
    num_steps: int,
    burn_in: int,
    thinning: int,
    temperature: float,
    top_k: int,
    sequential_update: bool = False,
) -> list[dict]:
    """
    Sample ``(g_t, m_t)`` states according to Definition 9.

    Initialization:
        g_0 ~ p_theta(. | b)
        m_0 ~ p_theta(. | b^{g_0,.})

    Literal Definition-9 transition (default):
        g_t ~ p_theta(. | b^{.,m_{t-1}})
        m_t ~ p_theta(. | b^{g_{t-1},.})

    Setting ``sequential_update=True`` instead uses the conventional sequential
    Gibbs update m_t ~ p_theta(. | b^{g_t,.}). The distinction is exposed because
    the paper's displayed Definition 9 conditions m_t on g_{t-1}.
    """
    _validate_gibbs_template(template)
    if num_steps < 0:
        raise ValueError("gibbs_steps must be non-negative.")
    if burn_in < 0:
        raise ValueError("gibbs_burn_in must be non-negative.")
    if thinning <= 0:
        raise ValueError("gibbs_thinning must be greater than zero.")

    allowed_ids = _valid_open_vocabulary_ids(tokenizer, device)

    # g_0 ~ p(. | b), where both slots are masked and the object mask is sampled.
    object_logits = _masked_slot_logits(
        template,
        tokenizer,
        model,
        device,
        slot="object",
        object_value=None,
        attribute_value=None,
    )
    object_id, object_text, object_logprob = _sample_token(
        object_logits, allowed_ids, tokenizer, temperature, top_k
    )

    # m_0 ~ p(. | b^{g_0,.}).
    attribute_logits = _masked_slot_logits(
        template,
        tokenizer,
        model,
        device,
        slot="attribute",
        object_value=object_text,
        attribute_value=None,
    )
    attribute_id, attribute_text, attribute_logprob = _sample_token(
        attribute_logits, allowed_ids, tokenizer, temperature, top_k
    )

    retained_states: list[dict] = []

    def retain(step: int, g_id: int, g: str, g_logp: float, m_id: int, m: str, m_logp: float) -> None:
        if step >= burn_in and (step - burn_in) % thinning == 0:
            retained_states.append(
                {
                    "step": step,
                    "object_id": g_id,
                    "object": g,
                    "object_sample_logprob": g_logp,
                    "attribute_id": m_id,
                    "attribute": m,
                    "attribute_sample_logprob": m_logp,
                }
            )

    retain(
        0,
        object_id,
        object_text,
        object_logprob,
        attribute_id,
        attribute_text,
        attribute_logprob,
    )

    for step in range(1, num_steps + 1):
        previous_object_text = object_text
        previous_attribute_text = attribute_text

        # g_t ~ p(. | b^{.,m_{t-1}})
        object_logits = _masked_slot_logits(
            template,
            tokenizer,
            model,
            device,
            slot="object",
            object_value=None,
            attribute_value=previous_attribute_text,
        )
        object_id, object_text, object_logprob = _sample_token(
            object_logits, allowed_ids, tokenizer, temperature, top_k
        )

        # Definition 9 literally uses g_{t-1}; standard sequential Gibbs uses g_t.
        conditioning_object = object_text if sequential_update else previous_object_text
        attribute_logits = _masked_slot_logits(
            template,
            tokenizer,
            model,
            device,
            slot="attribute",
            object_value=conditioning_object,
            attribute_value=None,
        )
        attribute_id, attribute_text, attribute_logprob = _sample_token(
            attribute_logits, allowed_ids, tokenizer, temperature, top_k
        )

        retain(
            step,
            object_id,
            object_text,
            object_logprob,
            attribute_id,
            attribute_text,
            attribute_logprob,
        )

    if not retained_states:
        raise ValueError(
            "No Gibbs states were retained. Reduce --gibbs_burn_in, increase --gibbs_steps, "
            "or reduce --gibbs_thinning."
        )
    return retained_states


def _unique_in_order(values: Sequence[str], limit: int) -> list[str]:
    """Deduplicate while preserving first appearance in the Markov chain."""
    unique = list(dict.fromkeys(values))
    if limit > 0:
        unique = unique[:limit]
    return unique


def reconstruct_conditional_context_from_samples(
    template: str,
    sampled_objects: Sequence[str],
    sampled_attributes: Sequence[str],
    tokenizer,
    model,
    device,
) -> torch.Tensor:
    """
    Compute p_theta(g | b^{.,m}) for every sampled object/attribute combination.

    Returns a tensor of shape ``[num_attributes, num_objects]``. Computing these
    indexed conditionals after sampling corresponds to the final sentence of
    Definition 9 and turns the sampled vocabulary into a probabilistic context.
    """
    object_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(token) for token in sampled_objects],
        dtype=torch.long,
        device=device,
    )
    rows = []
    for attribute in sampled_attributes:
        logits = _masked_slot_logits(
            template,
            tokenizer,
            model,
            device,
            slot="object",
            object_value=None,
            attribute_value=attribute,
        )
        probabilities = torch.softmax(logits, dim=-1)
        rows.append(probabilities.index_select(0, object_ids))
    return torch.stack(rows, dim=0)


def normalize_gibbs_context(
    probabilities: torch.Tensor,
    objects: Sequence[str],
    attributes: Sequence[str],
    threshold: float,
) -> tuple[torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Apply Equation (7)'s global log-probability min-max normalization."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("gibbs_threshold must lie in [0, 1].")
    log_probabilities = torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
    minimum = log_probabilities.min()
    maximum = log_probabilities.max()
    denominator = (maximum - minimum).clamp_min(torch.finfo(probabilities.dtype).eps)
    normalized = (log_probabilities - minimum) / denominator

    normalized_df = pd.DataFrame(
        normalized.t().detach().cpu().numpy(),
        index=pd.Index(objects, name="object"),
        columns=list(attributes),
    )
    binary_df = (normalized_df > threshold).astype(float)
    return normalized, normalized_df, binary_df


def run_gibbs_formal_context_generation(
    templates: Sequence[str],
    tokenizer,
    model,
    device,
    cache_dir: Path,
    args,
) -> None:
    """Run one Definition-9 chain per concept pattern and save pooled contexts."""
    print("\n=============DEFINITION 9: GIBBS SAMPLING==============")
    all_samples = []
    for template_index, template in enumerate(templates):
        print(f"Sampling template {template_index + 1}/{len(templates)}: {template}")
        samples = gibbs_sample_object_attribute_pairs(
            template=template,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_steps=args.gibbs_steps,
            burn_in=args.gibbs_burn_in,
            thinning=args.gibbs_thinning,
            temperature=args.gibbs_temperature,
            top_k=args.gibbs_top_k,
            sequential_update=args.gibbs_sequential_update,
        )
        for sample in samples:
            sample["template_index"] = template_index
            sample["template"] = template
        all_samples.extend(samples)

    samples_df = pd.DataFrame(all_samples)
    samples_df.to_csv(cache_dir / "gibbs_samples.csv", index=False)

    sampled_objects = _unique_in_order(samples_df["object"].tolist(), args.gibbs_max_objects)
    sampled_attributes = _unique_in_order(samples_df["attribute"].tolist(), args.gibbs_max_attributes)
    if not sampled_objects or not sampled_attributes:
        raise RuntimeError("Gibbs sampling did not produce a non-empty object and attribute vocabulary.")

    pd.DataFrame({"object": sampled_objects}).to_csv(cache_dir / "gibbs_objects.csv", index=False)
    pd.DataFrame({"attribute": sampled_attributes}).to_csv(cache_dir / "gibbs_attributes.csv", index=False)
    print(f"Retained {len(samples_df)} states, {len(sampled_objects)} unique objects, "
          f"and {len(sampled_attributes)} unique attributes.")

    # Y_hat has shape [patterns, attributes, objects].
    triadic_context = torch.stack(
        [
            reconstruct_conditional_context_from_samples(
                template,
                sampled_objects,
                sampled_attributes,
                tokenizer,
                model,
                device,
            )
            for template in templates
        ],
        dim=0,
    )
    np.save(cache_dir / "gibbs_triadic_context.npy", triadic_context.detach().cpu().numpy())

    for pooling_name, pooled in (
        ("maxpool", triadic_context.max(dim=0).values),
        ("avgpool", triadic_context.mean(dim=0)),
    ):
        normalized, normalized_df, binary_df = normalize_gibbs_context(
            pooled,
            sampled_objects,
            sampled_attributes,
            args.gibbs_threshold,
        )
        np.save(cache_dir / f"logprob_gibbs_{pooling_name}_normalized.npy", normalized.detach().cpu().numpy())
        normalized_df.to_csv(cache_dir / f"fc_gibbs_{pooling_name}_normalized.csv")
        binary_df.to_csv(cache_dir / f"fc_gibbs_{pooling_name}_normalized_rounded.csv")
        print(f"Saved Gibbs {pooling_name} context with shape {binary_df.shape}.")
    print("========================================================\n")



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

    # Load templates.
    with open(f'{input_dir}/templates.yaml', 'r', encoding='iso-8859-1') as f:
        templates_yaml = yaml.safe_load(f)
    templates = templates_yaml['templates']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)
    model_mlm.eval()

    # Run Gibbs sampling if specified.
    if args.run_gibbs or args.gibbs_only:
        run_gibbs_formal_context_generation(
            templates=templates,
            tokenizer=tokenizer,
            model=model_mlm,
            device=device,
            cache_dir=cache_dir,
            args=args,
        )
        if args.gibbs_only:
            return

    # Load the ground-truth formal context.
    with open(f'{cache_dir}/fc_animal_true.csv', 'r', encoding='iso-8859-1'):
        fc_true_df = pd.read_csv(f'{cache_dir}/fc_animal_true.csv', index_col=0)

    candidate_objects = fc_true_df.index.tolist()
    candidate_attributes = fc_true_df.columns.tolist()
    print(f"{len(candidate_objects)} candidate objects:\n{candidate_objects}\n")
    print(f"{len(candidate_attributes)} candidate attributes:\n{candidate_attributes}\n")

    model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir).to(device)
    model.eval()

    obj_embs, attr_embs = encode_candidate_objs_attrs(candidate_objects, candidate_attributes, tokenizer, model, device)
    
    neg_dist = -torch.cdist(obj_embs, attr_embs, p=2.0)
    df_neg = np.array(neg_dist.t().cpu())
    df_neg = pd.DataFrame(df_neg, columns=candidate_objects, index=pd.Index(candidate_attributes))
    df_neg.to_csv(f'{cache_dir}/fc_animal_pred_neg.csv')
    
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