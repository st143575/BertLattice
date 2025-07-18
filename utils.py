import torch
import pandas as pd


def load_data(input_dir, fn):
    with open(f'{input_dir}/{fn}', 'r', encoding='iso-8859-1') as l:
        animal_behavior_file = l.readlines()

    animals = set()
    behaviors = set()  # habits
    popularity = {}
    animal2behaviors = {}  # animal_habit
    behavior2animals = {}  # habit_animal
    for line in animal_behavior_file:
        animal, behavior, t = line.split(',')[0].strip(), line.split(',')[1].strip(), int(line.split(',')[2].split('\n')[0].strip())
        animals.add(animal)
        behaviors.add(behavior)
        if t == 1:
            if behavior not in popularity:
                popularity[behavior] = 1
            else:
                popularity[behavior] += 1

            if animal not in animal2behaviors:
                animal2behaviors[animal] = set()
                animal2behaviors[animal].add(behavior)
            else:
                animal2behaviors[animal].add(behavior)

            if behavior not in behavior2animals:
                behavior2animals[behavior] = set()
                behavior2animals[behavior].add(animal)
            else:
                behavior2animals[behavior].add(animal)
    return animals, behaviors, popularity, animal2behaviors, behavior2animals, animal_behavior_file


def count_data(animals, behaviors, popularity, animal2behaviors, behavior2animals):
    popularity_behaviors = dict(sorted(popularity.items(), key=lambda item: item[1],reverse = True))  # popularity_habits
    print("Data statistics:")
    print(
        f"Number of animals: {len(animals)},\n"
        f"Number of behaviors: {len(behaviors)},\n"
        f"Length of popularity: {len(popularity)},\n"
        f"Length of animal2behaviors: {len(animal2behaviors)},\n"
        f"Length of behavior2animals: {len(behavior2animals)},\n"
        f"Length of popularity_behaviors: {len(popularity_behaviors)}\n"
    )


def get_sentence_embedding(sentence: str, tokenizer, model, device): 
    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # The hidden states are in the second item in the outputs tuple
    last_hidden_state = outputs.last_hidden_state
    # Get the mean of the token embeddings to represent the sentence embedding
    sentence_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    return sentence_embedding


def hit_k(df: pd.DataFrame, truth: pd.DataFrame, k: int) -> float:
    total_pair = 0
    hits = 0
    for ind in df.index:
        keys = df.loc[ind].keys()
        values = df.loc[ind].values
        topk = values.argsort()[-k:][::-1]
        for attr in truth[ind]:
            if attr in set(keys[topk].tolist()):
                hits += 1
            total_pair += 1
    return hits / total_pair


def mrr(df: pd.DataFrame, truth: dict[str, set]) -> float:
    # print(f"df: {df}\n")
    # print(f"truth: {truth}\n")
    reciprocal_rank = 0
    count = 0
    for ind in df.index:
        # print(f"ind: {ind}")
        keys = df.loc[ind].keys()
        # print(f"keys: {keys}")
        values = df.loc[ind].values
        # print(f"values: {values}")
        rank = values.argsort()[::-1]
        # print(f"rank: {rank}")
        # print(f"truth: {truth[ind]}")
        for attr in truth[ind]:
            # print(f"attr: {attr}")
            if attr in keys.tolist():
                reciprocal_rank = reciprocal_rank + (1 / (rank.tolist().index(keys.tolist().index(attr)) + 1))
                count += 1
    return reciprocal_rank / count