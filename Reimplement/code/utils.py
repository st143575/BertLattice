def load_data(input_dir, fn):
    with open(f'{input_dir}/{fn}', 'r', encoding='iso-8859-1') as l:
        animal_behavior_file = l.readlines()

    animals = set()
    behaviors = set()  # habits
    popularity = {}
    animal_behavior = {}  # animal_habit
    behavior_animal = {}  # habit_animal
    for line in animal_behavior_file:
        animal, behavior, t = line.split(',')[0].strip(), line.split(',')[1].strip(), int(line.split(',')[2].split('\n')[0].strip())
        animals.add(animal)
        behaviors.add(behavior)
        if t == 1:
            if behavior not in popularity:
                popularity[behavior] = 1
            else:
                popularity[behavior] += 1

            if animal not in animal_behavior:
                animal_behavior[animal] = set()
                animal_behavior[animal].add(behavior)

            if behavior not in behavior_animal.keys():
                behavior_animal[behavior] = set()
                behavior_animal[behavior].add(animal)
            else:
                behavior_animal[behavior].add(animal)
    return animals, behaviors, popularity, animal_behavior, behavior_animal


def count_data(animals, behaviors, popularity, animal_behavior, behavior_animal):
    popularity_behaviors = dict(sorted(popularity.items(), key=lambda item: item[1],reverse = True))  # popularity_habits
    print("Data statistics:")
    print(
        f"Number of animals: {len(animals)},\n"
        f"Number of behaviors: {len(behaviors)},\n"
        f"Length of popularity: {len(popularity)},\n"
        f"Length of animal_behavior: {len(animal_behavior)},\n"
        f"Length of behavior_animal: {len(behavior_animal)},\n"
        f"Length of popularity_behaviors: {len(popularity_behaviors)}\n"
    )


def hit_k(df, truth, k):
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


def mean_rank(df, truth):
    reciprocal_rank = 0
    count = 0
    for ind in df.index:
        keys = df.loc[ind].keys()
        values = df.loc[ind].values
        rank = values.argsort()[::-1]
        for attr in truth[ind]:
            if attr in keys.tolist():
                reciprocal_rank = reciprocal_rank + (1 / (rank.tolist().index(keys.tolist().index(attr)) + 1))
                count += 1
    return reciprocal_rank / count