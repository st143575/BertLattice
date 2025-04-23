# Datasets

 We construct three new datasets of formal contexts in different domains, serving as the gold standards for evaluation. 
Two of them are derived from commonsense knowledge, and the third one is from the biomedical domain. 

1) Region-language details the official languages used in different administrative regions around the world. 

2) Animal-behavior captures the behaviors (e.g., live on land) of animals (e.g., tiger). This dataset is constructed through human curation. We compiled a set of the most popular animal names in English, considering only those with a single token. We identified $25$ behaviors based on animal attributes that aid in distinguishing them, such as habitat preferences, dietary habits, and methods of locomotion; 

3) Disease-symptom describes the symptoms associated with various diseases. We extracted diseases represented by a single token and their symptoms from a dataset available on Kaggle (https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).

## Formal context extraction

We consider three variants of BERT models: BERT-distill, BERT-base, and BERT-large. For each model, we use the uncased versions and compare the Average pooling and Max pooling variants, denoted as BertLattice (avg.) and BertLattice (max.), respectively.

Code can be found in the /Code/ directory. For example, to run the experiments on Animal-behavior, run BertConditionalProbability-Animal.ipynb.

## Formal context construction

Run FCA.ipynb in /FCA/

