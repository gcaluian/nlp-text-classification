# Natural Language Processing: Text Analysis and Classification

This project explores natural language processing techniques across three tasks: stylistic analysis of novels, supervised classification of UK parliamentary speeches using TF-IDF with SVM and Random Forest models, and zero-shot/few-shot classification using Phi-4-mini-instruct.

The work was completed as part of the Natural Language Processing module in my MSc Data Science programme


## Project Components

### Part One — Syntax and Style Analysis

Analysis of a collection of novels using NLTK and spaCy.

The script:
- reads and organises the novel texts into a pandas DataFrame
- calculates type-token ratio to examine lexical diversity
- calculates Flesch Reading Ease scores
- parses the texts using spaCy
- identifies the ten most common syntactic subjects
- calculates Pointwise Mutual Information (PMI) to identify verbs associated with the subjects `he` and `she`


### Part Two — Machine Learning Text Classification

Classification of UK parliamentary speeches by political party using traditional machine learning methods.

The script:
- prepares and filters the parliamentary speech dataset
- converts the text into TF-IDF features
- trains and evaluates Random Forest and linear SVM classifiers
- compares unigram features with unigram, bigram and trigram features
- implements a custom tokenizer that removes punctuation, numbers, short words and English stopwords
- evaluates the models using macro F1 score and classification reports



### Part Three — Zero-Shot and Few-Shot LLM Classification

Classification of UK parliamentary speeches using prompting with the `microsoft/Phi-4-mini-instruct` language model.

The script:
- prepares a smaller parliamentary speech dataset using the same four party labels
- uses Hugging Face Transformers to run Phi-4-mini-instruct
- performs zero-shot classification using a prompt that asks the model to return one party label
- performs few-shot classification using one labelled training example from each party
- cleans generated outputs so they can be compared with the true labels
- evaluates zero-shot and few-shot predictions using macro F1 score and classification reports.
