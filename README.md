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
