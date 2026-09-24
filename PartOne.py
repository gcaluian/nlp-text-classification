#NLP assessment template 2026

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import spacy
import nltk
from pathlib import Path
import pandas as pd
nltk.download("cmudict")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000
from collections import Counter
import math
import re
import string 


def fk_level(text, d):
    """Returns the Flesch-Kincaid Reading Ease Level of a text (higher grade is easier).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    """
    # Sentence tokenisation
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Word tokenisation
    tokens = nltk.word_tokenize(text)
    
    # Making the tokens lower case
    tokens = [t.lower() for t in tokens]
    
    # Removing punctuation
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', token) for token in tokens]
    
    # removing numbers and empty spaces
    tokens = [token for token in tokens if token.isalpha()]
    
    # Total words
    total_words = len(tokens)
    
    # Counting total syllables using the count_syl function which finds the syllable of a word
    total_syllables = 0
    for token in tokens:
        total_syllables += count_syl(token, d)
        
    # Calcluate Flesh-Kincaid reading ease score
    score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * ( total_syllables / total_words)
    
    return score
    
    


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower() # making the word lower case and matching it with the word from the dictionary
    phones_list = d.get(word)
    
    if phones_list: # checks whether the list is empty or not
        phones = phones_list[0] # choosing the first pronunciation
        
        total_syllables = 0
        
        # going through every sound that the dictionary returned for the word
        for phone in phones:
            if phone[-1].isdigit():  # checking if the sounds ends in a number
                total_syllables += 1 # and if yes we are adding it to the syllables 
            
        return total_syllables
    
    vowels = "aeiouy" # the letters that we consider vowels
    syllable_count = 0
    
    previous_char_was_vowel = False # we will use this to remember if the previous character was a vowel
    
    for char in word: # going through every character/letter of the word 
        if char in vowels: #checks if the character is a vowel
            if not previous_char_was_vowel: # here we check if the previous char is a vowel
                syllable_count +=1
            previous_char_was_vowel = True
        else:
            previous_char_was_vowel = False
                
    return syllable_count         
            
        
        
    


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    
    # creating an empty list which will later become the dataframe
    rows = []
    
    # creating a for loop that goes over each name of the file and is extracting title, author, year and text
    for file in path.glob("*.txt"):
        title, author, year = file.stem.split("-")
        title = title.replace("_", " ")
        text = file.read_text(encoding="utf-8")
        
        rows.append({
            "text": text,
            "title": title,
            "author": author,
            "year": int(year)
        })
        
    # returning the dataframe sorted by year and with the resetting the index
    return pd.DataFrame(rows).sort_values("year").reset_index(drop = True)
        

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    docs = []
    
    for i, row in df.iterrows(): # going over each text in the dataframe
        text = row["text"]
        doc = nlp(text)     # analysing each text with spaCy
        docs.append(doc)    # adding the spaCy doc to docs
        
    df["parsed"] = docs     # updating the dataframe with another column and the spaCy analysis of each text
    
    # creating the folder
    store_path.mkdir(exist_ok = True)
    
    # storing the pickle file inside the folder
    df.to_pickle(store_path / out_name)
    
    return df


def text_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    
    # Tokenising using  nltk
    tokens = nltk.word_tokenize(text)
    
    # making the tokens lower case
    tokens = [t.lower() for t in tokens]
    
    #removing punctuation
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', token) for token in tokens]
    
    #removing numbers and empty spaces
    tokens = [token for token in tokens if token.isalpha()]
    
    #calculate type-token ratio
    ttr = len(set(tokens)) / len(tokens)
    
    return ttr


def nltk_ttr(df):
    """helper function to add ttr to a dataframe""" "returns a dictionary mapping each novel title to its ttr"
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = text_ttr(row["text"])
    return results


def flesch_kincaid(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


#.. add functions for part (e) here

# function for the syntactic subjects
def common_subjects(doc):
    # returns the 10 most common syntactic subjects
    
    subjects = Counter()
    
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subjects[token.lemma_.lower()] += 1
            
    return subjects.most_common(10)


# function for the verbs
def subject_verb_pmi(doc, subject = "he"):
    # returns verbs most associated with a given subject ordered by PMI
    
    subject_verb_counts = Counter()     # stores counts for each subject-verb pair
    subject_counts = Counter()          # will store how often each subject will appear
    verb_counts = Counter()             # will store how often each verb will appear with any subject
    
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):    # checks if the token is a syntactic subject
            subj = token.lemma_.lower()             #stores the subject as a lemma
            verb = token.head.lemma_.lower()        # stores the verb as a lemma
            
            if token.head.pos_ in ("VERB", "AUX"):      # checks if the subjects head is verb or aux
                subject_verb_counts[(subj, verb)] +=1
                subject_counts[subj] +=1
                verb_counts[verb] +=1 
                 
    total_pairs = sum(subject_verb_counts.values())    # total pairs of subject - verbs
    
    pmi_scores = []
    
    for (subj, verb), count in subject_verb_counts.items(): 
        if subj == subject:
            p_subject_verb = count / total_pairs        # probability of the subject and verb together out of the total pairs
            p_subject = subject_counts[subj] / total_pairs      # probability of the subject 
            p_verb = verb_counts[verb] / total_pairs            # probability of the verb
            
            pmi = math.log2(p_subject_verb / (p_subject * p_verb)) 
            
            pmi_scores.append((verb, pmi, count))
            
    pmi_scores = sorted(pmi_scores, key=lambda x: x[1], reverse = True)  #ordering the pmi_scores 
        
    return pmi_scores[:10]    
        
if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    
    parse(df)
    
    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    
    print(nltk_ttr(df))
    print(flesch_kincaid(df))
    
    # call functions for part (e) here.
    
    # print the title and 10 most common syntactic subjects
    print(" 10 most common syntactic subjects:")
    for i, row in df.iterrows():
        print(row["title"])
        print(common_subjects(row["parsed"]))
        print()
        
    # print title and verbs most associated with "he"  
    print("verbs associated with 'he' by PMI")  
    for i, row in df.iterrows():
        print(row["title"])
        print(subject_verb_pmi(row["parsed"], subject = "he"))
        print()


    # print title and verbs most associated with "she"
    print("verbs associated with 'she' by PMI")
    for i, row in df.iterrows():
        print(row["title"])
        print(subject_verb_pmi(row["parsed"], subject = "she"))
        print()
    
