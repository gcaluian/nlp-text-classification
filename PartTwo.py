

import string
import re 
from pathlib import Path
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

path = Path.cwd() / "texts" / "hansard10000.csv"
df = pd.read_csv(path)

#print(df.head())
#print(df.columns)


# rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
#print(df["party"].value_counts().head(10))

# finding the four most common party names
top_four_parties = df["party"].value_counts().head(4).index
#print(top_four_parties)


# remove any rows where the value of the ‘party’ column is not one of the four most common party names.
# the top four are Conservative, Labour, SNP and Liberal Democrat
df = df[df["party"].isin(top_four_parties)]

# remove the Speaker rows
df = df[df["party"] != "Speaker"]

# print(df["party"].head())

# Removing any rows where the value in the speech_class column is not "Speech"
df = df[df["speech_class"] == "Speech"]
# print(df["speech_class"].value_counts())

# Removing the rows where the text in the speech column is less than 1000 characters long
df = df[df["speech"].str.len() >= 1000]

# printing the dimensions of the dataframe
print(df.shape)

# Separating the training data and the labels
X = df["speech"]
y = df["party"]

# creating the vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)

# Creating the data for training with the random seed 26
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=26)

# Applying the vectorizer
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the Random Forest
random_forest = RandomForestClassifier(n_estimators=300, random_state=26)
random_forest.fit(X_train_tfidf, y_train)
random_forest_predictions = random_forest.predict(X_test_tfidf)

# Printing results for Random Forest
print("Basic TF-IDF Random Forest")
print(f1_score(y_test, random_forest_predictions, average="macro"))
print(classification_report(y_test, random_forest_predictions, zero_division=0))

# Training the SVM with linear kernel
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train_tfidf, y_train)
svm_classifier_predictions = svm_classifier.predict(X_test_tfidf)

# Printing results for SVM
print("Basic tf-idf Linear SVM")
print(f1_score(y_test, svm_classifier_predictions, average="macro"))
print(classification_report(y_test, svm_classifier_predictions, zero_division=0))



# Training a new vectorizer with unigrams, bi-grams and tri-grams
new_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,3))

X_train_ngram = new_vectorizer.fit_transform(X_train)
X_test_ngram = new_vectorizer.transform(X_test)

# Training the ngram Random Forest
ngram_random_forest = RandomForestClassifier(n_estimators=300, random_state=26)
ngram_random_forest.fit(X_train_ngram, y_train)
ngram_random_forest_predictions = ngram_random_forest.predict(X_test_ngram)

# Printing results for ngram Random Forest
print("N-gram TF-IDF Random Forest")
print(f1_score(y_test, ngram_random_forest_predictions, average="macro"))
print(classification_report(y_test, ngram_random_forest_predictions, zero_division=0))

# Training the ngram SVM with linear kernel
ngram_svm_classifier = SVC(kernel="linear")
ngram_svm_classifier.fit(X_train_ngram, y_train)
ngram_svm_classifier_predictions = ngram_svm_classifier.predict(X_test_ngram)

# Printing results for ngram SVM
print("N-gram TF-IDF Linear SVM")
print(f1_score(y_test, ngram_svm_classifier_predictions, average="macro"))
print(classification_report(y_test, ngram_svm_classifier_predictions, zero_division=0))



# Making a tokenizer
def custom_tokeniser(text):
    # split and lower case
    tokens = text.split()
    tokens = [t.lower() for t in tokens]

    # remove punctuation
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', token) for token in tokens]

    # remove numbers
    tokens = [token for token in tokens if token.isalpha()]
    
    # removing short words
    tokens = [token for token in tokens if len(token) > 2]
    
    # removing English stopwords
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    return tokens

# vectorizing using the tokenizer
custom_vectorizer = TfidfVectorizer(tokenizer= custom_tokeniser, max_features=3000, token_pattern=None, ngram_range=(1,3))

X_train_custom = custom_vectorizer.fit_transform(X_train)
X_test_custom = custom_vectorizer.transform(X_test)

# Training the Random Forest with the custom tokenizer
custom_random_forest = RandomForestClassifier(n_estimators=300, random_state=26)
custom_random_forest.fit(X_train_custom, y_train)
custom_random_forest_predictions = custom_random_forest.predict(X_test_custom)

# Printing results for Random Forest with custom tokenizer
print("Custom tokenizer Random Forest")
print(f1_score(y_test, custom_random_forest_predictions, average="macro"))
print(classification_report(y_test, custom_random_forest_predictions, zero_division=0))


# Training the SVM with linear kernel and custom tokenizer
custom_svm_classifier = SVC(kernel="linear")
custom_svm_classifier.fit(X_train_custom, y_train)
custom_svm_classifier_predictions = custom_svm_classifier.predict(X_test_custom)

# Printing results for ngram SVM
print("Custom tokenizer Linear SVM")
print(f1_score(y_test, custom_svm_classifier_predictions, average="macro"))
print(classification_report(y_test, custom_svm_classifier_predictions, zero_division=0))


