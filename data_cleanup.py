import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Lower Casing
def lower_casing(essays):
    essays['TEXT'] = essays['TEXT'].str.lower()
    return essays

def remove_stopwords(essays):
    essays['TEXT'] = essays['TEXT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return essays

def remove_punctuation(essays):
    essays['TEXT'] = essays['TEXT'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))
    return essays

def main():
    essays = pd.read_csv('essays.csv',encoding='cp1252')

    # Lower Case the given text
    essays = lower_casing(essays)

    # Remove the stopword in the given text
    essays = remove_stopwords(essays)
    
    # Remove Punctuation:
    essays = remove_punctuation(essays)

    # Print essay head (test)
    print(essays.head())
    # essays.to_csv("filtered.csv", sep=',')


if __name__ == "__main__":
    main()