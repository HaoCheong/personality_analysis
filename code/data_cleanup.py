import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
import re
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
    essays['TEXT'] = essays['TEXT'].apply(lambda x: x.replace(u"\u2019", ""))
    return essays

def remove_long_white_space(essays):
    essays['TEXT'] = essays['TEXT'].apply(lambda x: x.strip())
    essays['TEXT'] = essays['TEXT'].apply(lambda x: re.sub(r' +', ' ', x))
    return essays

def main():
    
    essays = pd.read_csv('essays.csv', encoding='cp1252')

    # Lower Case the given text
    essays = lower_casing(essays)

    # Remove the stopword in the given text
    essays = remove_stopwords(essays)

    # Remove long white space removal
    essays = remove_long_white_space(essays)

    # Remove Punctuation:
    essays = remove_punctuation(essays)

    essays = remove_long_white_space(essays)

    # Print essay head (test)
    print(essays.head(35))
    essays.to_csv("filtered.csv", sep=',', index=False)


if __name__ == "__main__":
    main()