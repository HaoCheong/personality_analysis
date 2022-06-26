from data_cleanup import *
from feature_extraction import *

import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
import re
stop = stopwords.words('english')



def main():

    # Pass essay into pd dataframe
    essays = pd.read_csv('essays.csv', encoding='cp1252')

    # Preliminary Clean up
    essays = lower_casing(essays)
    essays = remove_long_white_space(essays)

    # Punctuation dependent feature extraction
    # Word need to split: word + stopword, just stop word, just word

    essays = num_sentences(essays)
    
    essays = remove_punctuation(essays)
    essays = num_any_words(essays)

    essays = remove_long_white_space(essays)
    # Stop word dependent feature extraction



    essays = remove_stopwords(essays)
    essays = remove_long_white_space(essays)
    # Remaining clean up

    print(essays.head(10))
    
    essays.to_csv("processed.csv", sep=',', index=False)
    

if __name__ == "__main__":
    main()