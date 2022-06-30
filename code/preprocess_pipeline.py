from data_cleanup_line import *
from feature_extraction_line import *

import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords

import re
stop = stopwords.words('english')

pd.set_option('display.max_columns', None)

def process_column(table, input_column, output_column, func):
    table[output_column] = table[input_column].apply(lambda x: func(x))
    return table

#Necessary preprocessing irrespective of features
def required_preprocess(essays):
    # lower casing
    essays['TEXT'] = essays['TEXT'].apply(lambda x: lower_casing(x))

    # Replace counterparts
    essays['TEXT'] = essays['TEXT'].apply(lambda x: replace_to_counterpart(x))

    # Weird character
    essays['TEXT'] = essays['TEXT'].apply(lambda x: remove_weird_char(x))

    # Spelling fix with NLTK
    # essays['TEXT'] = essays['TEXT'].apply(lambda x: spelling_fix(x))

    # Remove long spaces
    essays['TEXT'] = essays['TEXT'].apply(lambda x: remove_all_large_space(x))
    
    return essays

# Pipeline for feature extraction
def feature_extract(essays):
    essays = process_column(essays, 'TEXT', 'num_of_char', num_of_char) # Always missing 2568
    essays = process_column(essays, 'TEXT', 'num_any_words', num_any_word) # Correct 658
    essays = process_column(essays, 'TEXT','num_long_words', num_long_words) # Correct 112
    # essays = process_column(essays, 'TEXT','num_short_words', num_short_words) 

    # essays = process_column(essays, 'TEXT','num_sentence', num_sentences)
    # essays = process_column(essays, 'TEXT', "num_diff_word_stop", num_diff_word_stop)
    # essays = process_column(essays, 'TEXT', "num_diff_word_nstop", num_diff_word_nstop)
    # essays = process_column(essays, 'TEXT', 'avg_sentence_length',avg_sentence_length)

    return essays

def main():
    essays = pd.read_csv('essays.csv', encoding='cp1252')
    processed_essays = required_preprocess(essays)

    processed_essays.to_csv('pre_processed.csv', sep=',', encoding='utf-8', index = False)

    test_line = processed_essays['TEXT'][0]
    
    final_processed = feature_extract(processed_essays)

    # print((num_long_words(test_line)))
    print(final_processed.head(1))

    # final_processed.to_csv('final.csv', sep=',', encoding='utf-8', index = False)

if __name__ == "__main__":
    main()