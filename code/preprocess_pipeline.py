from data_cleanup_line import *
from feature_extraction_line import *

import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
import sys
from os.path import exists


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

    
    essays = process_column(essays, 'TEXT', 'num_of_char', num_of_char) # Always missing 2 2568
    essays = process_column(essays, 'TEXT', 'num_any_words', num_any_word) # Correct 658, always 1 off
    essays = process_column(essays, 'TEXT','num_long_words', num_long_words) # Correct 109, 3 off, 112 after diff check
    essays = process_column(essays, 'TEXT','num_short_words', num_short_words) # Counterd 548, 2 off

    essays = process_column(essays, 'TEXT','num_sentences', num_sentences) # Correct 38
    essays = process_column(essays, 'TEXT', "num_diff_word_stop", num_diff_word_stop) # Correct 272
    essays = process_column(essays, 'TEXT', "num_diff_word_nstop", num_diff_word_nstop)
    essays = process_column(essays, 'TEXT', 'avg_sentence_length',avg_sentence_length)
    essays = process_column(essays, 'TEXT', 'avg_word_length',avg_word_length)

    essays = process_column(essays, 'TEXT', 'num_syllables',num_syllables)

    essays = process_column(essays, 'TEXT', 'most_freq_word_length',most_freq_word_length) 
    essays = process_column(essays, 'TEXT', 'most_freq_sentence_length',most_freq_sentence_length)

    # Readability
    essays = process_column(essays, 'TEXT', 'flesch_reading_ease',flesch_reading_ease)
    essays = process_column(essays, 'TEXT', 'flesch_kincaid_grade_level',flesch_kincaid_grade_level)
    essays = process_column(essays, 'TEXT', 'automated_readability_index',automated_readability_index)
    essays = process_column(essays, 'TEXT', 'LIX_readability',LIX_readability)
    essays = process_column(essays, 'TEXT', 'dale_chall_readability',dale_chall_readability)

    return essays

def main():
    processed_essays = None
    if (not exists('pre_processed.csv')):
        essays = pd.read_csv('essays.csv', encoding='cp1252')
        processed_essays = required_preprocess(essays)
        processed_essays.to_csv('pre_processed.csv', sep=',', encoding='utf-8', index = False)
    else:
        processed_essays = pd.read_csv('pre_processed.csv', encoding='utf-8')
    
    # test_line = processed_essays['TEXT'][5]
    # print(SMOG_readability(test_line))
    final_processed = feature_extract(processed_essays)
    print(final_processed.head(10))

    # final_processed.to_csv('final.csv', sep=',', encoding='utf-8', index = False)

if __name__ == "__main__":
    main()