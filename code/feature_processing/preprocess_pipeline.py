from data_cleanup_line import *
from feature_extraction_sgt import *

import numpy as np
import pandas as pd

import string
from nltk.corpus import stopwords
import sys
from os.path import exists

import re
stop = stopwords.words('english')

TEXT_COL_NAME = 'TEXT'
PRE_PROCESS_FILE = '../mark_pre_processed.csv'
FINAL_PROCESS_FILE = '../MARK_final_v1.csv'
DATASET_FILE = '../MARK1012_Essays_20221028.csv'

pd.set_option('display.max_columns', None)

# Pipeline of feature extraction, order dependent
feature_pipeline = []

def add_feature(output_column, func):
    feature_tuple = (output_column, func)
    feature_pipeline.append(feature_tuple)

def process_feature_pipeline(essays):

    # For each row, cache the text
    for index in range(len(essays.index)):
        print(index)
        refresh_cache()
        row_text = essays.at[index,TEXT_COL_NAME]
        # Process along the feature pipeline
        for feature in feature_pipeline:
            essays.at[index,feature[0]] = feature[1](row_text)
    
    return essays

def process_column(table, input_column, output_column, func):
    table[output_column] = table[input_column].apply(lambda x: func(x))
    return table

#Necessary preprocessing irrespective of features
def required_preprocess(essays):

    # lower casing
    essays[TEXT_COL_NAME] = essays[TEXT_COL_NAME].apply(lambda x: lower_casing(x))

    # Replace none utf8 counterparts
    essays[TEXT_COL_NAME] = essays[TEXT_COL_NAME].apply(lambda x: replace_to_counterpart(x))

    # Weird character
    essays[TEXT_COL_NAME] = essays[TEXT_COL_NAME].apply(lambda x: remove_weird_char(x))

    # Spelling fix with NLTK
    # essays[TEXT_COL_NAME] = essays[TEXT_COL_NAME].apply(lambda x: spelling_fix(x))

    # Remove long spaces
    essays[TEXT_COL_NAME] = essays[TEXT_COL_NAME].apply(lambda x: remove_all_large_space(x))
    
    return essays

# Feature extract 2:
def feature_extract_row(processed_essays):
    add_feature('num_of_char', num_of_char)
    add_feature('num_any_words', num_any_word)
    add_feature('num_long_words', num_long_words)
    add_feature('num_short_words', num_short_words)
    add_feature('num_sentences', num_sentences)

    add_feature("num_diff_word_stop", num_diff_word_stop)
    add_feature("num_diff_word_nstop", num_diff_word_nstop)
    add_feature('avg_sentence_length',avg_sentence_length)
    add_feature('avg_word_length',avg_word_length)
    add_feature('num_syllables',num_syllables)

    add_feature('most_freq_word_length',most_freq_word_length)
    add_feature('most_freq_sentence_length',most_freq_sentence_length)
    add_feature('flesch_reading_ease',flesch_reading_ease)
    add_feature('flesch_kincaid_grade_level',flesch_kincaid_grade_level)
    add_feature('automated_readability_index',automated_readability_index)

    add_feature('LIX_readability',LIX_readability)
    add_feature('dale_chall_readability',dale_chall_readability)
    add_feature('SMOG_readability',SMOG_readability)
    add_feature('type_token_ratio',type_token_ratio)
    add_feature('hapax_legomena',hapax_legomena)

    add_feature('num_diff_pos',num_diff_pos)

    add_feature('num_pos_coord_conj',num_pos_coord_conj)
    add_feature('num_pos_num',num_pos_num)
    add_feature('num_pos_det',num_pos_det)
    add_feature('num_pos_sub_conj',num_pos_sub_conj)
    add_feature('num_pos_adj',num_pos_adj)
    add_feature('num_pos_aux',num_pos_aux)
    add_feature('num_pos_noun',num_pos_noun)
    add_feature('num_pos_adv',num_pos_adv)
    add_feature('num_pos_verb',num_pos_verb)

    return process_feature_pipeline(processed_essays)

# Pipeline for feature extraction
def feature_extract_column(essays):

    # Pure Lexical Features
    essays = process_column(essays, TEXT_COL_NAME, 'num_of_char', num_of_char)
    essays = process_column(essays, TEXT_COL_NAME, 'num_any_words', num_any_word)
    essays = process_column(essays, TEXT_COL_NAME,'num_long_words', num_long_words)
    essays = process_column(essays, TEXT_COL_NAME,'num_short_words', num_short_words)
    essays = process_column(essays, TEXT_COL_NAME,'num_sentences', num_sentences)

    essays = process_column(essays, TEXT_COL_NAME, "num_diff_word_stop", num_diff_word_stop)
    essays = process_column(essays, TEXT_COL_NAME, "num_diff_word_nstop", num_diff_word_nstop)
    essays = process_column(essays, TEXT_COL_NAME, 'avg_sentence_length',avg_sentence_length)
    essays = process_column(essays, TEXT_COL_NAME, 'avg_word_length',avg_word_length)
    essays = process_column(essays, TEXT_COL_NAME, 'num_syllables',num_syllables)

    essays = process_column(essays, TEXT_COL_NAME, 'most_freq_word_length',most_freq_word_length) 
    essays = process_column(essays, TEXT_COL_NAME, 'most_freq_sentence_length',most_freq_sentence_length)

    # Readability
    essays = process_column(essays, TEXT_COL_NAME, 'flesch_reading_ease',flesch_reading_ease)
    essays = process_column(essays, TEXT_COL_NAME, 'flesch_kincaid_grade_level',flesch_kincaid_grade_level)
    essays = process_column(essays, TEXT_COL_NAME, 'automated_readability_index',automated_readability_index)
    
    essays = process_column(essays, TEXT_COL_NAME, 'LIX_readability',LIX_readability)
    essays = process_column(essays, TEXT_COL_NAME, 'dale_chall_readability',dale_chall_readability)
    essays = process_column(essays, TEXT_COL_NAME, 'SMOG_readability',SMOG_readability)

    # Lexical Diversity
    essays = process_column(essays, TEXT_COL_NAME, 'type_token_ratio',type_token_ratio)
    essays = process_column(essays, TEXT_COL_NAME, 'hapax_legomena',hapax_legomena)

    # Part of Speech
    essays = process_column(essays, TEXT_COL_NAME, 'num_diff_pos',num_diff_pos)
    return essays

def main():

    # Preprocess the raw text prior if preprocess not done yet
    processed_essays = None
    if (not exists(PRE_PROCESS_FILE)):
        essays = pd.read_csv(DATASET_FILE)
        # print(essays['Text'].head(1))
        processed_essays = required_preprocess(essays)
        processed_essays.to_csv(PRE_PROCESS_FILE, sep=',', encoding='utf-8', index = False)
    else:
        processed_essays = pd.read_csv(PRE_PROCESS_FILE, encoding='utf-8')
    
    # test_line = processed_essays[TEXT_COL_NAME][5]
    # spelling_fix_tb(test_line)

    # Test reduced essays
    # reduced_essays = processed_essays.iloc[:3,:]

    final_processed = feature_extract_row(processed_essays)
    print(final_processed.head(10))

    final_processed.to_csv(FINAL_PROCESS_FILE, sep=',', encoding='utf-8', index = False)

if __name__ == "__main__":
    main()