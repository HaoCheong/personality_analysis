from data_cleanup_line import *
from feature_extraction_line import *

import numpy as np
import pandas as pd
import spacy
import string
from nltk.corpus import stopwords
import re
stop = stopwords.words('english')

def process_column(table, input_column, output_column, func):
    table[output_column] = table[input_column].apply(lambda x: func(x))
    return table


def main():
    essays = pd.read_csv('essays.csv', encoding='cp1252')
    essays = process_column(essays, 'TEXT', 'num_of_char', num_of_char)
    essays = process_column(essays, 'TEXT', 'num_any_words', num_any_word)
    essays = process_column(essays, 'TEXT','num_long_words', num_long_words)
    essays = process_column(essays, 'TEXT','num_short_words', num_short_words)

    essays = process_column(essays, 'TEXT','num_sentence', num_sentences)
    essays = process_column(essays, 'TEXT', "num_diff_word_stop", num_diff_word_stop)
    essays = process_column(essays, 'TEXT', "num_diff_word_nstop", num_diff_word_nstop)
    essays = process_column(essays, 'TEXT', 'avg_sentence_length',avg_sentence_length)
    print(essays.head(10))

if __name__ == "__main__":
    main()