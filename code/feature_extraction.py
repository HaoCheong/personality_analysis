import numpy as np
import pandas as pd
import re

short_word_char = 5
long_word_char = 5

# Adds column to count num char

# Lex Soph - Count char in string
def num_of_char(essays):
    essays['num_char'] = essays['TEXT'].apply(lambda x: len(''.join([word for word in x.split()])))
    return essays

# Remove numbers for text processing
def remove_num(line):
    return ''.join([i for i in line if not i.isdigit()])

def remove_all_large_space(line):
    return re.sub(r' +', ' ', line.strip())

# Lex Soph - Count long words
def num_long_words(essays):
    essays['num_long_words'] = essays['TEXT'].apply(lambda x: len((remove_all_large_space(re.sub(r'\b\w{1,5}\b', '', remove_num(x)))).split(" ")))
    return essays

# Lex Soph - Count short words
def num_short_words(essays):
    essays['num_short_words'] = essays['TEXT'].apply(lambda x: len((remove_all_large_space(re.sub(r'\b\w{6,}\b', '', remove_num(x)))).split(" ")))
    return essays

# Lex Soph - Count words (with number removed)
def num_any_words(essays):
    essays['num_any_words'] = essays['TEXT'].apply((lambda x: len(remove_num(x).split(" "))))
    return essays

def num_diff_words(essays):
    essays['num_diff_words'] = essays['TEXT'].apply(lambda x: len(set(x.split(" "))))
    return essays

def main():
    essays = pd.read_csv('filtered.csv', encoding='cp1252')
    essays = num_of_char(essays)
    essays = num_any_words(essays)
    essays = num_long_words(essays)
    essays = num_short_words(essays)
    essays = num_diff_words(essays)
    print(essays.head(10))


if __name__ == "__main__":
    main()