# Feature Extraction Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re
from data_cleanup_line import *

# Number of characters (including stop words + punctuation)
def num_of_char(line):
    newline = remove_all_space(line)
    return len(newline)

# Number of long words, defined in regex
def num_long_words(line):
    newline = remove_num(line)
    newline = remove_punctuation(line)
    newline = re.sub(r'\b\w{1,5}\b', '', newline)
    newline = remove_all_large_space(newline)
    return len(newline.split(" "))

# Number of short words, defined in regex
def num_short_words(line):
    newline = remove_num(line)
    newline = remove_punctuation(line)
    newline = re.sub(r'\b\w{6,}\b', '', newline)
    newline = remove_all_large_space(newline)
    return len(newline.split(" "))

# Number of any words (needs clean up)
def num_any_word(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    return len(newline.split(" "))

# Number of different word + stop words
def num_diff_word_stop(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    return len(set(newline.split(" ")))

# Number of different word + no stop words
def num_diff_word_nstop(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_stop_words(newline)
    newline = remove_all_large_space(newline)
    return len(set(newline.split(" ")))

# Number of sentences (Tokenised based on ". ")
# Need to include ?, ..., !, and other enders
def num_sentences(line):
    return len(re.split(r'[(. )(! )(? )]', line))

# Average sentences length (Total word count / sentence count)
def avg_sentence_length(line):
    return float(num_any_word(line))/(num_sentences(line))
