# Feature Extraction Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

import re
from data_cleanup_line import *
import pandas as pd

# --------Helpers--------
def get_word_sylb_dict():
    print("USED")
    word_sylb_dict = {}
    sylb_df = pd.read_csv('syllable_count_ndup.csv')
    for index, row in sylb_df.iterrows():
        # print(row[2])
        word_sylb_dict[row[1]] = row[2]

    return word_sylb_dict

## GLOBAL
word_sylb_dict = get_word_sylb_dict()


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

# Number of any words (not including stop words)
def num_nstop_word(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_stop_words(newline)
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
    return len(re.split(r'[.!?]{1}', line))

# Average sentences length (Total word count / sentence count)
def avg_sentence_length(line):
    return float(num_any_word(line))/(num_sentences(line))

def avg_word_length(line):
    return float(num_of_char(line)/num_any_word(line))

# Most frequent word length
def most_freq_word_length(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    
    word_array = newline.split(" ")
    freq_dict = {}
    for word in word_array:
        if freq_dict.get(len(word)) == None:
            freq_dict[len(word)] = 1
        else:
            freq_dict[len(word)] = freq_dict[len(word)] + 1
    
    max_val = 0
    max_key = ""

    for key in freq_dict.keys():
        if freq_dict[key] > max_val:
            max_val = freq_dict[key]
            max_key = key
    
    return max_key

# Most frequent sentence length
def most_freq_sentence_length(line):
    sentence_array = re.split(r'[.!?]{1}', line)
    freq_dict = {}
    for sen in sentence_array:
        # print(sen)
        if freq_dict.get(len(sen)) == None:
            freq_dict[len(sen)] = 1
        else:
            freq_dict[len(sen)] = freq_dict[len(sen)] + 1
    
    max_val = 0
    max_key = ""
    for key in freq_dict.keys():
        if freq_dict[key] > max_val and key != 1: #Mitigates the . . . sentences but might need future adjusting
            max_val = freq_dict[key]
            max_key = key
    
    return max_key

# Number of stop words

# Number of syllables, known words (Might need changing)
def num_syllables(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_stop_words(newline)
    newline = remove_all_large_space(newline)
    word_list = newline.split(" ")
    total_sylb_count = 0
    for word in word_list:
        if (word_sylb_dict.get(word) != None):
            total_sylb_count = total_sylb_count + word_sylb_dict[word]

    return total_sylb_count

# --------READABILITY INDEX--------
def flesch_kincaid_grade_level(line):
    num_nstop_word_var = num_nstop_word(line)
    num_sentences_var = num_sentences(line)
    num_syllables_var = num_syllables(line)

    comp_1 = 0.39 * float(num_nstop_word_var/num_sentences_var)
    comp_2 = 11.8 * float(num_syllables_var/num_nstop_word_var)
    fkgl = float(comp_1 + comp_2 - 15.59)
    return fkgl


