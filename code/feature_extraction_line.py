# Feature Extraction Function (Per Line basis)

# Run on the assumptions that the line has never been processed before,
# Hence clean up is done accordingly per line

# Might need a freq_dict helper

# Singleton pattern might reduce processing
# Shift code to a row processesing, not a column processing (caching can reduce processing time)

import re
from data_cleanup_line import *
import pandas as pd
import math 
import spacy as sp
from nltk.corpus import stopwords

nlp = sp.load('en_core_web_lg')
stop = stopwords.words('english')

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
    sen_freq_dict = {}
    for sen in sentence_array:
        # print(sen)
        if sen_freq_dict.get(len(sen)) == None:
            sen_freq_dict[len(sen)] = 1
        else:
            sen_freq_dict[len(sen)] = sen_freq_dict[len(sen)] + 1
    
    max_val = 0
    max_key = ""
    for key in sen_freq_dict.keys():
        if sen_freq_dict[key] > max_val and key != 1: #Mitigates the . . . sentences but might need future adjusting
            max_val = sen_freq_dict[key]
            max_key = key
    
    return max_key

# Number of stop words
def num_stop_words(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)

    word_list = newline.split(" ")

    stop_count = 0 
    stop_list = []

    for word in word_list:
        if word in stop:
            stop_list.append(word)

    return stop_list 
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

# Fletch Kincaid Grade Level
def flesch_kincaid_grade_level(line):
    n_nstop = num_nstop_word(line)
    n_sen = num_sentences(line)
    n_sylb = num_syllables(line)

    comp_1 = 0.39 * float(n_nstop/n_sen)
    comp_2 = 11.8 * float(n_sylb/n_nstop)
    fkgl = float(comp_1 + comp_2 - 15.59)
    return fkgl

# Fletch Reading Ease Index
def flesch_reading_ease(line):
    n_nstop = num_nstop_word(line)
    n_sen = num_sentences(line)
    n_sylb = num_syllables(line)
    comp_1 = 1.015 * float(n_nstop/n_sen)
    comp_2 = 84.6 * float(n_sylb/n_nstop)

    fre = float(206.835 - comp_1 - comp_2)
    return fre

# Automated Readability Measure
def automated_readability_index(line):
    n_char = num_of_char(line)
    n_any_word = num_any_word(line)
    n_sen = num_sentences(line)

    comp_1 = 4.71 * float(n_char/n_any_word)
    comp_2 = 0.5 * float(n_any_word/n_sen)

    ari = float(comp_1 + comp_2 - 21.43)
    return ari

# LIX readability measure
def LIX_readability(line):
    n_any_word = num_any_word(line)
    n_long_word = num_long_words(line)
    n_period = len(re.split(r'([\.:]|(\s[a-z]))',line))

    comp_1 = float(n_any_word/n_period)
    comp_2 = float((n_long_word * 100)/n_any_word)
    lix = float(comp_1 + comp_2)

    return lix

# Dale Chall Reasabilitty Measure
def dale_chall_readability(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    word_list = newline.split()
    
    diffc_count = 0
    with open('../data_set_master/DaleChallEasyWordList.txt','r') as f:
        easy_words = f.read().splitlines()
        for word in word_list:
            if (word not in easy_words):
                diffc_count = diffc_count + 1

    n_sen = num_sentences(line)
    n_any_word = num_any_word(line)
    diffc_perc = (diffc_count/n_any_word) * 100

    comp_1 = float(0.1579 * (diffc_perc))
    comp_2 = float(0.0496 * (n_any_word/n_sen))
    dcr = comp_1 + comp_2
    if (diffc_perc > 5):
        dcr = dcr + 3.6365 
    
    return dcr

# Simple Measure of Gobbledygook Readability Measure
def SMOG_readability(line):
    polysylb_word = 0
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    word_list = newline.split()

    for word in word_list:
        if ((word_sylb_dict.get(word) != None) and (word_sylb_dict[word] >= 3)):
            polysylb_word = polysylb_word + 1
    
    smog = float(3 + math.sqrt(polysylb_word))
    return smog


# ---------- LEXICAL DIVERSITY ----------

# The ratio of unique words used
def type_token_ratio(line):
    return float(num_diff_word_stop(line)/num_any_word(line))

# Hapax Legomena, count number of words used once
def hapax_legomena(line):

    word_freq_dict = {}

    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_all_large_space(newline)
    word_list = newline.split(" ")

    for word in word_list:
        if word_freq_dict.get(word) == None:
            word_freq_dict[word] = 1
        else:
            word_freq_dict[word] = word_freq_dict[word] + 1

    hpx_lgmn = 0

    for key in word_freq_dict.keys():
        if word_freq_dict[key] == 1:
            hpx_lgmn = hpx_lgmn + 1

    return hpx_lgmn

# ----------- GRAMMAR ----------

# Number of unique Part Of Speech Tags
def num_diff_pos(line):
    newline = remove_num(line)
    newline = remove_punctuation(newline)
    newline = remove_stop_words(newline)
    newline = remove_all_large_space(newline)

    doc = nlp(newline)
    pos_set = set()

    for i in range(1, len(newline.split(" "))):
        pos_set.add(doc[i].pos_)

    return len(pos_set)
