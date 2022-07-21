import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_theme(style="darkgrid")

''' ALL FEATURE
cEXT
cNEU
cAGR
cCON
cOPN

num_of_char
num_any_words
num_long_words
num_short_words
num_sentences
num_diff_word_stop
num_diff_word_nstop
avg_sentence_length
avg_word_length
num_syllables
most_freq_word_length
most_freq_sentence_length

flesch_reading_ease
flesch_kincaid_grade_level
automated_readability_index
LIX_readability
dale_chall_readability
SMOG_readability
type_token_ratio
hapax_legomena

num_diff_pos
num_pos_coord_conj
num_pos_num
num_pos_det
num_pos_sub_conj
num_pos_adj
num_pos_aux
num_pos_noun
num_pos_adv
num_pos_verb
'''

# print(sys.argv[1])
if len(sys.argv) != 2:
    print('USAGE: python3 feature_plotter.py <csv_file>')
    exit()

csv_file = sys.argv[1]
feature_df = pd.read_csv(csv_file, index_col = 0)

# Simple plotting field/column
def basic_plot(field):
    sns.displot(feature_df[field])

# Plot 2 fields against each other
def two_col_pairplot(field1, field2):
    sns.relplot(x=field1, y=field2, data=feature_df)

# Plot every field again every field
def multi_pairplot(reduced_col):
    sns.pairplot(feature_df[reduced_col], markers=["o", "s"])
    
def mass_plotter(col_names):
    i = 0
    while (i < len(col_names)):
        j = i + 1
        while (j < len(col_names)):
            # print(col_names[i],col_names[j])
            two_col_pairplot(col_names[i], col_names[j])
            plt.savefig('./plots/' + col_names[i] + "_" + col_names[j])
            j = j + 1

        i = i + 1

def main():
    # all_features = feature_df.columns[6:36].tolist()
    reduced = feature_df[['num_long_words','num_short_words','num_diff_word_stop','num_diff_word_nstop','num_of_char','num_any_words']].copy()
    # multi_pairplot(reduced.columns)
    mass_plotter(reduced.columns)
    # plt.savefig('./plots/' + plot_filename)

if __name__ == "__main__":
    main()