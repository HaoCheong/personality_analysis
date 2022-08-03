# Require massive rewrites in terms of readability 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_theme(style="darkgrid")



# Simple plotting field/column
def dis_plot(field, feature_df):
    sns.displot(feature_df[field])

# Simple plotting field/column
def hist_plot(field, feature_df):
    sns.histplot(feature_df[field])

# Plot 2 fields against each other
def two_col_pairplot(field1, field2, feature_df):
    sns.relplot(x=field1, y=field2, data=feature_df)

# Plot every field again every field
def multi_pairplot(reduced_col, filename, feature_df):
    sns.pairplot(feature_df[reduced_col], markers=["o", "s"], corner=True)
    plt.savefig(f'./plots/{filename}_plot')

# Plot every field again every field into separate files
def indv_plotter(col_names):
    i = 0
    while (i < len(col_names)):
        j = i + 1
        while (j < len(col_names)):
            # print(col_names[i],col_names[j])
            two_col_pairplot(col_names[i], col_names[j])
            plt.savefig('./plots/' + col_names[i] + "_" + col_names[j])
            j = j + 1

        i = i + 1

# Bar plot for single field
def bar_plot(field,feature_df):
    sns.barplot(feature_df[field])

# Generate 
def solo_feature_plot(col_names, name, func):
    for field in col_names:
        func(field)
        plt.savefig('./plots/' + name + '/' + field)

def main():

    # ====== File Usage ======
    fv_csv_file = 'final_v2.csv'
    zsc_csv_file = 'z_score_final_v2.csv'

    fv_feature_df = pd.read_csv(fv_csv_file, index_col = 0)
    zsc_feature_df = pd.read_csv(zsc_csv_file, index_col = 0)

    print(zsc_feature_df.head(10))

    # ====== Feature Collection ======
    fv_all_features = fv_feature_df.columns[6:36].tolist()
    fv_directly_quantified_features = fv_feature_df[['num_of_char','num_any_words','num_sentences','num_syllables']].columns.tolist()
    fv_readability_features = fv_feature_df[['flesch_reading_ease','flesch_kincaid_grade_level','automated_readability_index','LIX_readability','dale_chall_readability','SMOG_readability']].columns.tolist()
    fv_pos_features = fv_feature_df[['num_diff_pos','num_pos_coord_conj','num_pos_num','num_pos_det','num_pos_sub_conj','num_pos_adj','num_pos_aux','num_pos_noun','num_pos_adv','num_pos_verb']].columns.tolist()

    zsc_all_features = zsc_feature_df.columns[6:36].tolist()
    zsc_directly_quantified_features = zsc_feature_df[['num_of_char','num_any_words','num_sentences','num_syllables']].columns.tolist()
    zsc_readability_features = zsc_feature_df[['flesch_reading_ease','flesch_kincaid_grade_level','automated_readability_index','LIX_readability','dale_chall_readability','SMOG_readability']].columns.tolist()
    zsc_pos_features = zsc_feature_df[['num_diff_pos','num_pos_coord_conj','num_pos_num','num_pos_det','num_pos_sub_conj','num_pos_adj','num_pos_aux','num_pos_noun','num_pos_adv','num_pos_verb']].columns.tolist()

    # plt.savefig('./plots/z_score_plot')

    # ====== Multi plotter ======
    # multi_pairplot(all_features.columns)

    # ====== Complete Plotter ======
    # multi_pairplot(fv_all_features,'fv2_all_features',fv_feature_df)
    # multi_pairplot(fv_directly_quantified_features,'fv2_directly_quantified_features',fv_feature_df)
    # multi_pairplot(fv_readability_features,'fv2_readability_features',fv_feature_df)
    # multi_pairplot(fv_pos_features,'fv2_pos_features',fv_feature_df)

    # multi_pairplot(zsc_all_features,'zsc_all_features',zsc_feature_df)
    # multi_pairplot(zsc_directly_quantified_features,'zsc_directly_quantified_features',zsc_feature_df)
    # multi_pairplot(zsc_readability_features,'zsc_readability_features',zsc_feature_df)
    # multi_pairplot(zsc_pos_features,'zsc_pos_features',zsc_feature_df)

    # ====== Feature Solo Plotting ======
    # solo_feature_plot(all_features, "displots", dis_plot)
    # solo_feature_plot(all_features, "histplots", hist_plot)

    # ====== Individual Plotting ======
    # dis_plot('num_of_char')
    two_col_pairplot('cEXT', 'num_sentences', fv_feature_df)
    plt.savefig('./plots/cEXT_num_sentences')

    two_col_pairplot('cEXT', 'avg_word_length', fv_feature_df)
    plt.savefig('./plots/cEXT_avg_word_length')

    two_col_pairplot('cEXT', 'LIX_readability', fv_feature_df)
    plt.savefig('./plots/cEXT_LIX_readability')

    # plt.savefig('./plots/temp')

if __name__ == "__main__":
    main()