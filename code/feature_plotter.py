# Require massive rewrites in terms of readability 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
sns.set_theme(style="darkgrid")

# ========== Basic Plotting ==========

# Bar plot for single field
def bar_plot(field,feature_df):
    sns.barplot(feature_df[field])

# Simple plotting field/column
def dis_plot(field, feature_df):
    sns.displot(feature_df[field])

# Simple plotting field/column
def hist_plot(field, feature_df, bins):
    sns.histplot(feature_df[field], bins=bins)

# Plot basic plots for all given columns (MAY NEED ADJUSTMENT OR REMOVAL)
def solo_feature_plot(col_names, name, func):
    for field in col_names:
        func(field)
        plt.savefig('./plots/' + name + '/' + field)

# ========== Multi variable plotting ==========

# Plot 2 fields against each other
def two_col_pairplot(field1, field2, feature_df):
    sns.relplot(x=field1, y=field2, data=feature_df, jitter=True)

# Plot every field again every field
def multi_pairplot(reduced_col, filename, feature_df):
    sns.pairplot(feature_df[reduced_col], markers=["o", "s"], corner=True)
    plt.savefig(f'./plots/feature_grouped/{filename}_plot')

# Plot every field against every other field into separate files
def indv_plotter(col_names):
    i = 0
    while (i < len(col_names)):
        j = i + 1
        while (j < len(col_names)):
            two_col_pairplot(col_names[i], col_names[j])
            plt.savefig('./plots/' + col_names[i] + "_" + col_names[j])
            j = j + 1

        i = i + 1

# Dual Histogram plots for given features
def compare_hist_plot(features, person_trait, feature_df, bins=10):
    n_df = feature_df[feature_df[person_trait] == 'n']
    y_df = feature_df[feature_df[person_trait] == 'y']
    for feature in features:
        plt.hist([n_df[feature], y_df[feature]], bins=bins, color=['orange','skyblue'], label=[f'n_{person_trait}',f'y_{person_trait}'])
        plt.xlabel(f"{feature}")
        plt.ylabel("Essay occurance")
        plt.title(f"{feature} occurance comparison histogram")
        plt.legend(loc='upper right')
        plt.savefig(f'./plots/compare_hist/{person_trait}_{feature}_compare_hist')
        plt.clf()

# Radar Diagram given features
def compare_radar_plot(features, person_trait, feature_df):
    n_df = feature_df[feature_df[person_trait] == 'n']
    y_df = feature_df[feature_df[person_trait] == 'y']
    n_r_mean = []
    y_r_mean = []
    for feature in features:
        # print(f'n_cEXT_{feature} mean: {n_df[feature].mean()}')
        # print(f'y_cEXT_{feature} mean: {y_df[feature].mean()}')
        n_r_mean.append(n_df[feature].mean())
        y_r_mean.append(y_df[feature].mean())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=n_r_mean,
        theta=features,
        fill='toself',
        name=f'N_{person_trait}'
    ))
    fig.add_trace(go.Scatterpolar(
        r=y_r_mean,
        theta=features,
        fill='toself',
        name=f'Y_{person_trait}'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        )),
    showlegend=False
    )

    fig.show()

def main():

    # ====== File Config ======

    fv_csv_file = 'final_v2.csv' # Raw Final Data
    zsc_csv_file = 'z_score_final_v2.csv' # Z Score data

    fv_feature_df = pd.read_csv(fv_csv_file, index_col = 0)
    zsc_feature_df = pd.read_csv(zsc_csv_file, index_col = 0)

    t_test_sig_file = './analysis_data/significant_features.csv' #significant features
    t_test_sig_df = pd.read_csv(t_test_sig_file, index_col = 0)
    cEXT_features = t_test_sig_df.loc['cEXT_t_stat','signf_features'].split(",")
    cNEU_features = t_test_sig_df.loc['cNEU_t_stat','signf_features'].split(",")
    cAGR_features = t_test_sig_df.loc['cAGR_t_stat','signf_features'].split(",")
    cCON_features = t_test_sig_df.loc['cCON_t_stat','signf_features'].split(",")
    cOPN_features = t_test_sig_df.loc['cOPN_t_stat','signf_features'].split(",")

    # ====== Feature Collection ======

    all_features = fv_feature_df.columns[6:36].tolist()
    directly_quantified_features = ['num_of_char','num_any_words','num_sentences','num_syllables']
    readability_features = ['flesch_reading_ease','flesch_kincaid_grade_level','automated_readability_index','LIX_readability','dale_chall_readability','SMOG_readability']
    pos_features = ['num_diff_pos','num_pos_coord_conj','num_pos_num','num_pos_det','num_pos_sub_conj','num_pos_adj','num_pos_aux','num_pos_noun','num_pos_adv','num_pos_verb']
    lex_sophistication_features = ['num_long_words','num_short_words','num_diff_word_nstop','avg_word_length','most_freq_word_length','type_token_ratio','hapax_legomena']

    significant_features = ['avg_word_length', 'LIX_readability', 'type_token_ratio', 'num_sentences','hapax_legomena']
    
    # ====== Multi plotter ======

    # multi_pairplot(all_features.columns)

    # ====== Complete Plotter ======

    # multi_pairplot(all_features,'fv2_all_features',fv_feature_df)
    # multi_pairplot(directly_quantified_features,'fv2_directly_quantified_features',fv_feature_df)
    # multi_pairplot(readability_features,'fv2_readability_features',fv_feature_df)
    # multi_pairplot(pos_features,'fv2_pos_features',fv_feature_df)
    # multi_pairplot(lex_sophistication_features,'fv2_lex_feature',fv_feature_df)

    # multi_pairplot(all_features,'zsc_all_features',zsc_feature_df)
    # multi_pairplot(directly_quantified_features,'zsc_directly_quantified_features',zsc_feature_df)
    # multi_pairplot(readability_features,'zsc_readability_features',zsc_feature_df)
    # multi_pairplot(pos_features,'fv2_pos_features',zsc_feature_df)
    # multi_pairplot(lex_sophistication_features,'zsc_lex_feature',zsc_feature_df)

    # ====== Feature Solo Plotting ======

    # solo_feature_plot(all_features, "displots", dis_plot)
    # solo_feature_plot(all_features, "histplots", hist_plot)

    # for feature in all_features:
    #     print(feature)
    #     sns.histplot(fv_feature_df[feature], bins=5)
    #     plt.savefig(f'./plots/histplots/bin_5/{feature}')
    #     plt.clf()

    # ====== Individual Plotting ======

    # two_col_pairplot('cEXT', 'num_sentences', fv_feature_df)
    # plt.savefig('./plots/cEXT_num_sentences')

    # two_col_pairplot('cEXT', 'avg_word_length', fv_feature_df)
    # plt.savefig('./plots/cEXT_avg_word_length')

    # two_col_pairplot('cEXT', 'LIX_readability', fv_feature_df)
    # plt.savefig('./plots/cEXT_LIX_readability')

    # two_col_pairplot('cEXT', 'num_pos_adj', fv_feature_df)
    # plt.savefig('./plots/cEXT_num_pos_adj')

    # sns.histplot(fv_feature_df['num_sentences'], bins=5)
    # plt.savefig('./plots/temp/num_of_char')

    # ========== Perrsonality Comparison Plotting ==========

    
    # compare_hist_plot(cEXT_features, "cEXT", fv_feature_df)
    # compare_hist_plot(cNEU_features, "cNEU", fv_feature_df)
    # compare_hist_plot(cAGR_features, "cAGR", fv_feature_df)
    # compare_hist_plot(cCON_features, "cCON", fv_feature_df)
    # compare_hist_plot(cOPN_features, "cOPN", fv_feature_df)

    # ======== Personality Features Radar ========

    compare_radar_plot(cEXT_features, 'cEXT', zsc_feature_df)
    compare_radar_plot(cNEU_features, 'cNEU', zsc_feature_df)
    compare_radar_plot(cAGR_features, 'cAGR', zsc_feature_df)
    compare_radar_plot(cCON_features, 'cCON', zsc_feature_df)
    compare_radar_plot(cOPN_features, 'cOPN', zsc_feature_df)

    # compare_radar_plot(all_features, 'cEXT', zsc_feature_df)
    # compare_radar_plot(all_features, 'cNEU', zsc_feature_df)
    # compare_radar_plot(all_features, 'cAGR', zsc_feature_df)
    # compare_radar_plot(all_features, 'cCON', zsc_feature_df)
    # compare_radar_plot(all_features, 'cOPN', zsc_feature_df)

    # ====== SANDBOX ======

    # n_cEXT_df = fv_feature_df[fv_feature_df['cEXT'] == 'n']
    # y_cEXT_df = fv_feature_df[fv_feature_df['cEXT'] == 'y']
    # sns.pairplot(n_cEXT_df[all_features], markers=["o", "s"], corner=True)
    # sns.pairplot(y_cEXT_df[all_features], markers=["o", "s"], corner=True)
    # for feature in all_features:
    #     plt.hist([n_cEXT_df[feature], y_cEXT_df[feature]], bins=20)
    #     sns.histplot(y_cEXT_df[feature], color='orange', stacked=True)
    #     plt.savefig(f'./plots/temp/cEXT_{feature}_compare')
    #     plt.clf()

    

if __name__ == "__main__":
    main()