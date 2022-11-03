import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import scipy as sp
from scipy.stats import f_oneway
import math
import plotly.express as px

# ===== STARTUP =====

fv_csv_file = '../final_v2.csv' # Raw Final Data
zsc_csv_file = '../z_score_final_v2.csv' # Z Score data

fv_feature_df = pd.read_csv(fv_csv_file, index_col = 0)
zsc_feature_df = pd.read_csv(zsc_csv_file, index_col = 0)

all_personality = ['cEXT','cNEU', 'cAGR', 'cCON', 'cOPN']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def two_col_pairplot(field1, field2, feature_df):
    sns.relplot(x=field1, y=field2, data=feature_df)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_sig_corr():
    corr_df = pd.read_csv('../analysis_data/correlation/ps_corr.csv')
    lo_sig_corr_df = corr_df[corr_df['ps_corr'] < -0.9]
    hi_sig_corr_df = corr_df[corr_df['ps_corr'] > 0.9]
    sig_corr_df = pd.concat([lo_sig_corr_df, hi_sig_corr_df])

    return sig_corr_df

# ===== EXTRACT ALL SIGNIFICANT CORRELATION =====

# Pair plot all the those of significant correlation
def pairplot_sig_correlation():
    sig_corr_df = get_sig_corr()
    for index, row in sig_corr_df.iterrows():
        two_col_pairplot(row[0], row[1], fv_feature_df)
        plt.savefig('../plots/sig_corr_pairplot/sig_corr_{}_{}'.format(row[0], row[1]))
        plt.clf()

# Get a list of features and their significant correlation
def feature_to_sig_corr_feature():
    sig_corr_df = get_sig_corr()
    sig_feats = sig_corr_df['feature_1'].values.tolist() + sig_corr_df['feature_2'].values.tolist()
    no_dup_sig_feat = list(set(sig_feats))
    all_feat = []
    for feat in no_dup_sig_feat:
    # feat = 'num_of_char'
        feat_sig_feat = []
        
        left_side = sig_corr_df[sig_corr_df['feature_1'] == feat]
        for i, row in left_side.iterrows():
            feat_sig_feat.append(row[1])

        right_side = sig_corr_df[sig_corr_df['feature_2'] == feat]
        for i, row in right_side.iterrows():
            feat_sig_feat.append(row[0])

        res = [feat,", ".join(feat_sig_feat), len(feat_sig_feat)]
        all_feat.append(res)


    feature_to_sig_feature = pd.DataFrame(all_feat, columns=['feature','sig_feature','count']).sort_values(by=['count'], ascending=False)
    feature_to_sig_feature.to_csv('../analysis_data/feat_to_sig_feat.csv', sep=',', encoding='utf-8', index = False)

# Starting at the top feature, loop through the top level and if the word appear, set their respective row to exclude true
# def set_feature_exclude():
#     feat_to_sig_feat_df = pd.read_csv('../analysis_data/feat_to_sig_feat.csv')
#     feat_to_sig_feat_df['exclude'] = False
#     # print(feat_to_sig_feat_df)
#     for i, row in feat_to_sig_feat_df.iterrows():
#         to_exclude = row[1].split(', ')
#         for te in to_exclude:
#             # print(te)
#             # print(feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te]['exclude'])
#             if feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te]['exclude'].bool == True:
#                 # print("SKIP")
#                 continue
#             else:
#                 # print("NO SKIP")
#                 feat_to_sig_feat_df.loc[feat_to_sig_feat_df['feature'] == te].exclude = True

    # print(feat_to_sig_feat_df)

def clean_feat_to_sig_feat():
    pass

if __name__ == "__main__":
    pass
    # pairplot_sig_correlation()
    # feature_to_sig_corr_feature()
    # set_feature_exclude()