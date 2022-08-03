import sys

import numpy as np
from scipy.stats import skew, kurtosis
import scipy.stats as stats

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Usage check, requires CSV file to process
if len(sys.argv) != 2:
    print('USAGE: python3 feature_plotter.py <csv_file>')
    exit()

# Global
csv_file = sys.argv[1]
feature_df = pd.read_csv(csv_file, index_col = 0)
analysis_df = {}

# Calculate pearson correlation coefficient
def pearson_correlation(df_cols):
    df = pd.DataFrame(columns = ['feature_1', 'feature_2'])

    i = 0
    while (i < len(df_cols)):
        # Skips comparing i with j column
        j = i + 1
        while (j < len(df_cols)):
            res = {'feature_1':f"{df_cols[i]}", 'feature_2':f"{df_cols[j]}"}
            x = feature_df[df_cols[i]].to_numpy()
            y = feature_df[df_cols[j]].to_numpy()
            res['ps_corr'] = np.corrcoef(x,y)[0][1]
            df = df.append(res, ignore_index=True)
            j = j + 1

        i = i + 1
    
    return df

# Calculate Skewness
def skewness(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['skewness'] = df['features'].apply(lambda x: skew(np.asarray(feature_df[x].tolist())))
    return df

# Calculate Variance
def variance(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['variance'] = df['features'].apply(lambda x: np.var(np.asarray(feature_df[x].tolist())))
    return df

# Calculate Kurtosis
def ktosis(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['kurtosis'] = df['features'].apply(lambda x: kurtosis(feature_df[x].tolist()))
    return df

# Calculate Covariance
def covariance(df_cols):
    df = feature_df(['feature_1', 'feature_2'])

    i = 0
    while (i < len(df_cols)):
        j = i + 1
        while (j < len(df_cols)):
            res = {'feature_1':f"{df_cols[i]}", 'feature_2':f"{df_cols[j]}"}
            x = feature_df[df_cols[i]].to_numpy()
            y = feature_df[df_cols[j]].to_numpy()
            res['covariance'] = np.cov(x,y)[0][1]
                
            df = df.append(res, ignore_index=True)
            j = j + 1

        i = i + 1
    return df

# Return z-score of the values instead
def z_score(df_cols):
    df = feature_df[['TEXT','cEXT','cNEU','cAGR','cCON','cOPN']]
    for col in df_cols:
        df[f'{col}_z_score'] = stats.zscore(np.asarray(feature_df[col]))

    return df

# Analyse individual feature
def analyse(cols, name, func, reproc):
    if (reproc):
        result = func(cols)
        result.to_csv('./analysis_data/'+name+'.csv', sep=',', encoding='utf-8', index = False)
        analysis_df[name] = result
    else:
        result = pd.read_csv('./analysis_data/'+name+'.csv')
        analysis_df[name] = result

# Analyse all the feature, given columns and reprocess boolean
def feature_analysis(cols, reproc = False):
    analyse(cols, 'ps_corr', pearson_correlation, reproc)
    analyse(cols, 'skw', skewness, reproc)
    analyse(cols, 'vari', variance, reproc)
    analyse(cols, 'covari', covariance, reproc)
    analyse(cols, 'kurtosis', ktosis, reproc)

def main():

    # All extracted features
    reduced = feature_df.columns[6:36]
    z_score(reduced).to_csv(f'./z_score_{csv_file}', sep=',', encoding='utf-8')
    # feature_analysis(reduced, True)

    # ps_corr = analysis_df['ps_corr']
    # ps_corr_signf = ps_corr[abs(ps_corr['ps_corr']) > 0.5]
    # print(ps_corr_signf)

    # skw = analysis_df['skw']
    # skw_normal = skw[skw['skewness'] < 1]
    # skw_normal = skw_normal[skw_normal['skewness'] > -1]
    # print(skw_normal)

    # print(kurtosis(np.asarray(feature_df['num_of_char'])))

    print(" ====== DONE ====== ")

if __name__ == "__main__":
    main()