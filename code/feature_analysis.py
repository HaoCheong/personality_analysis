from re import I

import numpy as np
import sys

import pandas as pd
import warnings

from scipy.stats import skew

warnings.simplefilter(action='ignore', category=FutureWarning)

if len(sys.argv) != 2:
    print('USAGE: python3 feature_plotter.py <csv_file>')
    exit()

csv_file = sys.argv[1]
feature_df = pd.read_csv(csv_file, index_col = 0)


def pearson_correlation(df_cols):
    df = pd.DataFrame(columns = ['feature_pair'])

    i = 0
    while (i < len(df_cols)):
        j = i + 1
        while (j < len(df_cols)):
            res = {'feature_pair':f"{df_cols[i]}_{df_cols[j]}"}
            x = feature_df[df_cols[i]].to_numpy()
            y = feature_df[df_cols[j]].to_numpy()
            res['pearson_correlation'] = np.corrcoef(x,y)[0][1]
                
            df = df.append(res, ignore_index=True)
            j = j + 1

        i = i + 1
    
    return df

# Calculating Skewness
def skewness(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    # print(df.head(10))
    df['skewness'] = df['features'].apply(lambda x: skew(np.asarray(feature_df[x].tolist())))
    
    return df

def variance(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['variance'] = df['features'].apply(lambda x: np.var(np.asarray(feature_df[x].tolist())))
    return df

def covariance(df_cols):
    df = pd.DataFrame(columns = ['feature_pair'])

    i = 0
    while (i < len(df_cols)):
        j = i + 1
        while (j < len(df_cols)):
            res = {'feature_pair':f"{df_cols[i]}_{df_cols[j]}"}
            x = feature_df[df_cols[i]].to_numpy()
            y = feature_df[df_cols[j]].to_numpy()
            res['covariance'] = np.cov(x,y)[0][1]
                
            df = df.append(res, ignore_index=True)
            j = j + 1

        i = i + 1
    
    return df
    
def main():
    reduced = feature_df.columns[6:36]
    ps_corr = pearson_correlation(reduced)
    skw = skewness(reduced)
    vari = variance(reduced)
    covar = covariance(reduced)
    covar.to_csv('covar.csv', sep=',', encoding='utf-8', index = False)


if __name__ == "__main__":
    main()