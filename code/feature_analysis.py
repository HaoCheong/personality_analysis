# Require massive rewrites in terms of readability
# Variable names need to be reconsidered

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
    df = pd.DataFrame(columns = ['feature_1', 'feature_2'])

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
        df[col] = stats.zscore(np.asarray(feature_df[col]))

    return df

#Individual Feature T-test (Helper for t_test()) 
def t_test_feat(feature, no_df, yes_df):
    no_field_df = no_df[feature].tolist()
    yes_field_df = yes_df[feature].tolist()
    return stats.ttest_ind(no_field_df,yes_field_df, equal_var=True)[0]        

# T-test independent analysis of every feature
def t_test(df_cols):
    df = pd.DataFrame(df_cols)
    df.columns=['features']

    # Filter columns into Y and N, based on personality
    for trait in ['cEXT','cNEU','cAGR','cCON','cOPN']:
        no_df = feature_df.loc[feature_df[trait] == 'n']
        yes_df = feature_df.loc[feature_df[trait] == 'y']

        # For each feature
        df[f'{trait}_t_stat'] = df['features'].apply(lambda x: t_test_feat(x,no_df,yes_df))

    return df

# Filter results of a t-test and returns significant list
def t_test_comparer(significance):

    # For all personalitty trait features, find the features which have significance
    df = pd.read_csv('./analysis_data/feature_t_test_p_val.csv')
    res_df = pd.DataFrame(columns = ['traits', 'signf_features'])
    for trait in df.columns[1:6]:
        reduced_df = df[['features',trait]].copy()
        filtered_df = reduced_df[reduced_df[trait] <= significance]
        res = {'traits':f"{trait}", 'signf_features':f"{filtered_df['features'].tolist()}"}
        res_df = res_df.append(res, ignore_index=True)

    return res_df

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

    # ====== All extracted features ======
    # reduced = feature_df.columns[6:36]

    # # Feature Analysis
    # t_test(reduced).to_csv(f'./analysis_data/feature_t_test_stat.csv', sep=',', encoding='utf-8', index=False)
    t_test_comparer(0.05).to_csv('./analysis_data/significant_features.csv', sep=',', encoding='utf-8', index = False)
    # feature_analysis(reduced, False)

    # z_score(reduced).to_csv(f'./z_score_{csv_file}', sep=',', encoding='utf-8')

    

    print(" ====== DONE ====== ")

if __name__ == "__main__":
    main()