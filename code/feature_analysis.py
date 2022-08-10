# Require massive rewrites in terms of readability
# Variable names need to be reconsidered
from typing import List

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

# GLOBAL FEATURES
csv_file = sys.argv[1] # CSV file name
feature_df = pd.read_csv(csv_file, index_col = 0) # Feature dataframe of the csv
analysis_df = {} # Caching analysis data (REDUNDANT)

def pearson_correlation(df_cols:List[str]) -> pd.DataFrame:
    '''Calculate the Pearson Correlation for a given list of features

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the correlation for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the processed data
    '''
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

def skewness(df_cols:List[str]) -> pd.DataFrame:
    '''Calculate the Skewness for a given list of features

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the skewness for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the skewness data
    '''
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['skewness'] = df['features'].apply(lambda x: skew(np.asarray(feature_df[x].tolist())))
    return df

def variance(df_cols:List[str]) -> pd.DataFrame:
    '''Calculate the Variance for a given list of features

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the variance for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the variance data
    '''
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['variance'] = df['features'].apply(lambda x: np.var(np.asarray(feature_df[x].tolist())))
    return df

def ktosis(df_cols:List[str]) -> pd.DataFrame:
    '''Calculate the Kurtosis for a given list of features

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the kurtosis for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the kurtosis data
    '''
    df = pd.DataFrame(df_cols)
    df.columns=['features']
    df['kurtosis'] = df['features'].apply(lambda x: kurtosis(feature_df[x].tolist()))
    return df

def covariance(df_cols:List[str]) -> pd.DataFrame:
    '''Calculate the Covariance for a given list of features

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the covariance for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the covariance data
    '''
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

def z_score(df_cols:List[str]) -> pd.DataFrame:
    '''Standardist given set of features into their z_scores

    Parameters
    ----------
    df_cols : List[str]
        The list of features to find the z_score for for

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the data converted to z_scores
    '''
    df = feature_df[['TEXT','cEXT','cNEU','cAGR','cCON','cOPN']]
    for col in df_cols:
        df[col] = stats.zscore(np.asarray(feature_df[col]))

    return df

def t_test_feat(feature:str, no_df:pd.DataFrame, yes_df:pd.DataFrame) -> float:
    '''HELPER: Conduct a T-Test on an individual feature against their y column and n column

    Parameters
    ----------
    df_cols : List[str]
        The feature having its T-test calculated

    Returns
    ----------
    stats.ttest_ind(no_field_df,yes_field_df, equal_var=True)[0] : float
        The statistical significance of the t-test
    '''
    no_field_df = no_df[feature].tolist()
    yes_field_df = yes_df[feature].tolist()
    print(stats.ttest_ind(no_field_df,yes_field_df, equal_var=True))
    return stats.ttest_ind(no_field_df,yes_field_df, equal_var=True)[1]

def t_test(df_cols:List[str]) -> pd.DataFrame:
    '''Conduct a T-Test on a list of features against their y column and n column

    Parameters
    ----------
    df_cols : List[str]
        The list of features to conduct their t-test against

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the t-test result for each given feature
    '''
    df = pd.DataFrame(df_cols)
    df.columns=['features']

    # Filter columns into Y and N, based on personality
    for trait in ['cEXT','cNEU','cAGR','cCON','cOPN']:
        no_df = feature_df.loc[feature_df[trait] == 'n']
        yes_df = feature_df.loc[feature_df[trait] == 'y']

        # For each feature
        df[f'{trait}_t_stat'] = df['features'].apply(lambda x: t_test_feat(x,no_df,yes_df))

    return df

def t_test_comparer(significance:float) -> pd.DataFrame:
    '''Filter t-test based on a significance, return significant features

    Parameters
    ----------
    significance : float
        The significance to be filtered against. Anything greater will be filtered out

    Returns
    ----------
    df : pd.DataFrame
        A dataframe with all the personality traits and each of their most significant features
    '''
    df = pd.read_csv('./analysis_data/feature_t_test_p_val.csv')
    res_df = pd.DataFrame(columns = ['traits', 'signf_features'])
    for trait in df.columns[1:6]:
        reduced_df = df[['features',trait]].copy()
        filtered_df = reduced_df[reduced_df[trait] <= significance]
        res = {'traits':f"{trait}", 'signf_features':f"{','.join(filtered_df['features'].tolist())}"}
        res_df = res_df.append(res, ignore_index=True)

    return res_df

def analyse(cols:List[str], name:str, func, reproc:bool) -> None:
    '''Wrapper for feature analysis functions

    Parameters
    ----------
    cols : List[str]
        The features which are to be processed
    name : str
        The name of the csv file to place the process data in
    func : function
        The analyses function to be ran
    reproc : bool
        Reprocess flag to indicate if the analyse should be recalculated
    '''
    if (reproc):
        result = func(cols)
        result.to_csv('./analysis_data/'+name+'.csv', sep=',', encoding='utf-8', index = False)
        analysis_df[name] = result
    else:
        result = pd.read_csv('./analysis_data/'+name+'.csv')
        analysis_df[name] = result

# Analyse all the feature, given columns and reprocess boolean
def feature_analysis(cols:List[str], reproc:bool = False):
    '''Wrapper for calling all feature analysis functions

    Parameters
    ----------
    cols : List[str]
        The features which are to be processed
    reproc : bool
        Reprocess flag to indicate if the analyse should be recalculated
        Default to False
    '''
    analyse(cols, 'ps_corr', pearson_correlation, reproc)
    analyse(cols, 'skw', skewness, reproc)
    analyse(cols, 'vari', variance, reproc)
    analyse(cols, 'covari', covariance, reproc)
    analyse(cols, 'kurtosis', ktosis, reproc)

def main():

    # ====== All extracted features ======
    reduced = feature_df.columns[6:36]

    # ====== Feature Analysis ======
    
    # t_test(reduced).to_csv(f'./analysis_data/feature_t_test_stat.csv', sep=',', encoding='utf-8', index=False)
    # t_test(reduced).to_csv(f'./analysis_data/feature_t_test_p_val.csv', sep=',', encoding='utf-8', index=False)
    t_test_comparer(0.05).to_csv('./analysis_data/significant_features.csv', sep=',', encoding='utf-8', index = False)
    # feature_analysis(reduced, False)

    # z_score(reduced).to_csv(f'./z_score_{csv_file}', sep=',', encoding='utf-8')

    print(" ====== DONE ====== ")

if __name__ == "__main__":
    main()