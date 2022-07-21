from re import I
import numpy as np
import sys

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if len(sys.argv) != 2:
    print('USAGE: python3 feature_plotter.py <csv_file>')
    exit()

csv_file = sys.argv[1]
feature_df = pd.read_csv(csv_file, index_col = 0)
analysis_pipeline = []
column_names = []
index = []

# Carry out calculation off all in the analysis pipeline
def comparison_calc(df_col):

    df = pd.DataFrame(columns = ['feature_pair'] + column_names)

    i = 0
    while (i < len(df_col)):
        j = i + 1
        while (j < len(df_col)):
            res = {'feature_pair':f"{df_col[i]}_{df_col[j]}"}
            for analysis in analysis_pipeline:
                res[analysis[0]] = analysis[1](df_col[i],df_col[j])
                
            df = df.append(res, ignore_index=True)
            j = j + 1

        i = i + 1
    
    return df

# Add analysis to the pipeline
def add_analysis(name, func):
    analysis_tuple = (name,func)
    analysis_pipeline.append(analysis_tuple)
    column_names.append(name)

# Get all the feature to process
def feature_analysis():
    add_analysis("pearson_corrcoef", pearson_correlation)

# Calculating Pearson Correlation
def pearson_correlation(col_name_1, col_name_2):
    x = feature_df[col_name_1].to_numpy()
    y = feature_df[col_name_2].to_numpy()
    return np.corrcoef(x,y)[0][1]
    
def main():
    reduced = feature_df.columns[6:36]
    feature_analysis()
    final_df = comparison_calc(reduced)
    final_df.to_csv('final_analysis.csv', sep=',', encoding='utf-8', index = False)
    print(comparison_calc(reduced).head(10))

if __name__ == "__main__":
    main()