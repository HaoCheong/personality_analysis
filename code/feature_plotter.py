import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

feature_df = pd.read_csv('final_v2.csv', index_col = 0)


def basic_plot(field):
    sns.displot(feature_df[field])


def two_col_pairplot(field1, field2):
    sns.relplot(x=field1, y=field2, data=feature_df)

def complete_pairplot():
    reduced = feature_df.columns[6:36].tolist()
    # g = sns.PairGrid(reduced)
    sns.pairplot(feature_df[reduced], markers=["o", "s"])

def main():
    # basic_plot('num_of_char')
    # two_col_pairplot('num_of_char','num_any_words')
    complete_pairplot()
    plt.savefig('plot.png')
    # plt.show()

if __name__ == "__main__":
    main()