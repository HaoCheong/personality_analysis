import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_theme(style="darkgrid")

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
def complete_pairplot():
    reduced = feature_df.columns[6:36].tolist()
    sns.pairplot(feature_df[reduced], markers=["o", "s"])

def main():
    complete_pairplot()
    plt.savefig('plot.png')

if __name__ == "__main__":
    main()