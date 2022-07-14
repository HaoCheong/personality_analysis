import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map(sns.scatterplot)

# tips = sns.load_dataset("tips")
# sns.relplot(x="total_bill", y="tip", data=tips)

plt.savefig('save_as_a_png.png')