import pandas as pd

df = pd.read_csv('essays.csv',encoding='cp1252')
data_top = data.head() 
print(df.to_string())