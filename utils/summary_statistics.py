# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../data/abalone.xls')
df = df.drop('Sex', axis=1)

basic_summary_statistics = df.describe()
print(basic_summary_statistics)

variance = df.var()
print(variance)

dfs = df.apply(lambda x: (x - x.mean()) / x.std())
cov_matrix = dfs.cov()
print(cov_matrix)

for column in df:
# Create a box plot for each column
    plt.title(f'Box plot of {column}')
    df[column].plot(kind='box', figsize=(16, 8))
    plt.figure(dpi=300)
    plt.show()


