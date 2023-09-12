# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_excel('../data/abalone.xls')
df = df.drop('Sex', axis=1)
    
basic_summary_statistics = df.describe()
print(basic_summary_statistics)
    
variance = df.var()
print(variance)

dfs = df.apply(lambda x: (x - x.mean()) / x.std())
cov_matrix = dfs.cov()
print(cov_matrix)
