# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt




def summary_statistics(file_name):
    df = pd.read_excel(file_name)
    df = df.drop('Sex', axis=1)

    basic_summary_statistics = df.describe()
    print(basic_summary_statistics)

    variance = df.var()
    print(variance)

    dfs = df.apply(lambda x: (x - x.mean()) / x.std())
    cov_matrix = dfs.cov()
    print(cov_matrix)

    return None