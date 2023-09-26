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

    # Group the DataFrame by "Sex" column
    # df2 = pd.read_excel(file_name)
    # grouped_df = df2.groupby("Sex")
    #
    # # Calculate the covariance matrix for each group
    # cov_matrix_m = grouped_df.get_group("M").cov()
    # cov_matrix_f = grouped_df.get_group("F").cov()
    # cov_matrix_i = grouped_df.get_group("I").cov()
    #
    # print("Covariance matrix for M:")
    # print(cov_matrix_m)
    #
    # print("Covariance matrix for F:")
    # print(cov_matrix_f)
    #
    # print("Covariance matrix for I:")
    # print(cov_matrix_i)

    return None