import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    attribute_names = doc.row_values(rowx=0, start_colx=0, end_colx=doc.ncols)
    category_labels = doc.col_values(0, 1, doc.nrows)
    category_names = sorted(set(category_labels))
    category_dict = dict(zip(category_names, range(len(category_names))))

    x0 = np.array([np.array([category_dict[value] for value in category_labels])])
    x0 = x0.T
    X = np.array([doc.row_values(i, 1, doc.ncols - 1) for i in range(1, doc.nrows)])
    y = np.array([doc.row_values(i, doc.ncols - 1, doc.ncols) for i in range(1, doc.nrows)])  # Rings only

    return x0, X, y, attribute_names, category_names
