import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    variable_names = doc.row_values(rowx=0, start_colx=0, end_colx=doc.ncols)
    class_labels = doc.col_values(0, 1, doc.nrows)
    class_names = sorted(set(class_labels))
    class_dict = dict(zip(class_names, range(len(class_names))))

    x0 = np.array([np.array([class_dict[value] for value in class_labels])])
    x0 = x0.T
    X = np.array([doc.row_values(i, 1, doc.ncols - 1) for i in range(1, doc.nrows)])
    y = np.array([doc.row_values(i, doc.ncols - 1, doc.ncols) for i in range(1, doc.nrows)])  # Rings only

    return x0, X, y, variable_names, class_names
