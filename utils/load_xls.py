import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    variable_names = doc.row_values(rowx=0, start_colx=1, end_colx=doc.ncols - 1)
    class_labels = doc.col_values(0, 1, doc.nrows)
    class_names = sorted(set(class_labels))

    # one_out_of K
    x0 = np.array([np.array([1 if class_names[0] == value else 0 for value in class_labels]),
                   np.array([1 if class_names[1] == value else 0 for value in class_labels]),
                   np.array([1 if class_names[2] == value else 0 for value in class_labels])]).T

    X = np.array([doc.row_values(i, 1, doc.ncols - 1) for i in range(1, doc.nrows)])
    y = np.array([doc.row_values(i, doc.ncols - 1, doc.ncols) for i in range(1, doc.nrows)])  # Rings only

    return x0, X, y, variable_names, class_names
