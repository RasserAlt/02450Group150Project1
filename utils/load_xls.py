import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    attribute_names = doc.row_values(rowx=0, start_colx=0, end_colx=doc.ncols - 1)
    class_labels = doc.col_values(0, 1, doc.nrows)
    class_names = sorted(set(class_labels))
    class_dict = dict(zip(class_names, range(len(class_names))))

    y = np.array([class_dict[value] for value in class_labels])

    X = np.empty((doc.nrows-1, doc.ncols-1))
    for i in range(doc.ncols-1):
        X[:, i] = np.array(doc.col_values(i, 1, doc.nrows)).T

    return X, y, attribute_names, class_dict