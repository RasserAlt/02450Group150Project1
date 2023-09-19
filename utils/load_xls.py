import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    attribute_names = doc.row_values(rowx=0, start_colx=0, end_colx=doc.ncols)
    catagory_labels = doc.col_values(0, 1, doc.nrows)
    catagory_names = sorted(set(catagory_labels))
    catagory_dict = dict(zip(catagory_names, range(len(catagory_names))))

    x0 = np.array([np.array([catagory_dict[value] for value in catagory_labels])])
    x0 = x0.T
    X = np.array([doc.row_values(i, 1, doc.ncols - 1) for i in range(1, doc.nrows)])
    y = np.array([doc.row_values(i, doc.ncols - 1, doc.ncols) for i in range(1, doc.nrows)]) # Rings only


    return x0,X,y, attribute_names, catagory_dict
