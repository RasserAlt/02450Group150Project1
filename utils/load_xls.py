import numpy as np
import xlrd


def load_xls(file_name):
    doc = xlrd.open_workbook(file_name).sheet_by_index(0)

    attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=doc.ncols - 1)

    classLabels = doc.col_values()