import utils as ut
import numpy as np

def main():
    #Load data
    X, y, attribute_names, class_dict = ut.load_xls(file_name='data/abalone.xls')

    #ut.pca_analysis(X, y)
    #print(X,y,attribute_names,class_dict)

    # Original table
    full_table = np.concatenate((X,y), axis=1)
    print(full_table)

    # Standardized table
    y_std = y - y.mean(axis=0)
    y_std = y_std / y_std.std(axis=0)
    full_std_table = np.concatenate((X,y_std), axis=1)
    print(full_std_table)

if __name__ == '__main__':
    main()