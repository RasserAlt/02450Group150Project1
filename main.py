import utils as ut
import numpy as np

def main():
    #Load data
    X, y, attribute_names, class_dict = ut.load_xls(file_name='data/abalone.xls')

    #ut.pca_analysis(X, y)
    #print(X,y,attribute_names,class_dict)

    full_table = np.concatenate((X,y), axis=1)
    print(full_table)
    
    #
if __name__ == '__main__':
    main()