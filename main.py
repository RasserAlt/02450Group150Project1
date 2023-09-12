import utils as ut
import numpy as np

def main():
    #Load data
    X, y, attribute_names, class_dict = ut.load_xls(file_name='data/abalone.xls')

    #ut.pca_analysis(X, y)
    #print(X,y,attribute_names,class_dict)

    y_reshaped = np.reshape(y, (1, y.size)) # transpose from vertical to horizontal
    full_table = np.vstack((X, y_reshaped))
    print(full_table)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
