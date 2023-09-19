import utils as ut
import numpy as np

def main(file_name):
    #Load data
    #x0 is the first row of the data containing the decrete catagory attribute, here only Sex
    #y is the goal attribute we're intterested in finding, here Rings
    #X are the remanding continues attributes
    x0, X, y, attribute_names, catagory_dict = ut.load_xls(file_name)

    # Summary Statistics
    ut.summary_statistics(file_name)

    # table of continues all attributes
    Xy = np.concatenate((X,y), axis=1)

    # Standardized table
    Xy_std = Xy - Xy.mean(axis=0)
    Xy_std = Xy_std / Xy_std.std(axis=0)

    #PCA Analysis
    ut.pca_analysis(Xy_std)



if __name__ == '__main__':
    main('data/abalone.xls')