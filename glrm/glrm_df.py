from glrm.glrm import GLRM 
from numpy import sqrt, repeat, tile, hstack, array, zeros, ones, sqrt, diag, asarray, hstack, vstack, split, cumsum
from numpy.random import randn
from copy import copy
from numpy.linalg import svd, norm
import cvxpy as cp
import numpy as np
import warnings

class GLRM_DF(object):
    """
    wrapper class for GLRM to handle dataframes easily
    input: df, num_columns, loss, regX, regY, k, missing_list = None, converge = None, scale=True, verbose=True
    
    output: low rank approximation of the input dataframe (df of X), 
            reversal matrix Y (XY~df_original), 
            convergence information
    """
    def __init__(self, df, num_columns, loss_dict:dict, regX, regY, k, missing_list = None, converge = None, scale=True, pre_onehot=True, verbose=True):
        self.df = df
        self.num_columns = num_columns
        self.loss = [loss_dict['numerical'], loss_dict['categorical']] # {'numerical': loss, 'categorical': loss}
        self.regX = regX
        self.regY = regY
        self.k = k
        self.missing_list = missing_list
        self.converge = converge
        self.scale = scale
        self.verbose = verbose
        self.pre_onehot = pre_onehot # if True, non-numerical columns are already one-hot encoded
        self.glrm_fit = None

    def fit(self):
        """
        fit the GLRM model
        """
        if not self.pre_onehot:
            warnings.warn("Non-numerical columns are not one-hot encoded. Please ensure they are one-hot encoded before fitting the model.")
            
        df_num, df_cat = self.seperate_num_cat()
        np_num, np_cat = self.df_to_np(df_num), self.df_to_np(df_cat)

        np_cat = self.onehot_to_bool(np_cat) # convert one-hot encoded (0, 1) columns to boolean (-1, 1) columns to satisfy glrm input
        A = [np_num, np_cat]

        glrm = GLRM(A, self.loss, self.regX, self.regY, self.k, self.missing_list, self.converge, self.scale)
        self.glrm_fit = glrm.fit()
        self.X, self.Y = self.glrm_fit.factors()

        result = self.eval_model(A)

        return self.X, self.Y, result
    
    def eval_model(self, A):
        """
        evaluate the model
        """
        convergence = self.glrm_fit.convergence()
        A_hat = self.glrm_fit.predict()
        reconstruction_error, relative_error, information_captured = self.glrm_fit.eval_reconstruction(A, A_hat)

        result = {
            "convergence": convergence,
            "reconstruction_error": reconstruction_error,
            "relative_error": relative_error,
            "information_captured": information_captured
        }

        if self.verbose:
            print("Convergence: ", convergence)
            print("Reconstruction error: ", reconstruction_error)
            print("Relative error: ", relative_error)
            print("Information captured: ", information_captured)

        return result


    def seperate_num_cat(self):
        """
        seperate numerical and categorical columns
        """
        # df_num = self.df.drop(self.cat_columns, axis=1)
        # df_cat = self.df[self.cat_columns]

        df_cat = self.df.drop(self.num_columns, axis=1)
        df_num = self.df[self.num_columns]

        return df_num, df_cat
    
    def onehot_to_bool(self, np_cat):
        """
        convert one-hot encoded (0, 1) columns to booloean (-1, 1) columns
        """
        # return df_cat.applymap(lambda x: 1 if x == 1 else -1)
        return np.where(np_cat == 1, 1, -1)


    def df_to_np(self, df):
        """
        convert the dataframe to numpy arrays
        """
        return df.to_numpy()