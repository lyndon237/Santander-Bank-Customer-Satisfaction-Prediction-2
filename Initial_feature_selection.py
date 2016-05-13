# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:02:41 2016

@author: Lyndon Quadros
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#Initial data loading and pre-processing to reduce the Number of Features.
import numpy as np
import itertools


#import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA



#def principal_component_analysis(x_train):
#
#    """
#    Principal Component Analysis (PCA) identifies the combination
#    of attributes (principal components, or directions in the feature space)
#    that account for the most variance in the data.
#
#    Let's calculate the 2 first principal components of the training data,
#    and then create a scatter plot visualizing the training data examples
#    projected on the calculated components.
#    """
#
#    # Extract the variable to be predicted
#    y_train = x_train["TARGET"]
#    x_train = x_train.drop(labels="TARGET", axis=1)
#    classes = np.sort(np.unique(y_train))
#    labels = ["Satisfied customer", "Unsatisfied customer"]
#
#    # Normalize each feature to unit norm (vector length)
#    x_train_normalized = normalize(x_train, axis=0)
#    
#    # Run PCA
#    pca = PCA(n_components=2)
#    x_train_projected = pca.fit_transform(x_train_normalized)


def remove_feat_constants(data_frame):
    # Remove feature vectors containing one unique value,
    # because such features do not have predictive value.
    print("")
    print("Deleting zero variance features...")
    # Let's get the zero variance features by fitting VarianceThreshold
    # selector to the data, but let's not transform the data with
    # the selector because it will also transform our Pandas data frame into
    # NumPy array and we would like to keep the Pandas data frame. Therefore,
    # let's delete the zero variance features manually.
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold()
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_ix_delete


def remove_feat_identicals(data_frame):
    # Find feature vectors having the same values in the same order and
    # remove all but one of those redundant features.
    print("")
    print("Deletinfg identical features...")
    n_features_originally = data_frame.shape[1]
    # Find the names of identical features by going through all the
    # combinations of features (each pair is compared only once).
    feat_names_delete = []
    for feat_1, feat_2 in itertools.combinations(
            iterable=data_frame.columns, r=2):
        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):
            feat_names_delete.append(feat_2)
    feat_names_delete = np.unique(feat_names_delete)
    # Delete the identical features
    data_frame = data_frame.drop(labels=feat_names_delete, axis=1)
    n_features_deleted = len(feat_names_delete)
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_names_delete

def final_feats(df_data):
    x_train = df_data.iloc[:,1:370] #removing the "ID" and the "Target" columns

    

    """Getting the first 2 PCs""" 
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(x_train, axis=0))
   
    x_train, del_constants = remove_feat_constants(x_train) 
    """ removing columns with no 
    variance; in our case the all-zero columns"""
    x_train, del_identicals = remove_feat_identicals(x_train)
    """removing columns that are identical to each other, and retainining
    only one of them"""
    y_train = df_data["TARGET"]


# Using L1 based feature selection on X_train with 308 columns
    lsvc = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    feat_ix_keep = model.get_support(indices=True) #getting indices of selected features
#so that I don't have to use "transform" and convert the data frame to a matrix.
    orig_feat_ix = np.arange(x_train.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)

    X_train_new = x_train.drop(labels=x_train.columns[feat_ix_delete],
                                 axis=1)
    X_train_new.insert(1, 'PCAOne', x_train_projected[:, 0])
    X_train_new.insert(1, 'PCATwo', x_train_projected[:, 1])
    return X_train_new, y_train, feat_ix_keep, pca, del_constants, del_identicals                             
