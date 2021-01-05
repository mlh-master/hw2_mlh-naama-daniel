# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import random

random.seed(10)

def nancount(T1D_dataset):
    """
    :param T1D_dataset: Pandas series of CTG features
    :return: A dictionary of number of nan's in each feature in T1D_dataset
    """
    T1D_dataset_temp = T1D_dataset.copy()
    c_T1D = {}  # initialize a dictionary
    for column in T1D_dataset_temp.columns:
        c_T1D[column] = len(T1D_dataset) - len(T1D_dataset_temp[column].dropna())
    return c_T1D


def nan2samp(T1D_dataset):
    """
    :param T1D_dataset: Pandas series of T1D_dataset
    :return: A pandas dataframe of the dictionary T1D_d containing the "clean" features
    """
    np.random.seed(10)
    T1D_d = {}
    T1D_dataset_temp = T1D_dataset.copy()
    for column in T1D_dataset_temp.columns:
        col = T1D_dataset_temp[column].copy()
        i = col.isnull()  # create a boolean vector with true values where col has nan values
        idx = np.zeros(np.sum(i))
        t = 0  # initialize a counter for the nan locations vector (idx)
        for j in range(1, len(i)):
            if i[j] == 1:
                idx[t] = j
                t += 1
        temp = np.random.choice(col.dropna(), size=len(idx))  # random sampling of len(idx) values from col
        col[idx] = temp
        T1D_d[column] = col
    return pd.DataFrame(T1D_d)


def dist_table(X_train,X_test):
    """
    :param x_train: train df of T1D features
    :param x_test: test df of T1D features
    :return: a table of the positive rates for every feature in the train/test groups
    """
    x_train = X_train.copy()
    x_test = X_test.copy()
    x_train.drop(columns=['Age'], inplace=True)
    x_test.drop(columns=['Age'], inplace=True)
    d_table = {}  # initialize a dictionary
    for column in x_train.columns:
        curr_train = x_train[column]
        curr_test = x_test[column]
        train_prc = 100*(curr_train.sum()/curr_train.size)
        test_prc = 100*(curr_test.sum()/curr_test.size)
        d_table[column] = {"Train %": train_prc, "Test %": test_prc, "Delta %": train_prc - test_prc}
    return pd.DataFrame(d_table)


def feat_lab_cor(T1D_dataset):
    """
    :param T1D_dataset: test df of T1D features
    """
    fig, axes = plt.subplots(3, 6, figsize=(16, 10))
    sns.set_context("paper", font_scale=0.8)
    fig.suptitle('Relationships between features and labels', fontsize=14)
    i = 0
    for column in T1D_dataset:
        if column == 'Gender':
            feat_diag = sns.countplot(ax=axes[0, 0], x='Gender', hue='Diagnosis', data=T1D_dataset)
            feat_diag.set(xticklabels=['Male', 'Female'])
            feat_diag.set(xlabel='', ylabel='')
            feat_diag.set_title('Gender', fontsize=13)
        elif column == 'Age':
            feat_diag = sns.countplot(ax=axes[0, 1], x='Age', hue='Diagnosis', data=T1D_dataset)
            feat_diag.xaxis.set_major_locator(ticker.LinearLocator(10))
            feat_diag.set(xlabel='', ylabel='')
            feat_diag.set_title('Age', fontsize=13)
        else:
            if i < 18:
                feat_diag = sns.countplot(ax=axes[i//6, i % 6], x=column, hue='Diagnosis', data=T1D_dataset)
                #feat_diag.set(xticklabels=['No', 'Yes'])
                feat_diag.set(xlabel='',  ylabel='')
                feat_diag.set_title(column, fontsize=13)
        i += 1
    plt.show()
    return()

def pred_log(logreg, X_train, y_train, X_test, flag=False):
    """
    :param logreg: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels
    :param X_test: Testing set samples
    :param flag: A boolean determining whether to return the predicted he probabilities of the classes
    :return: A two elements tuple containing the predictions and the weightning matrix
    """

    logreg.fit(X_train, y_train)
    w_log = logreg.coef_
    if flag == False:
        y_pred_log = logreg.predict(X_test)
    if flag == True:
        y_pred_log = logreg.predict_proba(X_test)

    return y_pred_log, w_log


def kcfold(X, y, C, penalty, K):
    """
    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :return: A dictionary of scores
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for p in penalty:
        for c in C:
            log_reg_func = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr', random_state=10)
            #loss_val_vec = np.zeros(K)
            roc_auc_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                y_pred_val, w_pred_val = pred_log(log_reg_func, x_train, y_train, x_val, flag=True)
                #loss_val_vec[k] = log_loss(y_val, y_pred_val)
                roc_auc_vec[k] = roc_auc_score(y_val, y_pred_val[:, 1])
                k += 1
            validation_dict.append({"C": c, "penalty": p, "roc_auc": roc_auc_vec.mean()})
    return validation_dict


def plt_2d_pca(X_pca, y):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='b')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
    ax.legend(('positive','negative'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')
    plt.show()
