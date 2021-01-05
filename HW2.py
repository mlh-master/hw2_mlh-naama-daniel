#!/usr/bin/env python
# coding: utf-8

# # BM 336546 - HW2: Type 1 Diabetes (T1D)
# Daniel sapir & Naama Rivlin
# 
# ### **Assignment Goal:**
# To predict if a patient has T1D, based on binary data from a simple yes/no questionnaire about the patient's medical history, using ML algorithms.

# ## **Theory Questions**:
# 
# **Q1:** To evaluate how well our model performs at T1D classification, we need to have evaluation
# metrics that measures of its performances/accuracy. Which evaluation metric is more
# important to us: model accuracy or model performance? Give a simple example that
# illustrates your claim.
# 
# **Q1 Answer:** Accurecy is the fraction of predictions our model got right. Sometimes, accuracy can be misleading, for example when we have very imbalanced data. classification algorithm with an accuracy of 90% could be considered a highly accurate algorithm, but if the data is with a ratio of 90%-10%, even a naive classifier would achieve this accuracy. Therefore, in many cases other evaluation metrics such as F1 score and AUROC are more important.  
# 
# **Q2:** T1D is often associated with other comorbidities such as a heart attack. You are asked to
# design a ML algorithm to predict which patients are going to suffer a heart attack. Relevant
# patient features for the algorithm may include blood pressure (BP), body-mass index (BMI),
# age (A), level of physical activity (P), and income (I). You should choose between two
# classifiers: the first uses only BP and BMI features and the other one uses all of the features
# available to you. Explain the pros and cons of each choice.
# 
# **Q2 Answer:** \
# Using only the **BP and BMI features**:
# * Advantage: The required computing power is smaller and it is easier to do data exploration and to clean the data
# * Disadvantage: We are not guaranteed that these are the two most important features for the purpose of classification (unless it has been tested before in some studies or algorithms), and even if they are the two most important features, throwing away the other features results in information loss so in some cases there will be some harm, even if small, in the quality of the classification
# 
# Using **all the features** available:
# * Advantage: The model will perform better because it has more information, assuming all the data is clean and reliable
# * Disadvantage: The required computing power is large, particularly when there are a lot of parameters to tune and when the training set is large.
# 
# In conclusion, using all the features allows for better performance but is more computationally expensive, whereas using only two features there is a risk of lower performance but less computational power is required.
# 
# **Q3:** A histologist wants to use machine learning to tell the difference between pancreas biopsies
# that show signs of T1D and those that do not. She has already come up with dozens of
# measurements to take, such as color, size, uniformity and cell-count, but she isn’t sure which
# model to use. The biopsies are really similar, and it is difficult to distinguish them from the
# human eye, or by just looking at the features. Which of the following is better: logistic
# regression, linear SVM or nonlinear SVM? Explain your answer.
# 
# **Q3 Answer:** Since the biopsies are very similar and it difficult to distinguish them from looking at the features, it is likely that the data is not linearly seperable. For this reason, logistic regression and linear SVM will probably not be good enough, and it will be better for her to use non-linear SVM.
# 
# **Q4:** What are the differences between LR and linear SVM and what is the difference in the
# effect/concept of their hyper-parameters tuning?
# 
# **Q4 Answer:** Generally, LR determines a probability that is a logistic function of a linear combination of the predictors, while SVM aims to fit a hyperplane that separates two classes of data based on the data points at the edge of each class. In other words, LR fits the data points as if they are along a continuous function, while SVM fits the data points assuming there are two classes that can be geometrically seperated.
# In SVM, a line is considered better than another line if it's margin is larger, meaning it is farther from both classes. In LR, a line is better than another line if the the distribution defined by it is low at points that belong to class −1 and high at points that belong to class +1  on average (compared to the distribution defined by another line).
# Accordingly, SVM only considers points near the margin (support vectors) while LR considers all the points in the data set.
# When there is a separating hyperplane, non-regularized logistic regression is not as good as linear SVM, because the maximum likelihood is achieved by any separating plane, and there is no guarantee we will get the best one. We might get poor predictive power near the margin.
# 
# Reagarding model hyperparameters tuning- 
# In both LR and linear SVM we tune the Hyper-parameters **C** and **penalty**. C adjusts how large the penalty will be. For the penalty we can choose batween ridge regression (uses $ L_2 $ norm) and lasso regression (uses $ L_1 $ norm). Ridge regression squares the coefficients in the penalty term, and therefore coefficients on less useful features will be *close* to zero. Lasso regression, on the other hand, sends some coefficients all the way down to zero. 
# 
# The purpose of the penalty is different in LR and in linear SVM:
# In linear SVM the hyper-parameters are used when we want to train a "soft margin" SVM which allows some misclassification, meaning being on the "wrong side" of decision boundary (sometimes a decision boundary cannot be found with a standard linear SVM). The optimization problem's goals are to increase the distance of decision boundary to classes, while maximizing the number of points that are correctly classified. The trade-off of the 2 goals is controlled by **C**, which adds a **penalty** for each misclassified data point. **The smaller C is, the smaller the penalty is, so we get a large margin at the expense of more misclassifications, and vice versa. Penalty is proportional to the distance from the data points to decision boundary.**
# for non-linear SVM, there is also **gamma hyper-parameter** which is used to prevent an overfit model, but for this question we will only discuss linear SVM.\
# In LR,
# 
# 

# ## **Coding Assignment:**
# 
# There are 565 patients in the database. Some features have missing values. We will start by loading the data from the file HW2_data.csv and do pre-processing.
# 
# ### **Part 1 - Data loading and pre-processing**

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import seaborn as sns

file = Path.cwd().joinpath('HW2_data.csv')
T1D_dataset = pd.read_csv(file)

random.seed(10)


# We know that the nurse who collected the data said that not all patients answered all the questions. So in some features there are empty values, and we have to decide what to do with them. One option is to through features that have missing values. Another option is delete rows that have missing values. A third option is to fill in the missing cells with random samples from the features. To choose the right option for us, we will first check how many features have empty cells and how many empty cells there are. Than we will check in how many rows there are empty cells, and weather there is a balance between the number of patients with empty values whose diagnosis is "Positive" and those diagnosed as "Negative".

# In[2]:


from functions import nancount as ncount
nan_count = ncount(T1D_dataset)
print("The number of nan's in each feature:")
nan_count


# We can observe that the largest number of nan's is in the feature 'Increased Thirst' and is equal to 20 (out of 565). This is a relatively small number, so we don't have to through the feature. In most of the features there are no nan's at all.

# In[3]:


feat = 'Diagnosis_Positive'
T1D_dummy = pd.get_dummies(T1D_dataset, dummy_na=False, drop_first=True)  # turn all the columns to 0/1 (except 'Age')
T1D_diag = pd.DataFrame(T1D_dummy[feat])

T1D_no_missing = T1D_dataset.copy().dropna()
T1D_no_missing_dummy = pd.get_dummies(T1D_no_missing, dummy_na=False, drop_first=True)
T1D_diag_no_missing = pd.DataFrame(T1D_no_missing_dummy[feat])

diag_cnt = T1D_dataset['Diagnosis'].value_counts().to_dict()
diag_cnt_no_missing = T1D_no_missing['Diagnosis'].value_counts().to_dict()
print("Before dropping rows with missing values we had:")
print(diag_cnt)
T1D_dataset['Diagnosis'].value_counts().plot(kind="pie", labels=['Positive', 'Negative'], colors=['steelblue', 'salmon'], autopct='%1.1f%%', fontsize=16)
plt.show()
print("After dropping rows with missing values we have:")
print(diag_cnt_no_missing)
T1D_no_missing['Diagnosis'].value_counts().plot(kind="pie", labels=['Positive', 'Negative'], colors=['steelblue', 'salmon'], autopct='%1.1f%%', fontsize=16)
plt.show()
print("Number of rows with missing values:")
len(T1D_dataset['Diagnosis'])-len(T1D_no_missing['Diagnosis'])


# We see that we lost a similar number of patients from each diagnosis class, and that the fractions of the classes are similar before and after deletion.
# We understand from it that by erasing full rows we did not change the balance dramatically.
# 
# However, deleting 42 lines out of 565 causes a loss of quite a bit of information, especially considering the fact that there are relatively few patients diagnosed as "Negative", so this information is important. 
# 
# For this reason we decided to complete the missing values by random sampling of features, in a way that takes the distribution of the feature into account:

# In[4]:


from functions import nan2samp
T1D_clean = nan2samp(T1D_dataset)


# ### **Part 2 - Train-Test Split**

# In[5]:


lbl = np.ravel(T1D_clean['Diagnosis'])
X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(T1D_clean, lbl, test_size=0.2, random_state=10, stratify=lbl)
X_train = X_train_tmp.drop(columns=['Diagnosis'])
X_test = X_test_tmp.drop(columns=['Diagnosis'])


# ### **Part 3 - Visualization and exploration of the data**
# 
# **3.a. An analysis to show that the distribution of the features is similar between test and train:**
# 
# We created a table showing distribution between each feature label in Train and Test Sets:

# In[6]:


from functions import dist_table as dist
X_test_dummy = pd.get_dummies(X_test, dummy_na=False, drop_first=True)
X_train_dummy = pd.get_dummies(X_train, dummy_na=False, drop_first=True)
d_table = dist(X_train_dummy, X_test_dummy)
d_table.transpose().round(decimals=2)


# **Q3.a.i:** What issues could an imbalance of features between train and test cause?
# 
# **Q3.a.i Answer:**
# 
# **Q3.a.ii:** How could you solve the issue?
# 
# **Q3.a.ii Answer:**
# 
# 
# **3.b. Plots to show the relationship between feature and label:**
# 
# We created a plot showing the frequency of each feature according to Diagnosis:

# In[7]:


from functions import feat_lab_cor as fl_cor
fl_cor(T1D_clean)


# The above graph makes it possible to distinguish a number of features that appear to be important for the diagnosis. For example, it can be seen that for "Increased Urination", almost all patients who testified that they had the symptom were diagnosed with the disease. This is a common symptom of T1D and therefore this result makes sense.
# 
# **3.c. Additional plots:**
# 
# We wanted to take a closer look into the "Age" feature, since it is not binary as the rest of the features:

# In[8]:


fig, axes = plt.subplots(figsize=(13, 8))
sns.set_context("paper", font_scale=1)
Age_cntplt = sns.countplot(ax=axes, x='Age', hue='Diagnosis', data=T1D_dataset)
Age_cntplt.set_title('Age frequency according to Diagnosis', fontsize=15)
plt.show()


# **3.d. Insights:**
# 
# *Some explanation as to why we decided to discard "Age"*
# 
# **Q3.d.i:** Was there anything unexpected?
# 
# **Q3.d.i Answer:**
# 
# **Q3.d.ii:** Are there any features that you feel will be particularly important to your model? Explain why.
# 
# **Q3.d.ii Answer:** 

# ### **Part 4 - Encoding all the data as one hot vectors**
# 
# Since we have already done a train-test split in part 2, we will perform the conversion to one hot vectors on the existing train and test sets (although it is possible to first convert to one hot vectors and then do the split).

# In[ ]:


# Encode X_train and X_test
X_columns = pd.get_dummies(X_train.iloc[:, 1:], dummy_na=False, drop_first=True).columns
X_train_ohe = pd.get_dummies(X_train.iloc[:, 1:], dummy_na=False, drop_first=True).to_numpy()
X_test_ohe = pd.get_dummies(X_test.iloc[:, 1:], dummy_na=False, drop_first=True).to_numpy()
# Encode y_train and y_test and
ohe = OneHotEncoder(sparse=False)
y_train_ohe = ohe.fit_transform(pd.DataFrame(y_train))
y_test_ohe = ohe.fit_transform(pd.DataFrame(y_test))
y_train_ohe_vec = y_train_ohe[:, 1]  # in the second column 0='Negative' and 1='Positive'
y_test_ohe_vec = y_test_ohe[:, 1]


# ### **Part 5 - Build and optimize Machine Learning Models**
# 
# **5.a.+5.b. Use 5k cross fold validation and tune the models to achieve the highest test AUC:**
# 
# **5.a.i. linear model- Logistic Regression**
# 
# first, we find the best hyper-parameters with the k-cross fold:

# In[ ]:


from functions import kcfold
C = np.array([0.01, 0.1, 1, 5, 10])  # regularization parameters
K = 5  # number of folds
penalty = ['l1', 'l2']  # types of penalties
val_dict = kcfold(X_train_ohe, y_train_ohe_vec, C=C, penalty=penalty, K=K)
val_dict_df = pd.DataFrame(val_dict)
idx_roc = val_dict_df['roc_auc'].idxmax()
print("The hyper parameters and AUROC for an optimized model are:")
val_dict[idx_roc]


# Now we can use the parameters to train the model and predict the diagnosis of y_test:

# In[ ]:


# choosing the best parameters and predict:
from functions import pred_log
c = 1
p = 'l2'
log_reg_best = LogisticRegression(solver='saga', multi_class='ovr', penalty=p, C=c, max_iter=10000, random_state=10)
y_pred_best, w_best = pred_log(log_reg_best, X_train_ohe, y_train_ohe_vec, X_test_ohe)
y_pred_p_best, w_p_best = pred_log(log_reg_best, X_train_ohe, y_train_ohe_vec, X_test_ohe, flag=True)
# evaluation metrics:
print("Evaluation metrics for logistic regression after 5K-cross fold validation:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_best), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_best))) + "%"))
print("loss is {:.2f}".format(metrics.log_loss(y_test_ohe_vec, y_pred_p_best[:, 1])))
print('AUROC is: {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p_best[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_best)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()


# **5.a.ii. non-linear model- non-linear SVM**
# 
# first, we find the best hyper-parameters with the k-cross fold using GridSearchCV:

# In[ ]:


n_splits = K
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
svc = SVC(probability=True)
C = np.array([1, 100, 1000])
pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
svm_nonlin = GridSearchCV(estimator=pipe, param_grid={'svm__C': C, 'svm__kernel': ['rbf', 'poly'],
                        'svm__gamma': ['auto', 'scale']}, scoring=['roc_auc'],
                        cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
svm_nonlin.fit(X_train_ohe, y_train_ohe_vec)
# Choose the best estimator and print them
best_svm_nonlin = svm_nonlin.best_estimator_
print("Non-linear SVM best parameters are:")
svm_nonlin.best_params_


# Now we can use the parameters to train the model and predict the diagnosis of y_test:

# In[ ]:


y_pred_test = best_svm_nonlin.predict(X_test_ohe)
y_pred_proba_test = best_svm_nonlin.predict_proba(X_test_ohe)
# evaluation metrics:
print("evaluation metrics for Non-linear SVM:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_test), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_test))) + "%"))
print("loss is {:.2f}".format(metrics.log_loss(y_test_ohe_vec, y_pred_proba_test[:, 1])))
print('AUROC is: {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_proba_test[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_test)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()


# **Q5.c:** What performs best on this dataset? Linear or non-linear models?
# 
# **Q5.c Answer:**

# ### **Part 6 - Feature Selection**
# 
# **6.a. Training a Random Forest on the data in order to explore feature importance:**

# In[ ]:


rf_clf = rfc(random_state=10)
rf_clf.fit(X_train_ohe, y_train_ohe_vec)
y_pred = rf_clf.predict(X_test_ohe)
y_pred_p_rf = rf_clf.predict_proba(X_test_ohe)
# evaluation metrics:
print("evaluation metrics for random forest classifier:")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test_ohe_vec, y_pred))) + "%")
print("F1 is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test_ohe_vec, y_pred, average='macro'))) + "%")
print('AUROC is {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p_rf[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()


# **6.a.i. finding the 2 most important features according to the random forest:**

# In[ ]:


features = rf_clf.feature_importances_
feature_names = np.array(X_columns)
sorted_idx = features.argsort()
y_ticks = np.arange(0, len(features))
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(y_ticks, features[sorted_idx])
ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importance",fontsize=16)
fig.tight_layout()
plt.show()


# **Q6.i:** What are the 2 most important features according to the random forest.
# 
# **Q6.i Answer:**
# 
# **Q6.ii:** Does this match up exactly with the feature exploration you did?
# 
# **Q6.ii Answer:**

# ### **Part 7 - Data Separability Visualization**
# 
# **7.a. Dimensionality reduction on the dataset and into 2D:**

# In[ ]:


scaler = StandardScaler()
X_train_ohe_scaled = scaler.fit_transform(X_train_ohe)
X_test_ohe_scaled = scaler.transform(X_test_ohe)
n_components = 2
pca = PCA(n_components=n_components, whiten=True)
X_train_ohe_pca = pca.fit_transform(X_train_ohe_scaled)
X_test_ohe_pca = pca.transform(X_test_ohe_scaled)
# plot the data in a 2d plot
from functions import plt_2d_pca
plt_2d_pca(X_test_ohe_pca,y_test_ohe_vec)


# In[ ]:


C = np.array([0.01, 0.1, 1, 5, 10])
val_dict_pca = kcfold(X_train_ohe_pca, y_train_ohe_vec, C=C, penalty=penalty, K=K)
val_dict_pca_df = pd.DataFrame(val_dict_pca)
idx_roc = val_dict_pca_df['roc_auc'].idxmax()
print("The hyper parameters and AUROC for an optimized model are:")
val_dict_pca[idx_roc]


# In[ ]:


c = 0.01
p = 'l2'
log_reg_pca = LogisticRegression(solver='saga', multi_class='ovr', penalty=p, C=c, max_iter=10000, random_state=10)
y_pred_pca, w_best = pred_log(log_reg_pca, X_train_ohe_pca, y_train_ohe_vec, X_test_ohe_pca)
y_pred_p_pca, w_p_best = pred_log(log_reg_pca, X_train_ohe_pca, y_train_ohe_vec, X_test_ohe_pca, flag=True)
# evaluation metrics:
print("Evaluation metrics for logistic regression after 5K-cross fold validation:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_pca), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_pca))) + "%"))
print("loss is {:.2f}".format(metrics.log_loss(y_test_ohe_vec, y_pred_p_pca[:, 1])))
print('AUROC is: {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p_pca[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_pca)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()


# In[ ]:


C = np.array([1, 100, 1000])
svm_nonlin_pca = GridSearchCV(estimator=pipe, param_grid={'svm__C': C, 'svm__kernel': ['rbf', 'poly'],
                        'svm__gamma': ['auto', 'scale']}, scoring=['roc_auc'],
                        cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
svm_nonlin_pca.fit(X_train_ohe_pca, y_train_ohe_vec)
# Choose the best estimator and print them
best_svm_nonlin_pca = svm_nonlin_pca.best_estimator_
print("Non-linear SVM best parameters are:")
svm_nonlin_pca.best_params_


# In[ ]:


y_pred_svm_pca = svm_nonlin_pca.predict(X_test_ohe_pca)
y_pred_proba_svm_pca = svm_nonlin_pca.predict_proba(X_test_ohe_pca)
# evaluation metrics:
print("evaluation metrics for Non-linear SVM:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_svm_pca), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_svm_pca))) + "%"))
print("loss is {:.2f}".format(metrics.log_loss(y_test_ohe_vec, y_pred_proba_svm_pca[:, 1])))
print('AUROC is: {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_proba_svm_pca[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_svm_pca)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()


# In[ ]:

# increased_urination_yes = T1D_dummy[:,3]
# increased_thirst_yes = T1D_dummy[:,4]
# new_T1D_dummy = np.array(df[[increased_urination_yes, increased_urination_yes])

Df_2_feat= T1D_clean[["Increased Urination", "Increased Thirst"]]
T1D_dummy_2_feat = pd.get_dummies(Df_2_feat, dummy_na=False, drop_first=False)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(T1D_dummy_2_feat, lbl, test_size=0.2, random_state=10, stratify=lbl)

C = np.array([0.01, 0.1, 1, 5, 10])
val_dict_2 = kcfold(X_train_2, y_train_2, C=C, penalty=penalty, K=K)
val_dict_2_df = pd.DataFrame(val_dict_2)
idx_roc = val_dict_2_df['roc_auc'].idxmax()
print("The hyper parameters and AUROC for an optimized model are:")
val_dict_2[idx_roc]


# In[ ]:


c = 0.01
p = 'l2'
log_reg_2 = LogisticRegression(solver='saga', multi_class='ovr', penalty=p, C=c, max_iter=10000, random_state=10)
y_pred_2, w_best = pred_log(log_reg_2, X_train_2, y_train_2, X_test_2)
y_pred_p_2, w_p_best = pred_log(log_reg_2, X_train_2, y_train_2, X_test_2, flag=True)
# evaluation metrics:
print("Evaluation metrics for logistic regression after 5K-cross fold validation:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_2), pd.DataFrame(y_pred_2), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred_2))) + "%"))
print("loss is {:.2f}".format(metrics.log_loss(y_test_2, y_pred_p_2[:, 1])))
print('AUROC is: {:.3f}'.format(roc_auc_score(y_test_2, y_pred_p_2[:, 1])))
# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test_2, y_pred_2)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

