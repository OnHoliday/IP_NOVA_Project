### CHURN MODELING - DATA SCIENCE PROJECT FOR INTRODUCTION TO PROGRAMMING #####
###############################################################################

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

###########  DATA PREPROCESSING  ##############################################
###############################################################################


### 1. IMPORTING THE DATASET ##################################################

dataset = pd.read_csv(r'C:\Users\dominika.leszko\Desktop\NOVA IMS\Introduction to Programming\PROJECT\Churn_Modelling.csv')

### 2. HANDLING MISSING VALUES ################################################

dataset.isna().any()
dataset.isnull().sum(axis=0)

y = dataset.iloc[:, 13]
X = dataset.iloc[:, :-1]

### 3. FEATURE SELECTION ######################################################

X.columns.values
X.drop(['Surname'], axis=1, inplace=True)
X.drop(['CustomerId'], axis=1, inplace=True)
X.drop(['RowNumber'], axis=1, inplace=True)

### 4. HANDLING ERRORS  ######################################################


# excluding binary variables for boxplots
X_boxplot = X[[i for i in list(X.columns) if i not in ['HasCrCard', 'IsActiveMember']]]

# boxplots for outliers
plt.style.use('ggplot')
X_boxplot.plot(kind='box', subplots=True, layout=(2, 3), sharex=False, sharey=False, figsize=(20, 10))
plt.suptitle('Column Boxplots', size=25, color='orange', fontweight='bold')

from pandas.plotting import scatter_matrix

scatter_matrix(dataset)
plt.show()

# histograms for columns of interest
sns.set_style('darkgrid')
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

sns.distplot(X['CreditScore'], kde=False, color='red', ax=ax1)
sns.distplot(X['Age'], kde=False, color='red', ax=ax2)
sns.distplot(X['Tenure'], kde=False, color='red', ax=ax3)
sns.distplot(X['Balance'], kde=False, color='red', ax=ax4)
sns.distplot(X['NumOfProducts'], kde=False, color='red', ax=ax5)
sns.distplot(X['EstimatedSalary'], kde=False, color='red', ax=ax6)

plt.suptitle('Column Histograms', size=25, color='grey', fontweight='bold')

sns.factorplot('Geography', 'Exited', hue='Gender', data=dataset)
plt.show()

sns.FacetGrid(dataset, hue="Exited", size=5).map(sns.kdeplot, "Balance").add_legend()
plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot("Geography", "Age", hue="Exited", data=dataset, split=True, ax=ax[0])
ax[0].set_title('Geography and Age vs Exited')
ax[0].set_yticks(range(0, 110, 10))
sns.factorplot('Geography', 'Exited', hue='Gender', data=dataset, split=True, ax=ax[1])
ax[1].set_title('Geography and Gender vs Exited')
ax[1].set_yticks(range(0, 1, 3))
plt.show()

tab = pd.crosstab(dataset['Geography'], dataset['Exited'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Geography')
dummy = plt.ylabel('Percentage')

# log transformation on Age
X['Age'] = np.log(X['Age'])

### 5. ENCODING CATEGORICAL VARIABLES #########################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder2 = LabelEncoder()
X['Gender'] = labelEncoder2.fit_transform(X['Gender'])

degree_dummies = pd.get_dummies(X['Geography'], drop_first=False)
X.drop(['Geography'], axis=1, inplace=True)
X = pd.concat([X, degree_dummies], axis=1)

X = X.apply(pd.to_numeric)
X.info()
### 5. SPLITTING THE DATASET INTO TRAINING SET AND TEST SET ###################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_ind = X_train.index.values
X_test_ind = X_test.index.values

## 6 FEATURE STANDARD SCALING #################################################
#   6.1 Standardize features by removing the mean and scaling to unit variance
#    6.1.1 with scaling dummy variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# to df
X_train = pd.DataFrame(data=X_train, index=X_train_ind, columns=list(X.columns.values))
X_test = pd.DataFrame(data=X_test, index=X_test_ind, columns=list(X.columns.values))

#    6.1.1 without scaling dummy variables

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train[:,3:4] = sc_X.fit_transform(X_train[:,3:4])
# X_test[:,3:4] = sc_X.transform(X_test[:,3:4])
# X_train[:,5:6] = sc_X.fit_transform(X_train[:,5:6])
# X_test[:,5:6] = sc_X.transform(X_test[:,5:6])
# X_train[:,7:9] = sc_X.fit_transform(X_train[:,7:9])
# X_test[:,7:9] = sc_X.transform(X_test[:,7:9])
# X_train[:,11:12] = sc_X.fit_transform(X_train[:,11:12])
# X_test[:,11:12] = sc_X.transform(X_test[:,11:12])

#   6.2 Transforms features by scaling each feature to a given range.

# from sklearn.preprocessing import MinMaxScaler
#
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)


###########2  DATA TRANSFORMATION AND VISULIZATION  ###########################
###############################################################################

### 1 BASIC DESCRIPTIVE STATISTICS ############################################ its all on the X is that right ??
import matplotlib.pyplot as plt

X.describe()

dataset.groupby('Exited').size()
benchmark = max(dataset.groupby('Exited').size()[0], dataset.groupby('Exited').size()[0]) / (
            dataset.groupby('Exited').size()[0] + dataset.groupby('Exited').size()[1])

print(pd.DataFrame(y_test).groupby('Exited').size())
print('>>>   Percentage of clients who Exited is : {0:.2%}'.format(
    y.index.isin(list(y.index[(y == 1) == True])).sum() / (
                y.index.isin(list(y.index[(y == 1) == True])).sum() + y.index.isin(
            list(y.index[(y == 1) == False])).sum())))
print(">>>   Base line accuracy of prediction is {0:.2%}".format(benchmark))

#### EXPLORATORY DATA VISUALIZATION#############################################

# Pairplots
train_df = pd.concat([X_train, y_train], axis=1)

sns.pairplot(train_df, hue="Exited", vars=[
    'CreditScore'
    #                                              , 'Gender'
    , 'Age'
    #                                              , 'Tenure'
    , 'Balance'
    #                                              , 'NumOfProducts'
    #                                              , 'HasCrCard'
    #                                              , 'IsActiveMember'
    #                                              , 'EstimatedSalary'
    #                                              , 'France'
    #                                              , 'Germany'
    #                                              , 'Spain'
]
             , diag_kind="kde")

# Active vs inactive members exiting
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
plt.suptitle('Number of Exits among Active and Inactive Members', size=13, color='grey', fontweight='bold')
sns.countplot(dataset[dataset['IsActiveMember'] == 1]['Exited'], ax=ax[0])
sns.countplot(dataset[dataset['IsActiveMember'] == 0]['Exited'], ax=ax[1])
ax[0].set_xlabel('Active Members', color='green')
ax[0].set_ylabel(' ')
ax[1].set_xlabel('Inactive Members', color='red')
ax[1].set_ylabel(' ')
# add labels %

# Exits per country
fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
plt.suptitle('Number of Exits per country', size=13, color='grey', fontweight='bold')
sns.countplot(dataset[dataset['Geography'] == 'Spain']['Exited'], ax=ax[0])
sns.countplot(dataset[dataset['Geography'] == 'France']['Exited'], ax=ax[1])
sns.countplot(dataset[dataset['Geography'] == 'Germany']['Exited'], ax=ax[2])
ax[0].set_xlabel('Spain')
ax[0].set_ylabel(' ')
ax[1].set_xlabel('France')
ax[1].set_ylabel(' ')
ax[2].set_xlabel('Germany')
ax[2].set_ylabel(' ')

### 3 CORRELATION #############################################################

pd.options.display.max_columns = 12
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth', 30)
# pd.reset_option('^display.', silent=True)
sns.heatmap(X.corr(), annot=True)

###########  MODELING DATASET  ################################################
###############################################################################

####  2. MULTIPLE LINEAR REGRESSION WITH BACKWARD ELIMINATION #################
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression

X_opt = X.copy()
X_train_opt = X_train.copy()
X_test_opt = X_test.copy()
y_train_opt = y_train.copy()

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['HasCrCard'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['EstimatedSalary'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['Tenure'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['Germany'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['NumOfProducts'], axis=1,
           inplace=True)  # Although below the adopted threshold worth to consider deleting this column also
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

regressor = LinearRegression()
X_train_opt.drop(['HasCrCard', 'EstimatedSalary', 'Tenure', 'Germany', 'NumOfProducts'], axis=1, inplace=True)
regressor.fit(X_train_opt, y_train_opt)
X_test_opt.drop(['HasCrCard', 'EstimatedSalary', 'Tenure', 'Germany', 'NumOfProducts'], axis=1, inplace=True)
y_pred = regressor.predict(X_test_opt)

plt.hist(y_pred)

# Assiging predicted values into decision
for q in range(len(y_pred)):
    if y_pred[q] < (max(y_pred) - min(y_pred)) / 2:
        y_pred[q] = 0
    else:
        y_pred[q] = 1

# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_reg = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

##### 2 lOGISTIC REGRESSION #############################################

X_pca = X.copy()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
y_train_pca = y_train.copy()

### 5 Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=10)
LR.fit(X_train_pca, y_train_pca)

# Predict the Test set results
y_pred = LR.predict(X_test_pca)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_log = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)
print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

##### 2 lOGISTIC REGRESSION WITH PCA #############################################

X_pca = X.copy()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
y_train_pca = y_train.copy()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)
explained_variance = pca.explained_variance_ratio_

### 5 Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

LR_PCA = LogisticRegression(random_state=10)
LR_PCA.fit(X_train_pca, y_train_pca)

# Predict the Test set results
y_pred = LR_PCA.predict(X_test_pca)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_log_pca = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

# Visualising the Test set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_test_pca, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, LR_PCA.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#### 3 Decision Tree ###############################

X_tre = X.copy()
X_train_tre = X_train.copy()
X_test_tre = X_test.copy()
y_train_tre = y_train.copy()

from sklearn.tree import DecisionTreeClassifier  # importing the tree

dtree = DecisionTreeClassifier(random_state=0)  # setting up a tree
dtree.fit(X_train_tre, y_train_tre)  # using the train data to train the tree
y_pred = dtree.predict(X_test_tre)  # making predicitons on the test data
from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print('Descision Tree')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)  # printing the confusion matrix
print(cm)
tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)

cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_tree = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

### in my opinion we should try to focus on predicting true positives and min the type II error -> amining for a sensitivity, hence applying the random forest
## is 1 = potivie and 0 negative ?

# problem it changes everytime i dont know why


#### random forest ####

X_rf = X.copy()
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
y_train_rf = y_train.copy()

from sklearn.ensemble import RandomForestClassifier  # omporting the RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300,
                             random_state=0)  # setting up a RFC with 300 estimators (number of randomly distributed trees classifing the object; default is 100)
rfc.fit(X_train_rf, y_train_rf)  # using the train data to train the rfc
y_pred = rfc.predict(X_test_rf)  # making predicitons on the test data
from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print('Random Forest')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)  # printing the confusion matrix
print(cm)
tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)

cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_rf = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

###turns out that type II error increases, the model is wore for us..


# My tree
#
# X_tre = X.copy()
# X_train_tre = X_train.copy()
# X_test_tre = X_test.copy()
# y_train_tre = y_train.copy()
#
# # Fitting Decision Tree Classification to the Training set
# from sklearn.tree import DecisionTreeClassifier
#
# dt_entr = DecisionTreeClassifier(criterion='entropy')
# dt_entr.fit(X_train_tre, y_train_tre)
#
# # Predicting the Test set results
# y_pred = dt_entr.predict(X_test_tre)
#
# # Making the Confusion Matrix
# from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods
#
# cm = confusion_matrix(y_test, y_pred)
#
# print(classification_report(y_test, y_pred))  # printing the confusion matrix
# print(cm)
# tn, fp, fn, tp = cm.ravel()
# sen = tp / (tp + fn)
# print('True positive  = ', tp)
# print('False positive = ', fp)
# print('False negative = ', fn)
# print('True negative  = ', tn)
# print('Sensitivity  = ', sen)
#
# cm = confusion_matrix(y_test, y_pred)
# (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])
#
# benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
# print(cm)
# score_tree_entro = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)
#
# print('>>>   Accuracy of this model is : {0:.2%}'.format(
#     (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
# print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
#     cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
# print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
# print(">>>   Current best prediction is {0:.2%}".format(benchmark))
#
# ### My forrest
#
# X_rf = X.copy()
# X_train_rf = X_train.copy()
# X_test_rf = X_test.copy()
# y_train_rf = y_train.copy()
#
# # Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
#
# rf_entr = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
# rf_entr.fit(X_train_rf, y_train_rf)
#
# # Predicting the Test set results
# y_pred = rf_entr.predict(X_test_rf)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)
#
# print(classification_report(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)  # printing the confusion matrix
# print(cm)
# tn, fp, fn, tp = cm.ravel()
# sen = tp / (tp + fn)
# print('True positive  = ', tp)
# print('False positive = ', fp)
# print('False negative = ', fn)
# print('True negative  = ', tn)
# print('Sensitivity  = ', sen)
#
# cm = confusion_matrix(y_test, y_pred)
# (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])
#
# benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
# print(cm)
# score_rf_entro = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)
#
# print('>>>   Accuracy of this model is : {0:.2%}'.format(
#     (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
# print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
#     cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
# print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
# print(">>>   Current best prediction is {0:.2%}".format(benchmark))

#############################SUPPORT VECTOR MACHINES###########################

X_svm = X.copy()
X_train_svm = X_train.copy()
X_test_svm = X_test.copy()
y_train_svm = y_train.copy()

from sklearn.svm import SVC

svc = SVC(verbose=True, random_state=0)

svc.fit(X_train_svm, y_train_svm)

predictions = svc.predict(X_test_svm)

from sklearn.metrics import confusion_matrix, classification_report

svm_matrix = confusion_matrix(y_test, predictions)

print(svm_matrix)

(svm_matrix[0][0] + svm_matrix[1][1]) / (svm_matrix[0][0] + svm_matrix[1][1] + svm_matrix[0][1] + svm_matrix[1][0])

print(classification_report(y_test, predictions))  # DOMINIKA: Interpretation

print('>>>   Accuracy of this model is : {0:.2%}'.format((svm_matrix[0][0] + svm_matrix[1][1]) / (
            svm_matrix[0][0] + svm_matrix[1][1] + svm_matrix[0][1] + svm_matrix[1][0])))

from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)  # printing the confusion matrix
print(cm)
tn, fp, fn, tp = svm_matrix.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)

cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_svc = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

############################### NAive BAis

import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

# predictions based on binary feautures
BernNB = BernoulliNB(binarize=True)
BernNB.fit(X_train, y_train)
print(BernNB)

predicted_y = BernNB.predict(X_test)
expected_y = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(expected_y, predicted_y))

from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)  # printing the confusion matrix
print(cm)
score_nb = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))

#### 3 SINGLE LAYER NEURAL NETWORL - PERCEPTRON ###############################

X_ann = X.copy()
X_train_ann = np.array(X_train.copy())
X_test_ann = np.array(X_test.copy())
y_train_ann = np.array(y_train.copy())


class NeuralNetwork():

    def __init__(self):
        np.random.seed(4)
        self.synaptic_weights = 2 * np.random.random((12, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for interation in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustment = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            # self.synaptic_weights = self.synaptic_weights + adjustment
            self.synaptic_weights += adjustment

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


y_train_ann = pd.DataFrame(y_train_ann)
neural_network = NeuralNetwork()
training_inputs = np.array([list(l) for l in X_train_ann])
training_outputs = np.array(list(np.array([list(a) for a in y_train_ann.values]).T.tolist())).T
neural_network.train(training_inputs, training_outputs, 100000)
print("Ending Weights After Training: ")
print(neural_network.synaptic_weights)

y_pred = []
for i in range(len(X_test_ann)):
    y_pred.append(neural_network.think(X_test_ann[i]))

for q in range(len(y_pred)):
    if y_pred[q] < 0.50:
        y_pred[q] = 0
    else:
        y_pred[q] = 1

cm = confusion_matrix(y_test, y_pred)
ann_accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]) * 100

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
score_NN = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))


### TABLe WITH SCORE COMPARISSON

def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_test, y_test)
    train_accuracy = np.round(train_accuracy * 100, 2)
    return train_accuracy


'''Models with best training accuracy:'''
train_accuracy = pd.DataFrame({'Train_accuracy(%)': [train_accuracy(LR),
                                                     train_accuracy(LR_PCA), train_accuracy(dtree), train_accuracy(rfc),
                                                     train_accuracy(BernNB), train_accuracy(svc)]})
train_accuracy.index = ['Logistic Regression', 'Logistic Regression with PCA', 'Deision Tree', 'Random Forest', 
                        'Bernoulli Naive Bayes', 'Support Vector Classification']
train_accuracy.loc['ANN'] = ann_accuracy
sorted_train_accuracy = train_accuracy.sort_values(by='Train_accuracy(%)', ascending=False)
print('**Training Accuracy of the Classifiers:**')
print(sorted_train_accuracy)

'''Create a function that returns mean cross validation score for different models.'''


def x_val_score(model):
    from sklearn.model_selection import cross_val_score
    x_val_score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy').mean()
    x_val_score = np.round(x_val_score * 100, 2)
    return x_val_score


"""Let's perform k-fold (k=10) cross validation to find the classifier with the best cross validation accuracy."""
x_val_score = pd.DataFrame({'X_val_score(%)': [x_val_score(LR), x_val_score(LR_PCA), x_val_score(dtree),
                                               x_val_score(rfc),
                                               x_val_score(BernNB), x_val_score(svc)]})
x_val_score.index = ['Logistic Regression', 'Logistic Regression with PCA', 'Deision Tree', 'Random Forest', 
                        'Bernoulli Naive Bayes', 'Support Vector Classification']
sorted_x_val_score = x_val_score.sort_values(by='X_val_score(%)', ascending=False)
print('**Models 10-fold Cross Validation Score:**')
print(sorted_x_val_score)

# '''For SVC, the following hyperparameters are usually tunned.'''
# svc_params = {'C': [6, 7, 8, 9, 10, 11, 12],
#              'kernel': ['linear','rbf'],
#              'gamma': [0.5, 0.2, 0.1, 0.001, 0.0001]}
#
# """Tune SVC's hyperparameters."""
# tune_hyperparameters(svc, params = svc_params)
# svc_best_score, svc_best_params = best_score, best_params


# '''Create a dataframe of tunned scores and sort them in descending order.'''
# tunned_scores = pd.DataFrame({'Tunned_accuracy(%)': [lr_best_score, gbc_best_score, svc_best_score, dt_best_score, rf_best_score, knn_best_score, abc_best_score, etc_best_score]})
# tunned_scores.index = ['LR', 'GBC', 'SVC', 'DT', 'RF', 'KNN', 'ABC', 'ETC']
# sorted_tunned_scores = tunned_scores.sort_values(by = 'Tunned_accuracy(%)', ascending = False)
# bold('**Models Accuracy after Optimization:**')
# display(sorted_tunned_scores)

'''#4.Create a function that compares cross validation scores with tunned scores for different models by plotting them.'''


def compare_scores(accuracy):
    global ax1
    font_size = 15
    title_size = 18
    ax1 = accuracy.plot.bar(legend=False, title='Models %s' % ''.join(list(train_accuracy.columns)), figsize=(18, 5),
                            color='sandybrown')
    ax1.title.set_size(fontsize=title_size)
    # Removes square brackets and quotes from column name after to converting list.
    #    pct_bar_labels()
    plt.ylabel('% Accuracy', fontsize=font_size)
    plt.show()


'''Compare cross validation scores with tunned scores to find the best model.'''
# bold('**Comparing Cross Validation Scores with Optimized Scores:**')
compare_scores(sorted_train_accuracy)


def compare_scores_val(accuracy):
    global ax1
    font_size = 15
    title_size = 18
    ax1 = accuracy.plot.bar(legend=False, title='Models %s after Cross Validation', figsize=(18, 5), color='sandybrown')
    ax1.title.set_size(fontsize=title_size)
    # Removes square brackets and quotes from column name after to converting list.
    #    pct_bar_labels()
    plt.ylabel('% Accuracy', fontsize=font_size)
    plt.show()


compare_scores_val(sorted_x_val_score)


#import matplotlib.pyplot as plt
#
mon_1 = ['Random Forrest', 'Support Vector Machine', 'Naive Bays', 'Logistic Regression','Logistic regression with PCA', 'Decision Tree', 'Perceptron', 'Linear Regression']
mon_2 = [score_rf, score_svc, score_nb, score_log, score_log_pca, score_tree, score_NN, score_reg]
mon_1 = mon_1[::-1]
mon_2 = mon_2[::-1]

import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
y_pos = np.arange(len(mon_1))
my_colors=['red','red','red', 'red', 'red', 'red', 'green', 'green']
plt.barh(y_pos, mon_2, align='center', alpha=0.5, color=my_colors)
plt.yticks(y_pos, mon_1)
plt.xlabel('Score')
plt.title('Monetary score of each model')

plt.show()


acc = train_accuracy.reset_index()
acc = acc.sort_values(by=['Train_accuracy(%)'], ascending=True)
acc_1 = list(acc['index'])
acc_2 = list(acc['Train_accuracy(%)'])

sns.set_style('darkgrid')
y_pos = np.arange(len(acc_1))

plt.barh(y_pos, acc_2, align='center', alpha=0.5)
plt.yticks(y_pos, acc_1)
plt.xlabel('Score')
plt.title('Accuracy of each model')

plt.show()

acc = x_val_score.reset_index()
acc = acc.sort_values(by=['X_val_score(%)'], ascending=True)
acc_1 = list(acc['index'])
acc_2 = list(acc['X_val_score(%)'])

sns.set_style('darkgrid')
y_pos = np.arange(len(acc_1))

plt.barh(y_pos, acc_2, align='center', alpha=0.5)
plt.yticks(y_pos, acc_1)
plt.xlabel('Score')
plt.title('Accuracy of each model after CrossValidation')

plt.show()



#
#
#####feature importance visualisation

import pandas as pd

feature_importances_rfc = pd.DataFrame(rfc.feature_importances_,
                                       index=X_train_rf.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
sns.barplot(x=feature_importances_rfc.index.values, y="importance", data=feature_importances_rfc)
plt.show()

from sklearn.model_selection import cross_val_score


def ROI(estimator, x, y):
    yPred = estimator.predict(x)
    cm = confusion_matrix(y, yPred)
    ROI = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

    return ROI

def cv(model):
    rfc_ROI = cross_val_score(model, X_train_rf, y_train_rf, scoring=ROI, cv=12, n_jobs=-2).mean()
    return rfc_ROI


a = cv(rfc)

cv_money = pd.DataFrame({'Train_accuracy(%)': [cv(LR), cv(LR_PCA), cv(dtree), cv(rfc), cv(BernNB), cv(svc)]})
cv_money.index = ['LogReg', 'LogReg_PCA', 'Dec_tree', 'RandForr', 'BernNB', "svc"]
sortedcv_money = cv_money.sort_values(by='Monetary Score after Cross-Validation', ascending=False)
print('**Training Accuracy of the Classifiers:**')
print(sortedcv_money)

acc = cv_money.reset_index()
acc = acc.sort_values(by=['Train_accuracy(%)'], ascending=True)
acc_1 = list(['Random Forrest', 'Support Vector Machine', 'Naive Bays', 'Logistic Regression','Logistic regression with PCA', 'Decision Tree'])
acc_2 = list(acc['Train_accuracy(%)'])


y_pos = np.arange(len(acc_1))
sns.set_style('darkgrid')
my_colors=['red', 'red', 'red', 'red', 'green', 'green']
acc['Train_accuracy(%)'].plot(kind='barh', color=my_colors, alpha=0.5)
plt.yticks(y_pos, acc_1)
plt.title('Monetary Score of each model after Cross-Validation')
plt.show()

sns.set_style('darkgrid')
y_pos = np.arange(len(acc_1))

plt.barh(y_pos, acc_2, align='center', alpha=0.5)
plt.yticks(y_pos, acc_1)
plt.xlabel('Score')
plt.title('Accuracy of each model after Cross-Validation')

plt.show()

###GRID SEARCH FOR BETTER RESULTS########

from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['gini', 'entropy'],'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [2, 5, 10, 50, 100],
              'min_samples_split': [2, 5, 10, 50, 100]}

grid = GridSearchCV(RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0), param_grid,
                    refit=True, verbose=2, n_jobs=-1)

grid.fit(X_train_rf, y_train_rf)

# Checking best parameters
grid.best_params_

grid_predictions=grid.predict(X_test)

grid_matrix = confusion_matrix(y_test, grid_predictions)

print(grid_matrix)

print(classification_report(y_test, grid_predictions))

(grid_matrix[0][0] + grid_matrix[1][1]) / (
            grid_matrix[0][0] + grid_matrix[1][1] + grid_matrix[0][1] + grid_matrix[1][0])
print('>>>   Accuracy of this model is : {0:.2%}'.format((grid_matrix[0][0] + grid_matrix[1][1]) / (
            grid_matrix[0][0] + grid_matrix[1][1] + grid_matrix[0][1] + grid_matrix[1][0])))

from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print(classification_report(y_test, grid_predictions))
cm = confusion_matrix(y_test, grid_predictions)  # printing the confusion matrix
print(cm)
tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)
cm = confusion_matrix(y_test, grid_predictions)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))
f_Score_rf = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)

###GRID SEARCH FOR BETTER RESULTS########

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)

grid.fit(X_train_svm, y_train_svm)

# Checking best parameters
grid.best_params_

grid_predictions = grid.predict(X_test)

grid_matrix = confusion_matrix(y_test, grid_predictions)

print(grid_matrix)

print(classification_report(y_test, grid_predictions))

(grid_matrix[0][0] + grid_matrix[1][1]) / (
            grid_matrix[0][0] + grid_matrix[1][1] + grid_matrix[0][1] + grid_matrix[1][0])
print('>>>   Accuracy of this model is : {0:.2%}'.format((grid_matrix[0][0] + grid_matrix[1][1]) / (
            grid_matrix[0][0] + grid_matrix[1][1] + grid_matrix[0][1] + grid_matrix[1][0])))

from sklearn.metrics import classification_report, confusion_matrix  # importing reporting methods

print(classification_report(y_test, grid_predictions))
cm = confusion_matrix(y_test, grid_predictions)  # printing the confusion matrix
print(cm)
tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn)
print('True positive  = ', tp)
print('False positive = ', fp)
print('False negative = ', fn)
print('True negative  = ', tn)
print('Sensitivity  = ', sen)
cm = confusion_matrix(y_test, grid_predictions)
(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format(
    (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print('>>>   The investment return when this model is applied is:  €{:0,.2f}'.format(
    cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)).replace('€-', '-€'))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark))
print(">>>   Current best prediction is {0:.2%}".format(benchmark))
f_Score_svc = cm[0][0] * 0 + cm[1][1] * 500 + cm[1][0] * (-140) + cm[0][1] * (-450)


plt.bar(y_pos, f2, alpha=0.5, color='lightgreen')
plt.xticks(y_pos, f1)
plt.ylabel('Score')
plt.title('Final monetary value of each model')
plt.show()

