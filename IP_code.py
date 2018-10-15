### CHURN MODELING - DATA SCIENCE PROJECT FOR INTRODUCTION TO PROGRAMMING #####
###############################################################################

import numpy as np
import pandas as pd
import random 


###############################################################################
###########  DATA PREPROCESSING  ##############################################
###############################################################################


### 1. IMPORTING THE DATASET ##################################################

dataset = pd.read_csv('Churn_Modelling.csv')
y = dataset.iloc[:, 13].values
X = dataset.iloc[:, :-1].values

#  Deleting randomly some data (for purpose of showing techniques for dealing with missing data (original data was complete))
a = np.shape(X)
i=0
for column in range(a[1]-1):
    j=0
    i+=1
    print('Columna: %d' %i)
    for row in range(a[0]-1):
        j+=1
        if i*j%random.randint(1000,2000)==0: # Generates few missing values per feature
            X[j,i] = ''
 
### 2. HANDLING MISSING DATA   ################################################     
            
#   2.1 Replacing missing data in categorical variable my most common value
from statistics import mode

for row in range(a[0]-1): #   3.2 Handling missing data in Geography
    if X[row, 4] == '':
        X[row, 4]=mode(X[:, 4])

for row in range(a[0]-1): #   3.3 Handling missing data in Gender  
    if X[row, 5] == '':
        X[row, 5]=mode(X[:, 5])  

for row in range(a[0]-1): #   3.4 Handling missing data in HasCrCard  
    if X[row, 10] == '':
        X[row, 10]=mode(X[:, 10])        

for row in range(a[0]-1): #   3.5 Handling missing data in IsActive  
    if X[row, 11] == '':
        X[row, 11]=mode(X[:, 11])        


#   2.2 Replacing missing data in countinous variable with mean of column
from sklearn.preprocessing import Imputer

df = pd.DataFrame(X)   # Tranfer to data frame to use df.where function        
df = df.where(df!='')  # Replacing empty values with NaN
X = df.iloc[:, 3:].values # Deleting unnecessary column (no logical input on outcome)

imputer = Imputer(missing_values=np.nan, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[0,3,4,5,6,9]])
X[:,[0,3,4,5,6,9]] = imputer.transform(X[:,[0,3,4,5,6,9]])

#   for column age, tenure, num of prod we need to round number to integer (logic constraint)
X[:,[3,4,6]] = X[:,[3,4,6]].astype(int)

### 3 ENCODING CATEGORICAL VARIABLES FOR GEOGRAPHT AND GENDER
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #this also works with pandas!

labelEncoder1 = LabelEncoder()
X[:,1] = labelEncoder1.fit_transform(X[:,1])
labelEncoder2 = LabelEncoder()
X[:,2] = labelEncoder2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

### 4 SPLITTING THE DATASET INTO TRAINING SET AND TEST SET ####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#from sklearn.model_selection import StratifiedShuffleSplit
#sss = StratifiedShuffleSplit(y, n_iter=2, test_size=0.3)
#train_index, test_index = next(iter(sss))
#X_train = features.iloc[train_index]
#y_train = y.iloc[train_index]
#X_test = features.iloc[test_index]
#y_test = y.iloc[test_index]

### z-scoring with scaling dummy variables ##################
#from scipy.stats import zscore

#X_train = pd.DataFrame(X_train)#converting to use the df.methods
#X_test = pd.DataFrame(X_test)#converting to use the df.methods

#X_train = X_train.apply(zscore)#applying the z-scoring
#X_test = X_test.apply(zscore)#applying the z-scoring


### 5 FEATURE STANDARD SCALING  with scaling dummy variables ##################
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

### 5 FEATURE STANDARD SCALING  without scaling dummy variables ###############
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,3:4] = sc_X.fit_transform(X_train[:,3:4])
X_test[:,3:4] = sc_X.transform(X_test[:,3:4])
X_train[:,5:6] = sc_X.fit_transform(X_train[:,5:6])
X_test[:,5:6] = sc_X.transform(X_test[:,5:6])
X_train[:,7:9] = sc_X.fit_transform(X_train[:,7:9])
X_test[:,7:9] = sc_X.transform(X_test[:,7:9])
X_train[:,11:12] = sc_X.fit_transform(X_train[:,11:12])
X_test[:,11:12] = sc_X.transform(X_test[:,11:12])

### z-scoring  without scaling dummy variables ###############
#from scipy.stats import zscore


#X_train[:,3:4] = zscore(X_train[:,3:4])
#X_test[:,3:4] = zscore(X_test[:,3:4])
#X_train[:,5:6] = zscore(X_train[:,5:6])
#X_test[:,5:6] = zscore(X_test[:,5:6])
#X_train[:,7:9] = zscore(X_train[:,7:9])
#X_test[:,7:9] = zscore(X_test[:,7:9])
#X_train[:,11:12] = zscore(X_train[:,11:12])
#X_test[:,11:12] = zscore(X_test[:,11:12])




###############################################################################
###########  DATA TRANSFORMATION AND VISULIZATION  ############################
###############################################################################

### 1 BASIC DESCRIPTIVE STATISTICS ############################################
import matplotlib.pyplot as plt
X = pd.DataFrame(X)
X[5].describe()

dataset.groupby('Exited').size()
benchmark = max(dataset.groupby('Exited').size()[0], dataset.groupby('Exited').size()[0])/(dataset.groupby('Exited').size()[0] + dataset.groupby('Exited').size()[1])
print("{0:.2%}".format(benchmark))

count_0=0
count_all=0
for i in range(len(y_test)):
    count_all+=1
    if y_test[i]==0:
        count_0+=1

count_0/count_all

### 2 HISTOGRAMS AND SCATTER PLOT #############################################
# interesting columns for histograms3,5,6,7,8,11
plt.hist(X[3])
plt.hist(X[5]) # possible dyscretization opporutity to split into young 0-30, 30-40, 40+
plt.hist(X[6])
plt.hist(X[7]) # split into those with 0, 0-100k, 100k-125k - 125k+
plt.hist(X[8]) # mayby it should be encoded into 1, 2 and other as binnary data
plt.hist(X[11])

#from pandas.plotting import scatter_matrix
#scatter_matrix(dataset)
#plt.show()

#df.plot(kind='box', subplots=True, layout=(3, 4), sharex = False, sharey=False)
#plt().show()

### 3 CORRELATION #############################################################

pd.options.display.max_columns = 12
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',30)
#pd.reset_option('^display.', silent=True)
X.corr()
X.cov()

#visusalistion of the correlations in a heatmap
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})
sns.heatmap(X.corr(), annot=True)
sns.heatmap(X.cov(), annot=True)


### Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)

# Builiding the optimal model (Backward Elimination)
import statsmodels.formula.api as sm
#X = pd.DataFrame(X)
#X = dataset.values
X = np.append(arr = np.ones((10000, 1)).astype(int), values = X,  axis=1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,11,12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
X_opt = X[:,[1,2,3,4,5,6,8,9,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[1,3,4,5,6,8,9,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

regressor = LinearRegression()
regressor.fit(X_train[:,[1,3,4,5,6,8,9,11]], y_train)
y_pred = regressor.predict(X_test[:,[1,3,4,5,6,8,9,11]])


plt.hist(y_pred)

for q in range(len(y_pred)):
    if y_pred[q]<0.48: #(max(y_pred) - min(y_pred))/2:
        y_pred[q]=0
    else: y_pred[q] = 1
    
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

a = pd.DataFrame(y_pred)
a.describe()    
    
# Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red' )
#plt.plot(X_train, regressor.predict(X_train), color = 'green')
#plt.title('Churn Model \n(Train Set)')
#plt.xlabel('Variables')
#plt.ylabel('IsExited')
#plt.show()

#### 4 Applying PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 4)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
#
#
#### 5 Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state=10)
#classifier.fit(X_train, y_train)
#
## Predict the Test set results
#y_pred = classifier.predict(X_test)
#
## Confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


## Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Logistic Regression (Training set)')
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.legend()
#plt.show()

## Visualising the Test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
#plt.title('Logistic Regression (Test set)')
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.legend()
#plt.show()



