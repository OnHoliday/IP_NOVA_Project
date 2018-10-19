### CHURN MODELING - DATA SCIENCE PROJECT FOR INTRODUCTION TO PROGRAMMING #####
###############################################################################

import numpy as np
import pandas as pd
import random 
import seaborn as sns


###########  DATA PREPROCESSING  ##############################################
###############################################################################


### 1. IMPORTING THE DATASET ##################################################

dataset = pd.read_csv('Churn_Modelling.csv')


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


### 4. ENCODING CATEGORICAL VARIABLES #########################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder1 = LabelEncoder()
#X['Geography'] = labelEncoder1.fit_transform(X['Geography'])
labelEncoder2 = LabelEncoder()
X['Gender'] = labelEncoder2.fit_transform(X['Gender'])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()

degree_dummies=pd.get_dummies(X['Geography'], drop_first=False)
X.drop(['Geography'], axis=1, inplace=True)
X=pd.concat([X, degree_dummies], axis=1)


### 5. SPLITTING THE DATASET INTO TRAINING SET AND TEST SET ###################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


## 6 FEATURE STANDARD SCALING #################################################
#   6.1 Standardize features by removing the mean and scaling to unit variance
#    6.1.1 with scaling dummy variables 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#    6.1.1 without scaling dummy variables 

#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train[:,3:4] = sc_X.fit_transform(X_train[:,3:4])
#X_test[:,3:4] = sc_X.transform(X_test[:,3:4])
#X_train[:,5:6] = sc_X.fit_transform(X_train[:,5:6])
#X_test[:,5:6] = sc_X.transform(X_test[:,5:6])
#X_train[:,7:9] = sc_X.fit_transform(X_train[:,7:9])
#X_test[:,7:9] = sc_X.transform(X_test[:,7:9])
#X_train[:,11:12] = sc_X.fit_transform(X_train[:,11:12])
#X_test[:,11:12] = sc_X.transform(X_test[:,11:12])

#   6.2 Transforms features by scaling each feature to a given range.

#from sklearn.preprocessing import MinMaxScaler
#
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)




###########  DATA TRANSFORMATION AND VISULIZATION  ############################
###############################################################################

### 1 BASIC DESCRIPTIVE STATISTICS ############################################
import matplotlib.pyplot as plt

X['Balance'].describe()

dataset.groupby('Exited').size()
benchmark = max(dataset.groupby('Exited').size()[0], dataset.groupby('Exited').size()[0])/(dataset.groupby('Exited').size()[0] + dataset.groupby('Exited').size()[1])

print(pd.DataFrame(y_test).groupby('Exited').size())
print('>>>   Percentage of clients who Exited is : {0:.2%}'.format(y.index.isin(list(y.index[(y == 1 )== True])).sum()/(y.index.isin(list(y.index[(y == 1 )== True] )).sum()+y.index.isin(list(y.index[(y == 1 )== False])).sum())))
print(">>>   Base line accuracy of prediction is {0:.2%}".format(benchmark))        


#### 2 HISTOGRAMS AND SCATTER PLOT #############################################
# interesting columns for histograms3,5,6,7,8,11
plt.hist(X['CreditScore'])
plt.hist(X['Age']) # possible dyscretization opporutity to split into young 0-30, 30-40, 40+
plt.hist(X['Tenure'])
plt.hist(X['Balance']) # split into those with 0, 0-100k, 100k-125k - 125k+
plt.hist(X['NumOfProducts']) # mayby it should be encoded into 1, 2 and other as binnary data
plt.hist(X['EstimatedSalary'])

from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

#df.plot(kind='box', subplots=True, layout=(3, 4), sharex = False, sharey=False)
#plt().show()

### 3 CORRELATION #############################################################

pd.options.display.max_columns = 12
pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth',30)
#pd.reset_option('^display.', silent=True)
X.corr()


###########  MODELING DATASET  ################################################
###############################################################################


####  1. MULTIPLE LINEAR REGRESSION ###########################################

X_reg = X.copy()
X_train_reg = X_train.copy()
X_test_reg = X_test.copy()
y_train_reg = y_train.copy()


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)

# Predict the Test set results
y_pred = regressor.predict(X_test_reg)

plt.hist(y_pred)

# Assiging predicted values into decision
for q in range(len(y_pred)):
    if y_pred[q]<0.48: #(max(y_pred) - min(y_pred))/2:
        y_pred[q]=0
    else: y_pred[q] = 1
    
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))       
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format((cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark)) 
print(">>>   Current best prediction is {0:.2%}".format(benchmark))        


####  2. MULTIPLE LINEAR REGRESSION WITH BACKWARD ELIMINATION #################
import statsmodels.formula.api as sm

X_opt = X.copy()
X_train_opt = X_train.copy()
X_test_opt = X_test.copy()
y_train_opt = y_train.copy()

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['HasCrCard'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['EstimatedSalary'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
X_opt.drop(['Tenure'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['Germany'], axis=1, inplace=True)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt.drop(['NumOfProducts'], axis=1, inplace=True)  # Although below the adopted threshold worth to consider deleting this column also
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

regressor = LinearRegression()
X_train_opt.drop(['HasCrCard', 'EstimatedSalary', 'Tenure', 'Germany', 'NumOfProducts'], axis=1, inplace=True)
regressor.fit(X_train_opt, y_train_opt)
X_test_opt.drop([ 'HasCrCard', 'EstimatedSalary', 'Tenure', 'Germany', 'NumOfProducts'], axis=1, inplace=True)
y_pred = regressor.predict(X_test_opt)

plt.hist(y_pred)

# Assiging predicted values into decision
for q in range(len(y_pred)):
    if y_pred[q]<0.5: #(max(y_pred) - min(y_pred))/2:
        y_pred[q]=0
    else: y_pred[q] = 1
    
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))       
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format((cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark)) 
print(">>>   Current best prediction is {0:.2%}".format(benchmark))        
  

##### 2 PCA & lOGISTIC REGRESSION #############################################

X_pca = X.copy()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
y_train_pca = y_train.copy()


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)
explained_variance = pca.explained_variance_ratio_


### 5 Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=10)
classifier.fit(X_train_pca, y_train_pca)

# Predict the Test set results
y_pred = classifier.predict(X_test_pca)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))       
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format((cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark)) 
print(">>>   Current best prediction is {0:.2%}".format(benchmark))        
 

# Visualising the Training set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train_pca, y_train_pca
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test_pca, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


#### 3 SINGLE LAYER NEURAL NETWORL - PERCEPTRON ###############################

X_ann = X.copy()
X_train_ann = X_train.copy()
X_test_ann = X_test.copy()
y_train_ann = y_train.copy()


class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(4)
        self.synaptic_weights = 2 * np.random.random((12,1)) -1
        
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1-x)
    
    def train(self, training_inputs, training_outputs, training_iterations):
        
        for interation in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustment = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            #self.synaptic_weights = self.synaptic_weights + adjustment
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
    y_pred.append( neural_network.think(X_test_ann[i]))

for q in range(len(y_pred)):
    if y_pred[q]<0.50:
        y_pred[q]=0
    else: y_pred[q] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])

benchmark = max(benchmark, (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1]))       
print(cm)
print('>>>   Accuracy of this model is : {0:.2%}'.format((cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])))
print(">>>   Best accuracy of prediction was {0:.2%}".format(benchmark)) 
print(">>>   Current best prediction is {0:.2%}".format(benchmark))        

