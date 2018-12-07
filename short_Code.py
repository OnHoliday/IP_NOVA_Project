### CHURN MODELING - DATA SCIENCE PROJECT FOR INTRODUCTION TO PROGRAMMING #####
###############################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

### 4. HANDLING ERRORS  ######################################################


# excluding binary variables for boxplots
X_boxplot = X[[i for i in list(X.columns) if i not in ['HasCrCard', 'IsActiveMember']]]

# boxplots for outliers
plt.style.use('ggplot')
X_boxplot.plot(kind='box', subplots=True, layout=(2, 3), sharex=False, sharey=False, figsize=(20, 10))
plt.suptitle('Column Boxplots', size=25, fontweight='bold', color='grey')
plt.show()

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

sns.factorplot('Geography', 'Exited', hue='Gender', data=dataset, split=True)
plt.title('Exited vs Country by Gender', size=13, color='grey', fontweight='bold')
plt.xlabel('')
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

sns.set_style('darkgrid')
sns.heatmap(X.corr(), vmin=-1, vmax=1, cmap = "RdBu_r", center=0 ,annot=True, fmt=".0%")
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Correlation Heatmap', color='grey', size=25, fontweight='bold')


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
score_LlR = cm[0][0] * 0 + cm[1][1] * 590 + cm[1][0] * (-300) + cm[0][1] * (-250)
LlR_accuracy = (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])


##### 2 lOGISTIC REGRESSION #############################################

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=0)

LR.fit(X_train, y_train)

##### perfroming pca  #############################################

X_pca = X.copy()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
y_train_pca = y_train.copy()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)
explained_variance = pca.explained_variance_ratio_

### 2.1 Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

LR_PCA = LogisticRegression(random_state=00)
LR_PCA.fit(X_train_pca, y_train_pca)

##### 3 Decision Tree ###############################

from sklearn.tree import DecisionTreeClassifier  # importing the tree

dtree = DecisionTreeClassifier(random_state=0)  # setting up a tree

dtree.fit(X_train, y_train)  # using the train data to train the tree

#### random forest ###########################

from sklearn.ensemble import RandomForestClassifier  # omporting the RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300, random_state=0)  # setting up a RFC with 300 estimators (number of randomly distributed trees classifing the object; default is 100)

rfc.fit(X_train, y_train)  # using the train data to train the rfc

#####feature importance visualisation#######

feature_importances_rfc = pd.DataFrame(rfc.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
sns.barplot(x=feature_importances_rfc.index.values, y="importance", data=feature_importances_rfc)
plt.show()

#############################SUPPORT VECTOR MACHINES###########################

from sklearn.svm import SVC

svc = SVC(verbose=True, random_state=0)

svc.fit(X_train, y_train)

############################### Naive Bayes ##################################

from sklearn.naive_bayes import BernoulliNB

BernNB = BernoulliNB(binarize=True)

BernNB.fit(X_train, y_train)

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
score_NN = cm[0][0] * 0 + cm[1][1] * 590 + cm[1][0] * (-300) + cm[0][1] * (-250)
ann_accuracy = (cm[0][0] + cm[1][1] )/ (cm[0][0] + cm[1][1] + cm[1][0] + cm[0][1])




### Geting the accuracy scores for all the models ################

def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_test, y_test)
    train_accuracy = np.round(train_accuracy * 100, 2)
    return train_accuracy


'''Models with best training accuracy:'''

#setting up a DataFrame with the values
train_accuracy = pd.DataFrame({'Train_accuracy(%)': 
                                                    [train_accuracy(LR),
                                                     train_accuracy(LR_PCA),
                                                     train_accuracy(dtree), 
                                                     train_accuracy(rfc),
                                                     train_accuracy(BernNB), 
                                                     train_accuracy(svc)]}
                                                    )
    
train_accuracy.index = ['Logistic Regression',
                                        'Logistic Regression with PCA', 
                                        'Decision Tree', 
                                        'Random Forest', 
                                        'Naive Bayes',
                                        'Support Vector Classification']
#adding perceptron
train_accuracy.loc['Perceptron'] = np.round(ann_accuracy*100, 2)


acc = train_accuracy.reset_index()
acc = acc.sort_values(by=['Train_accuracy(%)'], ascending=True)

###### plot all the models accuracy #########
acc_1 = list(acc['index'])
acc_2 = list(acc['Train_accuracy(%)'])
sns.set_style('darkgrid')
y_pos = np.arange(len(acc_1))
plt.barh(y_pos, acc_2, align='center', alpha=0.5, color='#005F91')
plt.yticks(y_pos, acc_1, size=12)
plt.xticks(size=12)
plt.xlabel('Score', size=20)
plt.title('Accuracy of each model (%)', color='grey', fontweight='bold', size=25)
plt.xlabel('')
plt.show()




#### using our own metric of evaluation #############################

def ROI(estimator, x, y):
    yPred = estimator.predict(x)
    cm = confusion_matrix(y, yPred)
    ROI = cm[0][0] * 0 + cm[1][1] * 590 + cm[1][0] * (-300) + cm[0][1] * (-250)

    return ROI


##scoring each model ###
    
score_rf = ROI(rfc, X_test, y_test)
score_svc = ROI(svc, X_test, y_test)
score_nb = ROI(BernNB, X_test, y_test)
score_log = ROI(LR, X_test, y_test)
score_log_pca = ROI(LR_PCA, X_test, y_test)
score_tree = ROI(dtree, X_test, y_test)


mon_1 = ['Random Forrest', 'Support Vector Machine', 'Naive Bays', 'Logistic Regression','Logistic regression with PCA', 'Decision Tree', 'Perceptron']
mon_2 = [score_rf, score_svc, score_nb, score_log, score_log_pca, score_tree, score_NN]

# reversing
mon_1 = mon_1[::-1]
mon_2 = mon_2[::-1]

#### plotting the roi of each model in color   ########
import matplotlib.pyplot as plt;

plt.rcdefaults()

sns.set_style('darkgrid')
y_pos = np.arange(len(mon_1))
my_colors=['red','green','red', 'red', 'red', 'green', 'green']
plt.barh(y_pos, mon_2, align='center', alpha=0.5, color=my_colors)
plt.yticks(y_pos, mon_1, size=12)
plt.xticks(size=12)
plt.xlabel('')
plt.title('Monetary score of each model (€)', color='grey', fontweight='bold', size=25)
plt.show()







###### optional  accuracy after cross val for each model ######
'''Create a function that returns mean cross validation score for different models.'''


def x_val_score(model):
    from sklearn.model_selection import cross_val_score
    x_val_score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy').mean()
    x_val_score = np.round(x_val_score * 100, 2)
    return x_val_score


"""Let's perform k-fold (k=10) cross validation to find the classifier with the best cross validation accuracy."""
x_val_score = pd.DataFrame({'X_val_score(%)': 
                                            [x_val_score(LR), 
                                             x_val_score(LR_PCA),
                                             x_val_score(dtree),
                                             x_val_score(rfc),
                                             x_val_score(BernNB), 
                                             x_val_score(svc)]}
                                            )
    
x_val_score.index = ['Logistic Regression', 
                                 'Logistic Regression with PCA',
                                 'Deision Tree', 
                                 'Random Forest', 
                                 'Naive Bayes',
                                 'Support Vector Classification']


acc = x_val_score.reset_index()
acc = acc.sort_values(by=['X_val_score(%)'], ascending=True)
 
######   plot it  #######

acc_1 = list(acc['index'])
acc_2 = list(acc['X_val_score(%)'])

sns.set_style('darkgrid')
y_pos = np.arange(len(acc_1))
plt.barh(y_pos, acc_2, align='center', alpha=0.5)
plt.yticks(y_pos, acc_1, size=12)
plt.xticks(size=12)
plt.xlabel('')
plt.title('Accuracy of each model after CrossValidation (%)',color='grey', fontweight='bold', size=25)
plt.show()



######### cross validation with ROI as a metric ######

from sklearn.model_selection import cross_val_score
def cv(model):
    ROI_scores = cross_val_score(model, X_train, y_train, scoring = ROI, cv=12, n_jobs=-2).mean()
    return ROI_scores

cv_money = pd.DataFrame({'Monetary Score after Cross-Validation': 
                                        [cv(LR), 
                                         cv(LR_PCA), 
                                         cv(dtree), 
                                         cv(rfc), 
                                         cv(BernNB), 
                                         cv(svc)]}
                                        )
    
cv_money.index = ['Logistic Regression',
                              'Logistic regression with PCA', 
                              'Decision Tree', 
                              'Random Forrest', 
                              'Naive Bays', 
                              "Support Vector Machine"]

sortedcv_money = cv_money.sort_values(by='Monetary Score after Cross-Validation', ascending=True)
sortedcv_money['Monetary Score after Cross-Validation']*=3 #make results comparable with reluts on the test set by extrapolation
print('**Training Accuracy of the Classifiers:**')
print(sortedcv_money)

#####saving the results from svc and rfc for later comparison after grid search
a = sortedcv_money.iloc[5,0]
b = sortedcv_money.iloc[4,0]



######plotting the results######
acc = sortedcv_money.reset_index()
acc_1 = list(acc['index'])
acc_2 = list(acc['Monetary Score after Cross-Validation'])
y_pos = np.arange(len(acc_1))
sns.set_style('darkgrid')
my_colors=['red', 'red', 'red', 'red', 'green', 'green']
acc['Monetary Score after Cross-Validation'].plot(kind='barh', color=my_colors, alpha=0.5)
plt.yticks(y_pos, acc_1, size=12)
plt.xlabel('')
plt.xticks(size=12)
plt.title('Monetary Score of each model after Cross-Validation (€)',color='grey', fontweight='bold', size=25)
plt.show()


########GRID SEARCH FOR BETTER RESULTS########

''' !!!!!!!!!!! ATTENTION THIS TAKES SOME TIME !!!!!!!!!!!!!!!!!!'''

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

gridsvc = GridSearchCV(SVC(), param_grid, scoring = ROI,refit=True, verbose=2, n_jobs=-1, cv=12)

gridsvc.fit(X_train, y_train) #here the grid search happens 



#####GRID SEARCH FOR BETTER RESULTS######## 

''' !!!!!!!!!!! ATTENTION THIS TAKES EVEN MORE TIME !!!!!!!!!!!!!!!!!!'''

from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['gini', 'entropy'],'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [2, 5, 10, 50, 100],'min_samples_split': [2, 5, 10, 50, 100]}

gridrf = GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=0), param_grid, scoring = ROI, refit=True, verbose=2, n_jobs=-1, cv =12)

gridrf.fit(X_train, y_train) #here the grid search happens 



# here we also want to be able to combine the to modles so first we use the prdictions from the rfc then filter for when it predicted does  not leave (the curical thing) 
#then test the predictions of the svc and finally combining everything

####run together til end ####

grid_predictions=gridrf.predict(X_test)
grid_matrix_rf = confusion_matrix(y_test, grid_predictions)
rfc_grid_predictions = pd.Series(grid_predictions.copy())

x_t = X_test.copy()
y_t = y_test.copy()

x_t['Exited'] = y_t.values
x_t['Pred'] = rfc_grid_predictions.values

df_safe_rfc_test1 = x_t

df_second_test = df_safe_rfc_test1[df_safe_rfc_test1['Pred']==0]
df_second_test_ones = df_safe_rfc_test1[df_safe_rfc_test1['Pred']==1]

df_second_test_to_test = df_second_test.iloc[:,:-2]
y_second_test = df_second_test['Exited']

X_second_test = df_second_test_to_test


grid_predictions=gridsvc.predict(X_second_test)


df_second_test_to_test['Pred'] = grid_predictions

q = df_second_test_to_test['Pred']
rfc_grid_predictions = pd.DataFrame(rfc_grid_predictions)

x_t.drop(['Exited'], inplace =True, axis=1)

x_t['Pred2'] = df_second_test_to_test['Pred']
x_t['Pred3'] = x_t['Pred2'].fillna(0).astype('int') + x_t['Pred']

cm = confusion_matrix(y_test, x_t['Pred3'])
both_models = cm[0][0] * 0 + cm[1][1] * 590 + cm[1][0] * (-300) + cm[0][1] * (-250)
###end


####
#### plotting the comparison before and after grid search and the combined predictions ###########

c=0
ver=[a, b,c]
ver2=[ROI(gridrf, X_test, y_test), ROI(gridsvc, X_test, y_test), both_models]
labels=['RF', 'SVC', 'RF & SVC']
f1=['Random Forest', 'Support Vector Classification', 'Both models' ]
y_pos = np.arange(len(f1))
plt.bar(y_pos, ver2, alpha=0.5, color='limegreen', width=0.5)
plt.bar(y_pos, ver, alpha=0.5, color='darkgreen', width=0.5)
plt.xticks(y_pos, f1, size=20)
plt.ylabel('')
plt.yticks(size=15)
plt.title('Effect of model tuning using Grid Search (€)', color='grey', fontweight='bold', size=20)
plt.show()

#####
#####
