
import numpy as np
import pandas as pd
import random 

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

y = dataset.iloc[:, 13].values
X = dataset.iloc[:, :-1].values

# Deleting randomly some data
i=0
j=0
df = pd.DataFrame(X)
for column in df:
    j=0
    print('Columna: %d' %i)
    for row in df.iterrows():
        if i*j%random.randint(1000,2000)==0:
            X[j,i] = ''
        j+=1
    i+=1

# Here we can perform some data preprocessing 

 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

