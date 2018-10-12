
import numpy as np
import pandas as pd
import random 

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
y = dataset.iloc[:, 13].values
X = dataset.iloc[:, :-1].values
a = np.shape(X)

# Deleting randomly some data
i=0
for column in range(a[1]-1):
    j=0
    i+=1
    print('Columna: %d' %i)
    for row in range(a[0]-1):
        j+=1
        if i*j%random.randint(1000,2000)==0:
            X[j,i] = ''
   
df = pd.DataFrame(X)

# Here we can perform some data preprocessing 

 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

