
import pandas as pd 
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

split = 27720


# file is a pd.dataframe variable which contains the dataframe of train csv file
file = pd.read_csv( 'train.csv' , delimiter=',' , header=0)


# storing the file as an array
f = file.as_matrix()


# The training dataset in array form
X_train = f[:split, 1:]
y_train = f[:split, 0]

# The Cross validation set
X_cv = f[split: , 1:]
y_cv = f[split: , 0]



# Declaring error arrays
error_cv = []

import numpy as np
for layer_size in [200, 225, 250, 275]:
	for reg in [10, 25, 50 , 75 , 100]:

    	nn = MLPClassifier(solver='lbfgs' , alpha = reg , activation='logistic' , hidden_layer_sizes=(layer_size,))
    	nn.fit( X_train[ :m , ] , y_train[ :m , ] )

    	y_pred  = nn.predict(X_cv)
    	error_cv.append( sum( np.not_equal(y_pred ,y_cv ) )/len(y_pred) )


max_index = error_cv.index(np.amin(error_cv))
print("Layer_size: " ,min_index%4)
print("alpha: " , min_index/4)
