import pandas as pd 
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

split = 0.66*42000

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
error_train = []
m_values = list(range(500,split,500))

for m in m_values

	nn = MLPClassifier(solver='lbfgs' , alpha = 1 , activation='logistic' , hidden_layer_sizes=(100,))
	nn.fit( X_train[ :m , : ] , y_train[ :m , : ] )

	y_pred  = nn.predict(X_cv)
	error_cv.append( sum(y_pred != y_cv)/len(y_pred) )

	y_pred = nn.predict(X_train[ :m , : ])
	error_train.append( sum(y_pred != y_train)/len(y_pred) )




plt.plot(m_values, error_cv, 'ro' , m_values , error_train , 'bs')
plt.axis([0, split, 0, 1])
plt.show()


