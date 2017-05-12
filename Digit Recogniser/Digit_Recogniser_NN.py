
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

file = pd.read_csv( 'train.csv' , delimiter=',' , header=0)


# In[3]:

f = file.as_matrix()


# In[4]:

X = f[:,1:]


# In[5]:

y = f[:,0]


# In[6]:

from sklearn.neural_network import MLPClassifier


# In[7]:

for layer_size in [125, 150, 200]:
	for reg in [10, 20, 30]:
	
		nn = MLPClassifier(solver='lbfgs' , alpha = reg , activation='logistic' , hidden_layer_sizes=(layer_size,))


# In[8]:

		nn.fit(X,y)


# In[9]:

		test = pd.read_csv( 'C:/Users/pc/Downloads/test.csv' , delimiter=',' , header=0)
		t = test.as_matrix()


# In[11]:

		y_pred = nn.predict(t)


# In[12]:

		predDF = pd.DataFrame(data=y_pred,
                     index=list(range(1,28001))
                     )


# In[13]:

		predDF.to_csv('C:/Users/pc/Downloads/sample_NN_' +str(layer_size) +str(reg) +'.csv',
               header=['Label'])


# In[ ]:



