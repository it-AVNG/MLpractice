from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
iris_dataset = load_iris()

print("keys of the dataset: {}".format(iris_dataset.keys()))
print("Feature to Identify: {}".format(iris_dataset['feature_names']))
print("Spieces to catagorize: {}".format(iris_dataset['target_names']))
print("shape of data: {}".format(iris_dataset['data'].shape))
print("last 5 data: \n{}".format(iris_dataset['data'][-5:]))

'''
We see that the array contains measurements for 150 different flowers. 
With 150 Rows and 4 collumns according to 4 Features:
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

 The meanings of the numbers are given by the iris['target_names'] array:
0 means setosa, 1 means versicolor, and 2 means virginica.
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state =0)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

'''
Before building a machine learning model 
it is often a good idea to inspect the data,
to see if the task is easily solvable without machine learning, 
or if the desired infor‐mation might not be contained in the data.

One of the best ways to inspect data is to visualize it. 
One way to do this is by using a scatter plot.
'''

#create dataframe using data in X_train
#label the columns using strngs in irisdataset.featurenames
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#create a scatter matrix from the dataframe, color by y_train

grr= pd.plotting.scatter_matrix(iris_dataframe, 
                                c=y_train, 
                                figsize=(15,15), 
                                marker='o', 
                                hist_kwds={'bins':20},
                                s=60, alpha=.5)

# build our first model using Knearest neighbor

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_prediction =  knn.predict(X_test)
print(f"our prediction to X_test: \n {y_prediction}")
print(f"predicting target names:\n {iris_dataset['target_names'][y_prediction]}")
#check if we are correct
score = np.mean(y_prediction == y_test)
print("our test score {:.2f}".format(score))

#we can also use score property of the model knn.score(X_test, y_test) to evaluate our model






