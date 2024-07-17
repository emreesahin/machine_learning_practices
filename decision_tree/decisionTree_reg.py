from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np 

# Dataset 
diabet = load_diabetes()

X = diabet.data
y = diabet.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

# DT Regression 

tree_reg = DecisionTreeRegressor( random_state= 42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error' , mse)

rmse = np.sqrt(mse)
print('Root Mean Squared Error', rmse)