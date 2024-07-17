from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt

# Create a Dataset 
X = np.sort(5 * np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 0.5 * ( 0.5 - np.random.rand(16))

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.plot(X,y, c= 'red')
plt.scatter(X, y, c= 'red', label = "data")
plt.plot(X_test, y_pred_1, color = 'blue', label = 'Max Depth 2', linewidth = 2.2)
plt.plot(X_test, y_pred_2, color = 'green', label = 'Max Depth 5')
plt.xlabel = ('data')
plt.ylabel = ('target')
plt.legend()
plt.show()
# plt.scatter(X,y)
# plt.show()