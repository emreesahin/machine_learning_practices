import numpy as np 
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score , confusion_matrix



X = np.sort(5 * np.random.rand(40,1), axis= 0 ) # uniform
y = np.sin(X).ravel() # Target


# add noise 
y [::5] += 1 * (0.5 - np.random.rand(8))

T = np.linspace(0, 5, 500)[:, np.newaxis]



for i, weight in enumerate(['uniform', 'distance']):
    knn = KNeighborsRegressor(n_neighbors= 5, weights= weight)
    y_pred = knn.fit(X,y).predict(T)
    plt.subplot(2,1, i + 1)
    plt.scatter(X, y, color='green', label='data')
    plt.plot(T, y_pred, color='blue', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title('KNN Regressor weights = {}'.format(weight))
   
plt.tight_layout()
plt.show()





# plt.plot(X,y)
# plt.scatter(X,y)
# plt.show() 


