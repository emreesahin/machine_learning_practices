from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


#Dataset 
oli = fetch_olivetti_faces()

X = oli.data
y = oli.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 

rf_clf = RandomForestClassifier(n_estimators= 100, random_state= 42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)


# Metrics 
accuracy = accuracy_score(y_test, y_pred)
print('Acc: ', accuracy)



# # Gorsellestirme 

# plt.figure()
# for i in range(2):
#     plt.subplot(1, 2, i+1)
#     plt.imshow(oli.images[i], cmap='gray')
#     plt.axis('off')
# plt.show()