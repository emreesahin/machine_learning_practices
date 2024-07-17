# Veri seti incelenmesi

from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
df['target'] = cancer.target

# Modelin Secilmesi 
# Model Train Part 
X = cancer.data # features 
y = cancer.target # target 

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # KNN Modeli ve Training
# knn = KNeighborsClassifier(n_neighbors=3) # Komsu sayisi belirle
# knn.fit(X_train , y_train) 


# # Sonuc Degerlendirme
# y_pred = knn.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('Acc Value: ', accuracy)
# conf_matrix = confusion_matrix(y_test , y_pred)
# print('Confusion Matrix: \n', conf_matrix)

# Hipermetre duzenlemesi
accuracy_values = []
k_values = []


for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

for i in range(len(k_values)):
    print(k_values[i] , accuracy_values[i])


plt.figure()
plt.plot(k_values, accuracy_values)
plt.title('K degerine gore dogruluk')
plt.xlabel('K degeri') # x eksenindeki deger
plt.ylabel('Dogruluk') # y eksenindeki deger
plt.xticks(k_values) # x ekseninde tam sayilar 
plt.grid(True) # Grid ekler
plt.show()


