from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset 

iris = load_iris()
print(iris)

X = iris.data # feature
y = iris.target # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DT model and Train
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth= 5, random_state=42 ) #criterion = 'entropy' 
tree_clf.fit(X_train, y_train)

# DT evaluation test 
y_pred = tree_clf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Accuracy of DesicionTree with Iris Dataset:' , accuracy)
print('Confusion Matrix of DesicionTree with Iris Dataset \n' , conf_matrix)

# Visualization

plt.figure(figsize= (15,10))
plot_tree(tree_clf, filled = True, feature_names= iris.feature_names, class_names = list(iris.target_names))
plt.show()

# Feature Importance 
feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse= True)
for importance, feature_name in feature_importances_sorted:
    print(f'{feature_name}: {importance}')