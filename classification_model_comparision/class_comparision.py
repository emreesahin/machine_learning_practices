from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

# Create datasets
Xy_classification = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1, random_state=42)
Xy_classification = (Xy_classification[0] + 1.2 * np.random.uniform(size=Xy_classification[0].shape), Xy_classification[1])

datasets = [
    Xy_classification, 
    make_moons(noise=0.2, random_state=42),
    make_circles(noise=0.1, factor=0.3, random_state=42)
]

# Classifier names and instances
names = ['Nearest Neighbors', 'Linear SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes']
classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()
]

fig, axes = plt.subplots(len(datasets), len(classifiers) + 1, figsize=(18, 12))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Plot the dataset
    ax = axes[ds_cnt, 0]
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_title("Input Data")
    ax.set_xticks(())
    ax.set_yticks(())

    for clf_cnt, (name, clf) in enumerate(zip(names, classifiers)):
        ax = axes[ds_cnt, clf_cnt + 1]
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.7, ax=ax, eps=0.5)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        if ds_cnt == 0:
            ax.set_title(name)

        ax.text(X.max() - 0.3, X.min() + 0.3, f'{score:.2f}', size=15, horizontalalignment='right')

        ax.set_xticks(())
        ax.set_yticks(())

plt.tight_layout()
plt.show()
