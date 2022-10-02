import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pickle


dataset = pd.read_csv('urldata.csv')

dataset = dataset.sample(frac=1).reset_index(drop=True)

# print(dataset.head())
# print(dataset.status.value_counts())
# 5715 entries each for legitimate and phishing

# print(dataset.columns)
# print(dataset.describe())

# Plotting the data distribution
# dataset.hist(bins=50, figsize=(15, 15))
# plt.show()

# Correlation heatmap
# plt.figure(figsize=(15, 13))
# sns.heatmap(dataset.corr())
# plt.show()

# checking the data for null or missing values
# print(dataset.isnull().sum())

y = dataset.Label
X = dataset.iloc[:, 1:-1]

X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.2, random_state=12)

# creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

# function to call for storing the results
def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

# decision tree
tree = DecisionTreeClassifier(max_depth=5)

tree.fit(X_train, y_train)

y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

# computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train, y_train_tree)
acc_test_tree = accuracy_score(y_test, y_test_tree)

# print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
# print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

# checking the feature improtance in the model
# plt.figure(figsize=(9, 7))
# n_features = X_train.shape[1]
# plt.barh(range(n_features), tree.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), X_train.columns)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.show()

storeResults('Decision Tree', acc_train_tree, acc_test_tree)

# random forest
forest = RandomForestClassifier(max_depth=5)

forest.fit(X_train, y_train)

y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)

acc_train_forest = accuracy_score(y_train, y_train_forest)
acc_test_forest = accuracy_score(y_test, y_test_forest)

# print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
# print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

# plt.figure(figsize=(9,7))
# n_features = X_train.shape[1]
# plt.barh(range(n_features), forest.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), X_train.columns)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.show()

storeResults('Random Forest', acc_train_forest, acc_test_forest)

# Multilayer Perceptron
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100, 100, 100]))

mlp.fit(X_train, y_train)

y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)

acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

# print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
# print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))

storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

# XGBoost
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)

xgb.fit(X_train, y_train)

y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

acc_train_xgb = accuracy_score(y_train, y_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_test_xgb)

# print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
# print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

# support vector machine
svm = SVC(kernel='linear', C=1.0, random_state=12)

svm.fit(X_train, y_train)

y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)

acc_train_svm = accuracy_score(y_train, y_train_svm)
acc_test_svm = accuracy_score(y_test, y_test_svm)

# print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
# print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))

storeResults('SVM', acc_train_svm, acc_test_svm)

# creating dataframe to display accuracy from models
results = pd.DataFrame({
    'ML Model': ML_Model,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
print(results)

results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

pickle.dump(xgb, open("XGBoostClassifier.pickle.dat", "wb"))