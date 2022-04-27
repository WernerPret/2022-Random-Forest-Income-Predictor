#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

# import CSV + add delimeter => clean spaces in strings
income_data = pd.read_csv("income.csv", header=0, delimiter=", ")

# remove question mark rows 
for column in income_data.columns:
  income_data = income_data[income_data[column] != "?"]

# check the first record
# print(income_data.iloc[0])

def one_hot_enc(df, cols):
  cols_list = []
  for column in cols:
    onehot = OneHotEncoder()
    encoded = onehot.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded.toarray())
    encoded_df = encoded_df.dropna() 

    for i in range(len(encoded_df.columns)):
      new_name = f"{column}{i}"
      encoded_df = encoded_df.rename(columns={i: new_name})
      cols_list.append(new_name)

    df = df.join(encoded_df)
    df.drop([column], axis="columns", inplace=True)
    df = df.dropna()
  return df, cols_list

# encode features
cols_to_encode = ["occupation", "education", "workclass", "marital-status", "relationship", "race", "sex", "native-country"]
income_data, enc_features = one_hot_enc(income_data, cols_to_encode)

# isolate labels & features
labels = income_data["income"]
labels = labels.apply(lambda x: 0 if x=="<=50K" else 1)
features = income_data.loc[:, income_data.columns!="income"]

# Select important features
test_forest = RandomForestClassifier(n_estimators = 100)

test_forest.fit(features, labels)
sel = SelectFromModel(test_forest)
sel.fit(features, labels)

importance = sel.get_support()
importance_data = sel.estimator.feature_importances_
results = [f"{features.columns[i]} : {importance_data[i]}" for i in range(len(importance)) if importance_data[i] >= 0.1]
features_final = income_data[[features.columns[i] for i in range(len(importance)) if importance_data[i] >= 0.1]]
# pd.Series(sel.estimator_.feature_importances_.ravel()).hist()

# Normalise data
scaler = StandardScaler()
features_final = scaler.fit_transform(features_final)

# Splite test and train data
x_train, x_test, y_train, y_test = train_test_split(features_final, labels)

# create random forest
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

# Score model
# Note, remember to make labels numeric, or Precision, Recall, F1 scores will not work
print("Score Sheet:")
print(f"Score Model on Data: {round(tree.score(x_test, y_test), 2)}")
print(f"Accuracy Score: {round(accuracy_score(y_test, y_pred), 2)}")
print(f"Precision Score: {round(precision_score(y_test, y_pred), 2)}")
print(f"Recall Score: {round(recall_score(y_test, y_pred), 2)}")
print(f"F1 Score: {round(f1_score(y_test, y_pred), 2)}")

