# -*- coding: utf-8 -*-
"""Mondavi CART.ipynb


   
"""

import pandas as pd


#importing tess data
df=pd.read_excel('##')

)
#I have removed the dataset for privacy. The data contained 10 year historical ticketing data for a performing arts center with variables such as
#paid amount, performance date, seating type, section, row, subscription information, performance name

df2.head()

df.isna().sum()

# drop rows where 'paid_amt' is less than 5 
df = df[df['paid_amt'] >= 5]

#extractig seasons 

# Convert perf_date to a datetime object
df['perf_date'] = pd.to_datetime(df['perf_date'])

# Define a function to map months to seasons
def get_season(month, day):
    if (month >= 3) and (month <= 5):
        return 'Spring'
    elif (month >= 6) and (month <= 8):
        return 'Summer'
    elif (month >= 9) and (month <= 11):
        return 'Fall'
    else:
        return 'Winter'

# Extract the month and day of the year from perf_date
df['month'] = df['perf_date'].dt.month
df['day'] = df['perf_date'].dt.day

# Map the month and day to a season using the get_season function
df['season'] = df.apply(lambda row: get_season(row['month'], row['day']), axis=1)

df

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

data = df[["paid_amt", "price_zone", "row", "price_type", "season", "pkg_name"]].copy()

data

print(data.dtypes)

from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical columns
categorical_cols = ['price_zone', 'row', 'price_type', 'season', 'pkg_name']
encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
X = pd.concat([data.drop(categorical_cols, axis=1), encoded], axis=1)

Y = data['paid_amt']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train decision tree model
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)

# Perform grid search cross-validation to find the best hyperparameters
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': range(1, 10)}
grid_search = GridSearchCV(tree_model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found during cross-validation
print(grid_search.best_params_)

import numpy as np

# Define function for pruning decision tree
def prune_tree(clf, X_test, y_test):
    alpha_values = np.linspace(0, 0.01, num=100)
    best_score = float('-inf')
    best_tree = None
    for alpha in alpha_values:
        tree = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        tree = tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_tree = tree
    return best_tree

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
pruned_tree = prune_tree(tree_model, X_test, y_test)

# Print the best hyperparameters and score found during pruning
print("Best alpha: ", pruned_tree.ccp_alpha)
print("Best score: ", pruned_tree.score(X_test, y_test))

# Train decision tree model with best hyperparameters
tree_model = DecisionTreeRegressor(max_depth=9)
tree_model.fit(X_train, y_train)

from sklearn.tree import export_graphviz
import graphviz

# Train decision tree model with best hyperparameters
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Prune decision tree
pruned_tree = DecisionTreeRegressor(random_state=42, ccp_alpha=0.009797979797979799)
pruned_tree.fit(X_train, y_train)

# Export decision tree to DOT file
dot_data = export_graphviz(pruned_tree, out_file=None, 
                           feature_names=X_train.columns.values, 
                           filled=True, rounded=True, 
                           special_characters=True)
graph = graphviz.Source(dot_data)

# Save decision tree as PNG image
graph.format = 'png'
graph.render('decision_tree')

# Display decision tree in notebook
graph

# Print decision tree
from sklearn.tree import export_text
tree_rules = export_text(tree_model, feature_names=X.columns.tolist())
print(tree_rules)

from sklearn.metrics import mean_squared_error

# Generate predictions for test set using pruned decision tree
y_pred = pruned_tree.predict(X_test)

# Evaluate performance of pruned decision tree on test set
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error on test set: {:.4f}".format(mse))
