# Tom White Machine learning algorithm
# 21st Jan 2025

import pandas as pd
import matplotlib.pyplot as plt # used with Sklearn to build graphical comparison models.
import seaborn as sns # Additional libray built on matplot lib to build statisical graphs
import time #For Execution timings
    

# The sklearn library is license free software used to perform a supervised learning algorith using two models - Logistic Regression &  Decision Tree Classifier.
# More inforation Sklearn is https://scikit-learn.org/stable/

from sklearn.model_selection import train_test_split #Used for inputing training data

from sklearn.linear_model import LogisticRegression #Logistic Regression Model instance 
from sklearn.tree import DecisionTreeClassifier #Decission Tree Classier Instance

#analysis metrics used to compare performance of each model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

# Read data source from downloaded file related to obsetity levels based on eating habits and physical condition.
# Data set location https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
file_path = "ObesityDataSet_raw_and_data_sinthetic.csv" 
data = pd.read_csv(file_path)

PPtimer = time.time()

# Prepare the data.  The column labelled "NObeyesdad" represents the actual result, as is the target we want the model to achieve
X = data.drop('NObeyesdad', axis=1)  # Where axis 0= row, axis 1= column - we need to remove this column from input data
y = data['NObeyesdad']  # Target variable

# The data is required to be pre-processed before the machine learning algorithm is able to processs the dat.
# This requires 1) Convert categorical variables to numerical values. uses https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html function
# Then simplify data to remove repeating data repetition and replace with a categorical code.  See https://hyperskill.org/learn/step/32241
X = pd.get_dummies(X)
y = pd.Categorical(y).codes

# Identify unique classes in the target variable
unique_classes = pd.Categorical(data['NObeyesdad']).categories
print("Unique classes in target variable:", unique_classes) #debug

# Split data into training and testing sets
# Why 42 - Read Hitchhikers guide to universe and speak with Douglas Adams - it is the preferred value for randomness in machine learning.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
PPtimer = (time.time() - PPtimer)
PPtimer = PPtimer * 1000
print ("\n Pre-Processing Execution timer in millsecs ", PPtimer) #debug output


Logtimer = time.time()
# Using Logistic Regression and Decision Tree Classifier for demonstration
# Logistic Regression Model with different solver and increased max_iter
log_model = LogisticRegression(solver='liblinear', max_iter=3000)  # Changed solver and increased max_iter
log_model.fit(X_train, y_train)
log_y_pred_class = log_model.predict(X_test)

Logtimer = (time.time() - Logtimer)
Logtimer = Logtimer * 1000
print ("\n Logistic Regression Execution timer in millisecs  ", Logtimer) #debug output

DTtimer = time.time()
# Decision Tree Classifier Model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_y_pred_class = tree_model.predict(X_test)

DTtimer = (time.time() - DTtimer)

DTtimer = DTtimer * 1000
print ("\nDecision Tree  Execution timer in millsecs  ", DTtimer) #debug output



# Confusion Matrices
log_conf_matrix = confusion_matrix(y_test, log_y_pred_class)
tree_conf_matrix = confusion_matrix(y_test, tree_y_pred_class)

# Visualizing Confusion Matrix using Heatmap
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(pd.DataFrame(log_conf_matrix, index=unique_classes, columns=unique_classes), annot=True, fmt='d', cmap='YlGnBu')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(pd.DataFrame(tree_conf_matrix, index=unique_classes, columns=unique_classes), annot=True, fmt='d', cmap='YlGnBu')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')

# Additional Metrics
log_metrics = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, log_y_pred_class),
    'Precision': precision_score(y_test, log_y_pred_class, average='weighted', zero_division=0),
    'F1 Score': f1_score(y_test, log_y_pred_class, average='weighted', zero_division=0)
}

tree_metrics = {
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, tree_y_pred_class),
    'Precision': precision_score(y_test, tree_y_pred_class, average='weighted', zero_division=0),
    'F1 Score': f1_score(y_test, tree_y_pred_class, average='weighted', zero_division=0)
}

print ("\n Logistic Regression scores are: \n", log_metrics) #debug output
print ("\n Decision Tree scores are: \n", tree_metrics) #debug output

#Now build nice graphs
metrics_df = pd.DataFrame([log_metrics, tree_metrics])

# Bar Chart for Metrics
metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6), ylim=(0, 1), colormap='viridis')
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_metrics.png')

print("Graphs have been saved as 'confusion_matrices.png' and 'model_metrics.png'")
