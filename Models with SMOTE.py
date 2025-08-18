#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('C:/Users/Hp/Desktop/fyp MS/pdm/pdm.csv')
del data['Product ID']
del data['Type']
data.head(5)


# In[3]:


data.shape


# In[4]:


features = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = data[features]
y = data[label]


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from numpy.random import randn
from matplotlib import pyplot


# In[6]:


print (y)


# In[7]:



data = data.astype({'Air temperature [K]':'int64'})
data = data.astype({'Torque [Nm]':'int64'})
data = data.astype({'Process temperature [K]':'int64'})


# In[8]:


import pandas as pd
from imblearn.over_sampling import SMOTE

features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = data[features]
y = data[label]

# Import train_test_split to split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Combine X_train_resampled and y_train_resampled into a single DataFrame
resampled_data = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.DataFrame(y_train_resampled, columns=['Machine failure'])], axis=1)

# Check the class distribution before and after SMOTE
print("Class Distribution Before SMOTE:")
print(y_train.value_counts().to_frame())

print("\nClass Distribution After SMOTE:")
print(resampled_data['Machine failure'].value_counts().to_frame())


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets (adjust the test_size as needed)
# Assuming your dataset has a 'Machine failure' column as the target (label) and the rest are features
features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = resampled_data[features].values
y = resampled_data[label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier on the training data
dt=dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[10]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Set up a custom color palette
color_palette = plt.cm.get_cmap('viridis_r', len(thresholds))

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o', color='royalblue', label='DTC21', linewidth=2)
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('DTC2 Precision-Recall Curve', fontsize=16)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Use custom markers and color palette
for i, t in enumerate(thresholds):
    plt.scatter(recall[i], precision[i], c=[color_palette(i)], s=80, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# # SVM

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
# Split the data into training and testing sets
# Assuming your dataset has a 'Machine failure' column as the target (label) and the rest are features
features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = resampled_data[features].values
y = resampled_data[label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)


# In[21]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Set up a custom color palette
color_palette = plt.cm.get_cmap('viridis_r', len(thresholds))

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o', color='royalblue', label='DTC21', linewidth=2)
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('DTC2 Precision-Recall Curve', fontsize=16)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Use custom markers and color palette
for i, t in enumerate(thresholds):
    plt.scatter(recall[i], precision[i], c=[color_palette(i)], s=80, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# # KNN

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=500)
features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = resampled_data[features]
y = resampled_data[label]
# Create and train the KNN classifier
k = 6 # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[27]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# Set up a custom color palette
color_palette = plt.cm.get_cmap('viridis_r', len(thresholds))

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o', color='royalblue', label='DTC21', linewidth=2)
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('DTC2 Precision-Recall Curve', fontsize=16)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Use custom markers and color palette
for i, t in enumerate(thresholds):
    plt.scatter(recall[i], precision[i], c=[color_palette(i)], s=80, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# In[ ]:





# In[ ]:




