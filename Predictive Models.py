#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv('C:/Users/Hp/Desktop/fyp MS/pdm/pdm.csv')
del data['Product ID']
del data['Type']
data.head(5)


# In[4]:


data.shape


# In[5]:


features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = data[features]
y = data[label]


# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming you have reduced_features and label defined
features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

# Load your data and prepare X and y
X = data[features].values
y = data[label].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)


# In[7]:


import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/Hp/Desktop/fyp MS/pdm/pdm.csv')

# Define the new sampling interval (e.g., downsampling by every 2 UDI values)
downsampling_factor = 2  # You can change this to your desired downsampling factor

# Define the columns to downsample
columns_to_downsample =['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF','Machine failure']

# Perform downsampling by taking the mean over the specified interval
downsampled_data = data.groupby(data.index // downsampling_factor)[columns_to_downsample].mean()

# Save the downsampled data to a new CSV file
downsampled_data.to_csv('C:/Users/Hp/Desktop/fyp MS/pdm/pdm_downsampled.csv')


# In[8]:


downsampled_data.info()


# In[9]:


features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = downsampled_data[features]
y = downsampled_data[label]


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from numpy.random import randn
from matplotlib import pyplot


# In[11]:


print (y)


# In[12]:


downsampled_data['Machine failure'] = downsampled_data['Machine failure'].astype(int)


# In[13]:


downsampled_data.info()


# In[14]:


counts = downsampled_data['Machine failure'].value_counts()
print(counts)


# # Decision Tree Classifier

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Reduce features
reduced_features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = data[reduced_features].values
y = data[label].values

# Introduce irrelevant feature
X = np.hstack((X, np.random.random((X.shape[0], 1))))

# Introduce class imbalance by undersampling
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]
undersampled_indices = np.random.choice(class_0_indices, size=int(len(class_0_indices) * 0.8), replace=False)
undersampled_indices = np.concatenate((undersampled_indices, class_1_indices))
np.random.shuffle(undersampled_indices)

X = X[undersampled_indices]
y = y[undersampled_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier with limited depth
dt_classifier = DecisionTreeClassifier(max_depth=2, random_state=42)

# Train the Decision Tree classifier on the training data
dt = dt_classifier.fit(X_train, y_train)

# Introduce label noise
np.random.seed(42)
num_samples = y_test.shape[0]
num_noise_samples = int(num_samples * 0.3)  # Introduce noise to 30% of samples
noise_indices = np.random.choice(num_samples, size=num_noise_samples, replace=False)
y_test[noise_indices] = 1 - y_test[noise_indices]  # Flip labels for noise

# Make predictions on the noisy testing data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[16]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[17]:


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

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='DT1')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title(' DT1 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot
color = plt.cm.viridis_r(recall)  # You can choose any colormap you prefer
plt.scatter(recall, precision, c=color, cmap='viridis_r', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()




# In[24]:



import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
svm2 = confusion_matrix(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(svm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.ylabel('True Labels', fontsize=14)
output_file_path = 'confusion_matrix_plot.png'
plt.show()


# # SVM

# In[40]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you have already loaded your data into the 'data' variable

# Define features and label
features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = data[features].values
y = data[label].values

# Introduce some random noise to the labels
np.random.seed(42)
noise_indices = np.random.choice(len(y), size=int(len(y) * 0.1), replace=False)
y[noise_indices] = 1 - y[noise_indices]  # Flipping labels for a portion of the data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# In[41]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[42]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[43]:




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='SVM1')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title(' SVM1 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot
color = plt.cm.viridis_r(recall)  # You can choose any colormap you prefer
plt.scatter(recall, precision, c=color, cmap='viridis_r', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()




# In[44]:


import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
svm2 = confusion_matrix(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(svm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.ylabel('True Labels', fontsize=14)
output_file_path = 'confusion_matrix_plot.png'
plt.show()


# # KNN
# 

# In[46]:


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
X = data[features]
y = data[label]

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


# In[47]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[48]:



import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
svm2 = confusion_matrix(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(svm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.ylabel('True Labels', fontsize=14)
output_file_path = 'confusion_matrix_plot.png'
plt.show()


# In[49]:




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='KNN1')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title(' KNN1 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot
color = plt.cm.viridis_r(recall)  # You can choose any colormap you prefer
plt.scatter(recall, precision, c=color, cmap='viridis_r', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()




# In[ ]:




