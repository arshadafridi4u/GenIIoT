#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


fake_data = pd.read_csv('C:/Users/Hp/Downloads/fake_data.csv')


# In[3]:


features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = fake_data[features]
y = fake_data[label]


# In[4]:


counts = fake_data['Machine failure'].value_counts()
print(counts)


# In[5]:


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
X = fake_data[features].values
y = fake_data[label].values

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


# In[6]:


import pandas as pd

# Load your data into a DataFrame named 'fake_data'
fake_data = pd.read_csv('C:/Users/Hp/Downloads/fake_data.csv')

major_count = 7722  # Adjust this count to match the minor class
minor_count = 278

# Select rows for the major class (Machine Failure = 0)
major_class_rows = fake_data[fake_data['Machine failure'] == 0].sample(n=major_count, replace=True)

# Select rows for the minor class (Machine Failure = 1)
minor_class_rows = fake_data[fake_data['Machine failure'] == 1].sample(n=minor_count, replace=False)

# Concatenate the selected rows
selected_data = pd.concat([major_class_rows, minor_class_rows])

print(selected_data)


# In[7]:


counts = selected_data['Machine failure'].value_counts()
print(counts)


# # DTC3

# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into training and testing sets (adjust the test_size as needed)
# Assuming your dataset has a 'Machine failure' column as the target (label) and the rest are features
features = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = selected_data[features].values
y = selected_data[label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier on the Atraining data
dt=dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[36]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[37]:



print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[38]:



import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.ylabel('True Labels', fontsize=14)
output_file_path = 'confusion_matrix_plot.png'
plt.show()


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='DTC3')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('DTC3 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot using 'plasma' colormap
color = plt.cm.plasma(recall)  # Using 'plasma' colormap for a different look
plt.scatter(recall, precision, c=color, cmap='plasma', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# # SVM

# In[22]:


counts = selected_data['Machine failure'].value_counts()
print(counts)


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
# Split the data into training and testing sets
# Assuming your dataset has a 'Machine failure' column as the target (label) and the rest are features
features = [ 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF', 'OSF', 'PWF']
label = ['Machine failure']

X = selected_data[features].values
y = selected_data[label].values

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


# In[44]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[48]:



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[51]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='SVM3')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('SVM3 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot using 'plasma' colormap
color = plt.cm.plasma(recall)  # Using 'plasma' colormap for a different look
plt.scatter(recall, precision, c=color, cmap='plasma', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# # KNN
# 

# In[53]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)
features = ['UDI','Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'RNF','OSF','PWF']

label = ['Machine failure']
X = selected_data[features]
y = selected_data[label]
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


# In[ ]:





# In[54]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate example data (you should replace this with your actual data)
y_test = np.random.randint(2, size=100)
y_pred = np.random.rand(100)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='o', color='royalblue', label='KNN3')
plt.fill_between(recall, precision, color='lightsteelblue', alpha=0.2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('KNN3 Precision-Recall Curve', fontsize=12)
plt.legend(fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().tick_params(axis='both', which='both', length=0)

# Add some color gradients to the plot using 'plasma' colormap
color = plt.cm.plasma(recall)  # Using 'plasma' colormap for a different look
plt.scatter(recall, precision, c=color, cmap='plasma', s=70, edgecolors='k')

# Add colorbar to indicate the threshold variations
cbar = plt.colorbar()
cbar.set_label('Threshold', fontsize=12)

plt.show()


# In[11]:



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


# In[ ]:




