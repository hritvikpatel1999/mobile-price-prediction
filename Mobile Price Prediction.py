#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv(r'C:\Python\Datasets\battery_train.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[8]:


#df.apply(pd.Series.value_counts)


# In[12]:


df.isnull().sum()


# In[15]:


df.describe()


# FEATURE SCALING

# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df)
scaled_df.head()
#print(scaled_df.shape)


# In[47]:


#from sklearn.ensemble import ExtraTreesClassifier
#model = ExtraTreesClassifier()
#model.fit(scaled_df, y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
#feat_importances = pd.Series(model.feature_importances_, index=scaled_df.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
import seaborn as sns
corrmat = df.corr()
corrmat.sort_values(by = 'price_range', ascending=False)[:10]


# In[50]:


import seaborn as sns
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True)


# In[71]:


X = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)


# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[73]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
#print('Precision:')
#print(precision_score(y_test, y_pred))
#print('Recall:')
#print(recall_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# In[100]:


plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(confusion_matrix(y_test, y_pred),annot=True)


# In[75]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Dummy Accuracy: ')
print(accuracy_score(y_train, clf.predict(X_train) ))
print(accuracy_score(y_test, y_pred))


# As dummy classifier has very low accuracy, our RandomForestClassifier gives a decent performance

# In[76]:


#The whole dataset:
X1 = np.array(df.iloc[:, 0:20])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))


# Using a few features gave a better accuracy on the test set rather than the whole dataset

# In[77]:


X_new = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w']])
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=1234)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))


# In[87]:


lala = df.columns
scaled_df.columns = lala
scaled_df.head()


# In[89]:


#X3 = scaled_df.iloc[:, 0:20]
X_bleh = np.array(scaled_df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])

X_train, X_test, y_train, y_test = train_test_split(X_bleh, y, test_size=0.2, random_state=1234)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# 1. Class 0 is mostly correctly classified.
# 2. Class 1 is mostly incorrectly classified as class 0, 2
# 3. Class 2 is mostly incorrectly classified as class 1, 3
# 4. Class 3 is mostly incorrectly classified as class 2.

# In[93]:


X_logreg = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_logreg, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
#print('Precision:')
#print(precision_score(y_test, y_pred))
#print('Recall:')
#print(recall_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# LogReg sucks on this dataset

# In[96]:


X_svc = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_svc, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
#print('Precision:')
#print(precision_score(y_test, y_pred))
#print('Recall:')
#print(recall_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Gotta find out why svc is going to the doldrums -> Total Overfit

# In[98]:


X_dt = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dt, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy Score:')
print(accuracy_score(y_train, clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
#print('Precision:')
#print(precision_score(y_test, y_pred))
#print('Recall:')
#print(recall_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Probably overfitting, class 3 kaafi correctly predicted

# In[108]:


X_linreg = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_linreg, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred.shape


# In[111]:


X_knn = np.array(df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'sc_w', 'pc', 'three_g', 'sc_h']])
y = np.array(df['price_range'])
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_knn, y, test_size=0.2, random_state=1234)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

clf = KNeighborsRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf.score(X_test, y_test)


# In[ ]:




