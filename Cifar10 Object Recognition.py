#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import cifar10
import matplotlib.pyplot as plt


# In[2]:


cifar10.data_path = "data/CIFAR-10/"


# In[5]:


cifar10.maybe_download_and_extract()


# In[3]:


class_names = cifar10.load_class_names()
class_names


# In[4]:


images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


# In[5]:


fig=plt.figure(figsize=(8,8))
for i in range(64):
    ax=fig.add_subplot(8,8,i+1)
    ax.imshow(images_train[i],cmap=plt.cm.bone)
plt.show()


# In[6]:


x_train = images_train.reshape(images_train.shape[0],-1)
x_test = images_test.reshape(images_test.shape[0], -1)
y_train = cls_train
y_test = cls_test


# In[7]:


x_test.shape


# In[ ]:


pca=PCA()
pca.fit(x_train)


# In[10]:


pca = PCA(whiten = True)
pca.fit(x_train)
pca.components_.shape


# In[11]:


total = sum(pca.explained_variance_)
k = 0
current_variance = 0
while current_variance/total < 0.96:
    current_variance += pca.explained_variance_[k]
    k = k+1
    
k


# In[12]:


pca = PCA(n_components=k, whiten=True)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)


# In[28]:


svc = svm.SVC(C = 3.8)
svc.fit(x_train_pca, y_train)


# In[ ]:





# In[29]:


y_pred = svc.predict(x_test_pca)
y_pred_text = []
for val in y_pred:
    y_pred_text.append(class_names[val])
np.savetxt('predictions_cifar3.csv', y_pred_text, fmt = '%s')
y_pred_text


# In[30]:


accuracy_score(y_test, y_pred)

