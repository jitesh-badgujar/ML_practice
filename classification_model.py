#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


df = pd.read_csv('/Users/jitesh/Downloads/Medicine_review.csv') 
print(df)


# In[4]:


df["Is_Response"]=pd.factorize(df["Is_Response"])[0]
df.head
df.dtypes


# In[5]:


from collections import Counter
Counter(df.Is_Response)


# In[6]:


from string import punctuation
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

for i, row in df.iterrows():
   
    example_sent =  df.at[i,'Description']

    print("Original: " + example_sent) 
 
    example_sent = example_sent.lower()

    print("After lowercase: " + example_sent) 

    example_sent = ''.join(c for c in example_sent if c not in punctuation)

    print("After punctuation removal: " + example_sent) 

    word_tokens = word_tokenize(example_sent) 

    print("word tokens: ") 
    print(word_tokens) 

    words = [w for w in word_tokens if w.isalpha()] 

    print("After removing non alphabets: ") 

    print(words) 

    stop_words = set(stopwords.words('english')) 
  
    filtered_sentence = [w for w in words if not w in stop_words] 
  
    print("After removing stop words: ") 

    print(filtered_sentence)

    final_sentence = ' '.join(filtered_sentence)

    print("Final Sentence") 

    print(final_sentence)
    
    df.at[i,'Description'] = final_sentence





# In[7]:


df.head()


# In[8]:


# Converting text into vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(df.Description)
x.toarray()


# In[13]:


x.shape


# In[14]:


df.dtypes


# In[17]:


df.to_csv('/Users/jitesh/Downloads/Medicine_review_processed_new.csv')


# In[18]:


y = df.values[:,1]
Y=y.astype('int')
X = x


# In[19]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100, stratify=Y)


# In[21]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[22]:


print("Predicted values:") 
print(y_pred) 


# In[23]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[24]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train, clf.predict(X_train)))


# In[50]:


# Fitting Random Forest classifier to the dataset 

from sklearn.ensemble import RandomForestClassifier

#for i in range(1,10):

     # create Classifier object 
classifier = RandomForestClassifier(n_estimators = 30, random_state = 10) 

    # fit the regressor with x and y data 
classifier = classifier.fit(X_train,y_train) 

y_pred_rf = classifier.predict(X_test)

from sklearn import metrics
a_test= metrics.accuracy_score(y_test, y_pred_rf)
print("Accuracy Test:",a_test)

from sklearn import metrics
a_train= metrics.accuracy_score(y_train, classifier.predict(X_train))
print("Accuracy Train:",a_train)
    
print("Differance:", abs(a_test-a_train))


# In[39]:




