# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/b1c361b6-e5b3-4c28-908c-068c3cd8917d)


## DESIGN STEPS

### Step 1: Import the necessary packages & modules

### Step 2:  Load and read the dataset

### Step 3:  Perform pre processing and clean the dataset

### Step 4:  Encode categorical value into numerical values using ordinal/label/one hot encoding

### Step 5:  Visualize the data using different plots in seaborn

### Step 6:  Normalize the values and split the values for x and y

### Step 7:  Build the deep learning model with appropriate layers and depth

### Step 8:  Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration

### Step 9:  Use prediction for some random inputs

## PROGRAM

```python

import pandas as pd
import pandas as pd
df = pd.read_csv("customers.csv")

import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

df.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as conf

df.isnull().sum()

df = df.drop('ID',axis=1)
df = df.drop('Var_1',axis=1)

df_cleaned = df.dropna(axis=0)

df_cleaned.isnull().sum()

categories_list=[['Male', 'Female'],['No', 'Yes'],
               ['No', 'Yes'],['Healthcare', 'Engineer',
               'Lawyer','Artist', 'Doctor','Homemaker',
               'Entertainment', 'Marketing', 'Executive'],
               ['Low', 'Average', 'High']]
enc = OrdinalEncoder(categories=categories_list)

df1 = df_cleaned.copy()

df1[['Gender','Ever_Married',
     'Graduated','Profession',
     'Spending_Score']] = enc.fit_transform(df1[['Gender',
     						'Ever_Married','Graduated',
                            'Profession','Spending_Score']])
df1
df1.dtypes

le = LabelEncoder()
df1['Segmentation'] = le.fit_transform(df1['Segmentation'])

df1.dtypes

corr = df1.corr()

sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="Greys",
            annot= True)

sns.distplot(df1['Age'])

scale = MinMaxScaler()
scale.fit(df1[["Age"]]) # Fetching Age column alone
df1[["Age"]] = scale.transform(df1[["Age"]])

df1.describe()

df1['Segmentation'].unique()

x = df1[['Gender','Ever_Married','Age','Graduated',
		 'Profession','Work_Experience','Spending_Score',
         'Family_Size']].values

y1 = df1[['Segmentation']].values

ohe = OneHotEncoder()
ohe.fit(y1)

y = ohe.transform(y1).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=50)

ai = Sequential([Dense(9,input_shape = [8]),
               Dense(19,activation="relu"),
               Dense(19,activation="relu"),
               Dense(19,activation="relu"),
               Dense(4,activation="softmax")])

ai.compile(optimizer='adam',
         loss='categorical_crossentropy',
         metrics=['accuracy'])

early_stop = EarlyStopping(
  monitor='val_loss',
  mode='max', 
  verbose=1, 
  patience=20)

ai.fit( x = x_train, y = y_train,
      epochs=500, batch_size=256,
      validation_data=(x_test,y_test),
      callbacks = [early_stop]
      )

import numpy as np
metrics = pd.DataFrame(ai.history.history)


metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_pred = np.argmax(ai.predict(x_test), axis=1)
x_pred.shape

y_truevalue = np.argmax(y_test,axis=1)
y_truevalue.shape

x_prediction = np.argmax(ai.predict(x_test[1:17,:]), axis=1)

print(x_prediction)

print(le.inverse_transform(x_prediction))
```

## Dataset Information

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/c7d1123e-0c53-42e7-987f-a3b0bbc8f1f9)

### Before cleaning:

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/bbc1dfc9-c050-4dc5-b960-d0c455c7c667)

### After cleaning:

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/a10acd1e-e365-4bd2-a827-039afc05ab23)

### Heatmap:

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/21f939ed-e1db-461f-b69f-aeb76748f069)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/d031639b-1ca0-4cd5-86b3-9af77c9df274)

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/d182ffd0-2111-4789-82a1-574e6071f127)


### Classification Report

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/1f09c30d-d09f-4d5f-af6c-2dd8d48abadc)

### Confusion Matrix

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/242c52ed-2832-4487-a1e4-a1bfcc759b9a)



### New Sample Data Prediction

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/fac2ac88-da1b-43b7-ad50-30d8f102974c)

![image](https://github.com/EASWAR17/nn-classification/assets/94154683/bd5f90c6-1000-464a-99d3-10a07a38a772)


## RESULT

A neural network classsification model is deveploed for the given dataset
