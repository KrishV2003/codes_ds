import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
**Dataset:** Titanic Dataset  
**Source:** https://www.kaggle.com/datasets/yasserh/titanic-dataset
df = pd.read_csv('Titanic-Dataset.csv')
df.head(11)
df.isnull()
df.isnull().sum()
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
df.drop('Cabin', axis=1, inplace=True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
df.ndim
df.info()
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df.info()
df.dtypes
df["Sex"] = df["Sex"].replace({"male": 1,"female": 0}).astype(int)
df
df = pd.get_dummies(df, columns=['Embarked','Pclass'], drop_first=True)
df
df[['Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']] = df[['Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']].astype(int)
df
scaler = MinMaxScaler()
for col in ['Age','Fare','SibSp','Parch']:
    df[col] = scaler.fit_transform(df[[col]])

# Verify they’re between 0 and 1
print(df[['Age','Fare','SibSp','Parch']].describe().loc[['min','max']])
df
df.describe(include='all')