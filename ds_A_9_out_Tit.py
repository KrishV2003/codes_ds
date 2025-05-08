import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

titanic = pd.read_csv('Titanic-Dataset.csv') 
titanic.head()
titanic.isnull().sum()
titanic = titanic.dropna(subset=['Age'])

# 4. Fill missing categorical values
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

titanic.drop(columns='Cabin', inplace=True)
titanic.isnull().sum()

plt.figure(figsize=(8, 6))
plt.title('Age Distribution by Gender and Survival Status')
palette = {0: 'sandybrown', 1: 'mediumseagreen'}
sns.boxplot(x='Sex', y='Age', hue='Survived', data=titanic, palette=palette)
plt.legend(handles=[
    Patch(facecolor='sandybrown', label='No'),
    Patch(facecolor='mediumseagreen', label='Yes')
], title='Survived')
plt.tight_layout()
plt.show()

ages_no  = titanic.loc[titanic['Survived'] == 0, 'Age']
ages_yes = titanic.loc[titanic['Survived'] == 1, 'Age']

plt.figure(figsize=(8, 6))
plt.hist([ages_no, ages_yes],
         bins=30,
         color=['sandybrown', 'mediumseagreen'],
         label=['No', 'Yes'],
         edgecolor='black',
         alpha=0.7)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.tight_layout()
plt.show()