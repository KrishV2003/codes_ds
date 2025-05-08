import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
titanic = pd.read_csv('Titanic-Dataset.csv')
titanic.head()
titanic.isnull().sum()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median()).astype(int)
titanic.drop('Cabin', axis=1, inplace=True)
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
titanic.isnull().sum()
titanic.info()

plt.figure(figsize=(8, 6))
sns.histplot(titanic['Fare'], bins=30, kde=True, color='blue')
plt.title('Distribution of Titanic Passengers’ Fares')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.title('Survival Count by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# 7. Box Plot: Age Distribution by Passenger Class
plt.figure()
sns.boxplot(x="Pclass", y="Age", data=titanic)
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.tight_layout()
plt.show()
# 8. Violin Plot: Fare Distribution by Class
plt.figure()
sns.violinplot(x="Pclass", y="Fare", data=titanic)
plt.title("Fare Distribution by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

plt.figure()
sns.swarmplot(x="Pclass", y="Age", data=titanic.sample(100), size=3)
plt.title("Age Distribution by Class (Swarm Plot)")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.tight_layout()
plt.show()
# 11. Heatmap: Correlation of Numeric Features
corr = titanic.corr(numeric_only=True)
plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap of Numeric Features")
plt.tight_layout()
plt.show()