import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
titanic = sns.load_dataset('titanic')
titanic.head()
titanic.info()
titanic.isnull().sum()
titanic['age'] = titanic['age'].fillna(titanic['age'].median()).astype(int)
titanic.drop('deck', axis=1, inplace=True)
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])
titanic.isnull().sum()

plt.figure(figsize=(8, 6))
sns.histplot(titanic['fare'], bins = 30, kde = True, color='skyblue')
plt.title('Distribution of Titanic Pasengers Fares')
plt.xlabel('Fare')
plt.ylabel('Number of Pasengers')
plt.grid('True')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival Count by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# Set Seaborn style
sns.set(style="whitegrid")

# c. Box Plot
plt.figure()
sns.boxplot(x="pclass", y="age", data=titanic)
plt.title("Age Distribution by Passenger Class")
plt.show()

# d. Violin Plot
plt.figure()
sns.violinplot(x="class", y="fare", data=titanic)
plt.title("Fare Distribution by Class")
plt.show()

# c. Swarm Plot
plt.figure()
sns.swarmplot(x="class", y="age", data=titanic.sample(100), size=3)
plt.title("Age Distribution by Class (Swarm Plot)")
plt.show()

# a. Heatmap
corr = titanic.corr(numeric_only=True)
plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap of Numeric Features")
plt.show()