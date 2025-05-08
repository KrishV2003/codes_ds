import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
titanic.head()
titanic.isnull().sum()
titanic = titanic.dropna(subset = 'age')

# Fill missing categorical values
titanic['embraked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])
titanic.drop(columns='deck', inplace=True)
titanic.isnull().sum()

from matplotlib.patches import Patch
plt.figure(figsize=(8, 6))
plt.title('Age Distribution by Gender and Survival Status')
palette = {0: 'sandybrown', 1: 'mediumseagreen'}
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette=palette)
plt.legend(handles=[
    Patch(facecolor='sandybrown', label='No'),
    Patch(facecolor='mediumseagreen', label='Yes')
], title='Survived')
plt.tight_layout()
plt.show()

# Split ages by survival
ages_no  = titanic.loc[titanic['survived'] == 0, 'age']
ages_yes = titanic.loc[titanic['survived'] == 1, 'age']

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
