import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.head()

df.dtypes

df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Feature Distributions - Histograms', fontsize=16)
plt.show()

plt.figure(figsize=(10, 8))
sns.boxplot(data=df.drop(columns=['species']))
plt.title('Feature Distribution - Boxplots', fontsize=16)
plt.show()

for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'{feature.capitalize()} Distribution by Species')
    plt.show()
    
def remove_outliers_iqr(data, features):
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
    return data

numeric_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df_clean = remove_outliers_iqr(df, numeric_features)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Cleaned dataset shape (after removing outliers): {df_clean.shape}")

plt.figure(figsize=(10, 8))
sns.boxplot(data=df_clean.drop(columns=['species']))
plt.title('Boxplots After Outlier Removal', fontsize=16)
plt.show()
