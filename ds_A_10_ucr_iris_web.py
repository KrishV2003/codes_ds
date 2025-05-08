import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

df.head()

# 1. List down the features and their types
print("Features and their types:")
print(df.dtypes)

# 2. Create histograms for each feature
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Feature Distributions - Histograms', fontsize=16)
plt.show()

# 3. Create boxplot for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=df.drop(columns=['species']))
plt.title('Feature Distribution - Boxplots', fontsize=16)
plt.show()

# 4. Boxplot by species for each feature to check class-wise outliers
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'{feature.capitalize()} Distribution by Species')
    plt.show()
    
# 5. Outlier detection using IQR method
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

# 6. Apply outlier removal
numeric_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df_clean = remove_outliers_iqr(df, numeric_features)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Cleaned dataset shape (after removing outliers): {df_clean.shape}")

# 7. Boxplot after removing outliers
plt.figure(figsize=(10, 8))
sns.boxplot(data=df_clean.drop(columns=['species']))
plt.title('Boxplots After Outlier Removal', fontsize=16)
plt.show()

