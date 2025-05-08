import pandas as pd
import numpy as np
data = {
    'StudentID' : range(1, 11),
    'Gender' : ['M', 'F', 'Male', 'Female', 'M', 'F', np.nan, 'Female', 'male', 'F'],
    'MathScore' : [78, 82, 92, np.nan, 85, 90, 65, 45, 102, 110],
    'ReadingScore' : [72, 90, 87, 61, np.nan, 114, 76, 120, 48, 93],
    'WritingScore' : [70, 56, 89, 97, 73, np.nan, 118, 107, 49, 95]
}
df = pd.DataFrame(data)
df
df.isnull()
df.isnull().sum()
df['Gender'] = df['Gender'].str.strip().str.lower().map({
    'm': 'Male',
    'male': 'Male',
    'f': 'Female',
    'female': 'Female'
})
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
for col in['MathScore', 'ReadingScore', 'WritingScore']:
    df[col] = df[col].fillna(df[col].mean())
df.isnull().sum()
df.describe()
# Outliers Detect using IQR and capping the outliers
df_iqr = df.copy()
print("IQR Outlier bounds and clipping:")
for col in ['MathScore', 'ReadingScore', 'WritingScore']:
    Q1, Q3 = df_iqr[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    print(f" • {col}: IQR clip to [{lower:.2f}, {upper:.2f}]")
    df_iqr[col] = df_iqr[col].clip(lower, upper)
df_iqr[['MathScore', 'ReadingScore' ,'WritingScore']].agg(['min', 'max', 'mean', 'std'])
# Outliers Detect using Z-score and capping the outliers
df_z = df.copy()
print("Z-Score Outliers bounds and clipping:")
for col in ['MathScore', 'ReadingScore', 'WritingScore']:
    mean = df_z[col].mean()
    std = df_z[col].std()
    lower, upper = mean - 3 * std, mean + 3 * std
    print(f" • {col}: Z-score clip to [{lower:.2f}, {upper:.2f}]")
    df_z[col] = df_z[col].clip(lower, upper)
df_z[['MathScore', 'ReadingScore' ,'WritingScore']].agg(['min', 'max', 'mean', 'std'])
print("\nSkewness before and after log transformation (using IQR cleaned data):")
print(" MathScore skewness:", df_iqr['MathScore'].skew())
print("\nSkewness before and after log transformation (using Z-score cleaned data):")
print(" MathScore skewness:", df_z['MathScore'].skew())
# Apply log transformation
df_iqr['Log_MathScore'] = np.log(df_iqr['MathScore'])
df_z['Log_MathScore'] = np.log(df_z['MathScore'])
print("Skewness after log transform (using IQR cleaned data)")
print(" Log_MathScore skewness:", df_iqr['Log_MathScore'].skew(), "\n")

print("Skewness after log transform (using Z-score cleaned data)")
print(" Log_MathScore skewness:", df_z['Log_MathScore'].skew(), "\n")
print("Final IQR cleaned dataset with log-transformed MathScore:\n", df_iqr)
print("Final Z-score cleaned dataset with log-transformed MathScore:\n", df_z)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plot for df_iqr ---
plt.figure(figsize=(12, 5))
plt.suptitle('df_iqr: MathScore Before and After Log Transformation', fontsize=14)

plt.subplot(1, 2, 1)
sns.histplot(df_iqr['MathScore'], kde=True, color='skyblue')
plt.title('Before Log Transformation (IQR)')

plt.subplot(1, 2, 2)
sns.histplot(df_iqr['Log_MathScore'], kde=True, color='lightgreen')
plt.title('After Log Transformation (IQR)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# --- Plot for df_z ---
plt.figure(figsize=(12, 5))
plt.suptitle('df_z: MathScore Before and After Log Transformation', fontsize=14)

plt.subplot(1, 2, 1)
sns.histplot(df_z['MathScore'], kde=True, color='orange')
plt.title('Before Log Transformation (Z-score)')

plt.subplot(1, 2, 2)
sns.histplot(df_z['Log_MathScore'], kde=True, color='lightcoral')
plt.title('After Log Transformation (Z-score)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()