import pandas as pd
import numpy as np
df = pd.read_csv('Mall_Customers.csv')
df.head()
df.isnull().sum()
df.info()
cat_col = 'Genre'
num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
grouped_stats = (
    df
    .groupby(cat_col, observed=True)[num_cols]
    .agg(['mean', 'median', 'min', 'max', 'std'])
    .round(2)
)
print('\nSummary statistics by', cat_col)
for category, stats in grouped_stats.iterrows():
    print(f"\n>>> {category} <<<")
    print(stats)
df[cat_col] = pd.Categorical(df[cat_col])
numeric_codes = df[cat_col].cat.codes.tolist()
print('\nNumeric code list for', cat_col, ':')
print(numeric_codes)
df_iris = pd.read_csv('IRIS.csv')
df_iris
percentiles = [0, 25, 50, 75, 100]
# For each species, compute and print statistics
for sp in df_iris['species'].unique():
    sub = df_iris[df_iris['species'] == sp]
    print(f"\n=== Statistics for {sp} ===")
    
    # Basic descriptive stats (rounded)
    desc = sub.describe().loc[['count','mean','std','min','25%','50%','75%','max']]
    print(desc.round(2))

    # Custom percentiles (rounded)
    p1 = np.round(np.percentile(sub['sepal_length'], percentiles), 2)
    p2 = np.round(np.percentile(sub['sepal_width'], percentiles), 2)
    p3 = np.round(np.percentile(sub['petal_length'], percentiles), 2)
    p4 = np.round(np.percentile(sub['petal_width'], percentiles), 2)

    # Labeled and formatted output
    print(f"\nSepal Length ({sp}) percentiles (0,25,50,75,100): {p1}")
    print(f"Sepal Width ({sp}) percentiles (0,25,50,75,100): {p2}")
    print(f"Petal Length ({sp}) percentiles (0,25,50,75,100): {p3}")
    print(f"Petal Width ({sp}) percentiles (0,25,50,75,100): {p4}")
