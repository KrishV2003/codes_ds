import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("customer_churn_dataset.csv")
df.head()
df.shape
df.ndim
df.isnull().sum()
df.describe()
df.info()
df.dtypes
# Variable descriptions:
# | Column              | Type     | Description                                          |
# |---------------------|----------|------------------------------------------------------|
# | CustomerID          | int64    | Unique customer identifier                           |
# | Age                 | int64    | Customer’s age                                       |
# | Gender              | object   | Male or Female                                       |
# | Tenure              | int64    | Months as a subscriber                               |
# | Usage Frequency     | int64    | Number of uses per month                             |
# | Support Calls       | int64    | Number of support calls made                         |
# | Payment Delay       | int64    | Days of delay in payment                             |
# | Subscription Type   | object   | Tier: Basic, Standard, or Premium                    |
# | Contract Length     | object   | Billing cycle: Monthly, Quarterly, or Annual         |
# | Total Spend         | int64    | Total amount spent                                   |
# | Last Interaction    | int64    | Days since last interaction                          |
# | Churn               | int64    | 1 = churned, 0 = retained                            |
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    df[col] = df[col].astype('category')
df.dtypes
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
df['Churn']  = df['Churn'].astype(int)
df = pd.get_dummies(
    df,
    columns=['Subscription Type', 'Contract Length'],
    drop_first=True
)
numeric_cols = [
    'Age', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Total Spend', 'Last Interaction', 'Churn'
]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df[numeric_cols].agg(['min', 'max'])
df.info()
