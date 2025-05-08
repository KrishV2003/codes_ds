import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('Employee.csv')
#     • Education (Bachelors, Masters, PHD)
#     • JoiningYear (year they joined company)
#     • City (Bangalore, Pune, New Delhi)
#     • PaymentTier (1–3)
#     • Age
#     • Gender
#     • EverBenched (Yes/No)
#     • ExperienceInCurrentDomain (years)
#     • LeaveOrNot (0 = stayed, 1 = left)
df.head()
df.shape
df.ndim
df.info()
df.dtypes
df.isnull().sum()
df.describe()
for col in ['Education', 'City']:
    df[col] = df[col].astype('category')
df.dtypes
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).infer_objects(copy=False).astype(int)
df['EverBenched'] = df['EverBenched'].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False).astype(int)
df = pd.get_dummies(
    df,
    columns=['Education', 'City', 'PaymentTier'],
    drop_first=True
)
scaler = MinMaxScaler()
for num_col in ['Age', 'JoiningYear', 'ExperienceInCurrentDomain', 'LeaveOrNot']:
    df[num_col] = scaler.fit_transform(df[[num_col]])
df[['Age', 'JoiningYear', 'ExperienceInCurrentDomain', 'LeaveOrNot']].agg(['min', 'max'])
df.info()
