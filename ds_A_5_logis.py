import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
df = pd.read_csv('Social_Network_Ads.csv')
df
df.info()
df.dtypes
df.duplicated()
df.isnull().sum()
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state=0
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
TP, FP = cm[1, 1], cm[0, 1]
TN, FN = cm[0, 0], cm[1, 0]
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
print(f'Confuision Matrix : \n {cm}')
print(f'TP : {TP}, TN : {TN}, FP : {FP}, FN : {FN}')
print(f'Accuracy : {accuracy:.2f}')
print(f'Error rate : {error_rate:.2f}')
print(f'Precision : {precision:.2f}')
print(f'Recall : {recall:.2f}')
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Not Purchased', 'Purchased'], fontsize=10)
plt.yticks([0, 1], ['Not Purchased', 'Purchased'], fontsize=10)
plt.xlabel('Predicted', fontsize = 12)
plt.ylabel('Actual', fontsize = 12)
plt.title('Confusion Matrix', fontsize = 16)
for i in range(2):
    for j in range(2):
        plt.text(j ,i , cm[i, j], ha='center', va='center', color='red', fontsize = 14)
plt.gca().xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.show()