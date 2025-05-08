import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
df = pd.read_csv('iris.csv')
if df['species'].dtype == 'object':
    class_names = df['species'].unique()
    class_names.sort()  # To match sklearn's order: ['setosa', 'versicolor', 'virginica']
    df['species'] = df['species'].astype('category').cat.codes
X = df.drop('species', axis=1).values
y = df['species'].values
class_names = ['setosa', 'versicolor', 'virginica']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')
total = cm.sum()
for i in range(len(class_names)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = total - TP - FP - FN

    accuracy = (TP + TN) / total
    error_rate = 1 - accuracy
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    print(f"\nClass: {class_names[i]}")
    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Error Rate: {error_rate:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix", fontsize=16)
plt.colorbar()
plt.xticks([0, 1, 2], class_names, fontsize=10)
plt.yticks([0, 1, 2], class_names, fontsize=10)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='lightcoral', fontsize=14)

plt.xlabel('Predicted label', fontsize=12)
plt.ylabel('Actual label', fontsize=12)
plt.tight_layout()
plt.show()