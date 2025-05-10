import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 读取数据（使用正确分隔符）
df = pd.read_csv("/Users/austin/Desktop/Project/Telco Customer Churn/modeling_data.csv", delimiter=";")

# 2. 目标变量设定（大写 Churn）
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. 类别变量独热编码
X_encoded = pd.get_dummies(X, drop_first=True)

# 4. 标准化数值变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. 定义三个模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier()
}

# 7. K折交叉验证评估准确率
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("=== Cross Validation Accuracy ===")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# 8. 用逻辑回归模型进行最终训练和预测
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\n=== Final Model Evaluation ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 10. 自定义阈值预测（例如：0.3）
threshold = 0.3
y_probs = final_model.predict_proba(X_test)[:, 1]
y_custom_pred = (y_probs >= threshold).astype(int)

print("\n=== Custom Threshold (0.3) Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_custom_pred))
print("Classification Report:\n", classification_report(y_test, y_custom_pred))

# 11. 混淆矩阵（自定义阈值）
conf_matrix_custom = confusion_matrix(y_test, y_custom_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_custom, annot=True, fmt="d", cmap="Oranges", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Custom Threshold (0.3)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 12. 分析逻辑回归模型的特征重要性（解释 key churn reasons）
coefficients = pd.Series(final_model.coef_[0], index=X_encoded.columns)
coefficients_sorted = coefficients.sort_values(key=abs, ascending=False)

print("\n=== Top 10 Churn Drivers (Logistic Regression) ===")
print(coefficients_sorted.head(10))

# 可视化 Top 10
plt.figure(figsize=(8, 6))
coefficients_sorted.head(10).plot(kind='barh')
plt.title("Top 10 Churn Influencing Features")
plt.xlabel("Coefficient (positive = more likely to churn)")
plt.tight_layout()
plt.show()
