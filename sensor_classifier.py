import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 读取数据
df = pd.read_csv("sensor_data.csv")

# 计算合成加速度
df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

print(f"总数据量: {len(df)} 行")
print(f"各类别数量:\n{df['activity'].value_counts()}")

# ---- 特征提取函数 ----
def extract_features(data, window=50, step=25):
    X = []
    y = []
    for i in range(0, len(data) - window, step):
        window_data = data['acc_mag'].iloc[i:i+window]
        features = [
            window_data.mean(),
            window_data.std(),
            window_data.max(),
            window_data.min(),
            window_data.max() - window_data.min(),
            window_data.quantile(0.75) - window_data.quantile(0.25),
            np.sum(np.abs(np.diff(window_data))) / window
        ]
        X.append(features)
        from collections import Counter
        most_common = Counter(data['activity'].iloc[i:i+window]).most_common(1)[0][0]
        y.append(most_common)
    return np.array(X), np.array(y)

# ---- 先提取所有特征 ----
X_all, y_all = extract_features(df)
print(f"总样本数: {len(X_all)}")

# ---- 打乱并划分训练集和测试集 ----
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
print(f"训练集类别分布:\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布:\n{pd.Series(y_test).value_counts()}")

# ---- 训练模型 ----
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# ---- 评估 ----
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"准确率: {acc:.2f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))
