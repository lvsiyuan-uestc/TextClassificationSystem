from matplotlib import rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            label, content = line.strip().split(" ", 1)
            texts.append(content)
            labels.append(label.split()[0])


    return texts, labels


train_texts, train_labels = load_data("processed_data/cnews.train_seg.txt")
val_texts, val_labels = load_data("processed_data/cnews.val_seg.txt")
test_texts, test_labels = load_data("processed_data/cnews.test_seg.txt")

# 使用 TF-IDF 特征提取，包含 bi-gram 且维度提高
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
# 指示向量化器不仅提取单个词，还提取连续两个词组成的词组
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# 保存 TF-IDF 模型
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("原始 TF-IDF 训练集维度:", X_train.shape)

# 使用 TruncatedSVD 降维
svd = TruncatedSVD(n_components=200, random_state=42)
# Truncated SVD 是一种常用的降维技术，特别适用于稀疏矩阵（如 TF-IDF 输出）。它通过找到数据的主要成分来将数据投影到较低维空间。
X_train_svd = svd.fit_transform(X_train)
X_val_svd = svd.transform(X_val)
X_test_svd = svd.transform(X_test)

# 保存 SVD 模型
joblib.dump(svd, "models/truncated_svd.pkl")

print("降维后训练集维度:", X_train_svd.shape)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 尝试1-10K值
k_values = list(range(1, 11))
val_accuracies = []

print("正在进行不同K值下的验证集测试...\n")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
    knn.fit(X_train_svd, train_labels)
    val_preds = knn.predict(X_val_svd)
    acc = accuracy_score(val_labels, val_preds)
    val_accuracies.append(acc)
    print(f"K = {k}，验证集准确率: {acc:.4f}")

# 可视化准确率随K变化的图
plt.figure(figsize=(10, 6))
plt.plot(k_values, val_accuracies, marker='o', linestyle='-', color='teal')
plt.title("不同K值下的验证集准确率")
plt.xlabel("K值 (n_neighbors)")
plt.ylabel("准确率")
plt.grid(True)
plt.xticks(k_values)
plt.ylim(0, 1)
plt.savefig("models/knn_validation_accuracy.png")
plt.show()

# 选出最佳 K 值
best_k = k_values[np.argmax(val_accuracies)]
print(f"\n最优K值为: {best_k}, 验证集最高准确率为: {max(val_accuracies):.4f}")

# 测试集评估
print("\n正在测试集上进行评估...")
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
best_knn.fit(X_train_svd, train_labels)
test_preds = best_knn.predict(X_test_svd)

test_acc = accuracy_score(test_labels, test_preds)
print(f"测试集准确率: {test_acc:.4f}")
print("\n[测试集分类报告]")
print(classification_report(test_labels, test_preds, digits=4))

# 保存模型
joblib.dump(best_knn, f"models/best_knn_k{best_k}.pkl")
