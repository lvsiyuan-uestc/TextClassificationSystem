import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import sys
import jieba

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import re
# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


# 加载数据函数
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            label, content = line.strip().split(" ", 1)
            texts.append(content)
            labels.append(label.split()[0])
    return texts, labels


# 重定向输出类
class RedirectText(object):
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass


class TextClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文本分类系统")

        self.create_widgets()
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(x=510, y=250)
        self.ax.set_title("各类别预测概率分布", fontsize=12, fontweight='bold')
    def create_widgets(self):
        main_frame = tk.Frame(root, bg="darkblue")  # 设置背景色深蓝色



        main_frame.pack(fill="both", expand=True)
        # 按钮：开始训练

        self.train_button = tk.Button(main_frame, text="开始训练模型", command=self.start_training_thread,
                                      font='Simsun 14 bold')
        self.train_button.place(x=15, y=10)

        # log标签
        self.log_label = tk.Label(main_frame, text="运行日志在这里显示：", bg="lightblue", fg="black",
                                  font='FangSong 14 bold')
        self.log_label.place(x=15, y=60)

        # 文本输入框和分类按钮
        self.input_label = tk.Label(main_frame, text="请输入待分类文本：", bg="lightblue", fg="black",
                                    font='FangSong 14 bold')
        self.input_label.place(y=60, x=505)
        # 日志输出区域
        self.log_area = scrolledtext.ScrolledText(main_frame, width=65, height=35)
        self.log_area.place(x=15, y=90)

        # 输入框
        self.input_entry = tk.Entry(main_frame, width=65)
        self.input_entry.place(y=90, x=505, height=50)

        # 预测按钮
        self.predict_button = tk.Button(main_frame, text="开始分类", command=self.predict_text, bg="white",
                                        font='Simsun 16 bold')
        self.predict_button.place(y=170, x=505)

        # 预测结果
        self.result_label = tk.Label(main_frame, text="", bg="darkblue", fg="white",font='Simsun 16 bold')
        self.result_label.place(y=220, x=505)

        self.load_button = tk.Button(main_frame, text="加载已有模型", command=self.load_model, font='Simsun 14 bold')
        self.load_button.place(x=505, y=10)

        sys.stdout = RedirectText(self.log_area)  # 重定向输出

    def start_training_thread(self):
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        print("加载数据中...")
        train_texts, train_labels = load_data("processed_data/cnews.train_seg.txt")
        val_texts, val_labels = load_data("processed_data/cnews.val_seg.txt")
        test_texts, test_labels = load_data("processed_data/cnews.test_seg.txt")

        print("开始 TF-IDF 特征提取...")
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)
        print("TF-IDF 特征维度:", X_train.shape)

        os.makedirs("models", exist_ok=True)
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

        print("进行 SVD 降维...")
        svd = TruncatedSVD(n_components=200, random_state=42)
        X_train_svd = svd.fit_transform(X_train)
        X_val_svd = svd.transform(X_val)
        X_test_svd = svd.transform(X_test)
        print("降维后维度:", X_train_svd.shape)

        joblib.dump(svd, "models/truncated_svd.pkl")

        print("正在进行不同 K 值验证...")
        k_values = list(range(1, 11))
        val_accuracies = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
            knn.fit(X_train_svd, train_labels)
            val_preds = knn.predict(X_val_svd)
            acc = accuracy_score(val_labels, val_preds)
            val_accuracies.append(acc)
            print(f"K = {k}，验证集准确率: {acc:.4f}")

        # plt.figure(figsize=(10, 6))
        # plt.plot(k_values, val_accuracies, marker='o', linestyle='-', color='teal')
        # plt.title("不同K值下的验证集准确率")
        # plt.xlabel("K值 (n_neighbors)")
        # plt.ylabel("准确率")
        # plt.grid(True)
        # plt.xticks(k_values)
        # plt.ylim(0, 1)
        # plt.savefig("models/knn_validation_accuracy.png")
        # plt.close()

        best_k = k_values[np.argmax(val_accuracies)]
        print(f"\n最优K值为: {best_k}")

        print("开始测试集评估...")
        best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
        best_knn.fit(X_train_svd, train_labels)
        test_preds = best_knn.predict(X_test_svd)

        test_acc = accuracy_score(test_labels, test_preds)
        print(f"测试集准确率: {test_acc:.4f}")
        print("\n[测试集分类报告]")
        print(classification_report(test_labels, test_preds, digits=4))

        joblib.dump(best_knn, f"models/best_knn_k{best_k}.pkl")

        self.vectorizer = vectorizer
        self.svd = svd
        self.classifier = best_knn

        messagebox.showinfo("训练完成", f"模型训练完毕，最优K值为{best_k}，已保存模型。")

    def load_model(self):
        try:
            self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
            self.svd = joblib.load("models/truncated_svd.pkl")
            # 自动查找最佳 K 模型文件
            for f in os.listdir("models"):
                if f.startswith("best_knn_k") and f.endswith(".pkl"):
                    self.classifier = joblib.load(os.path.join("models", f))
                    break
            messagebox.showinfo("模型加载", "模型加载成功！")
        except Exception as e:
            messagebox.showerror("加载错误", f"加载模型失败：{str(e)}")

    def clean_text(self,text):
        # 去除标点符号和特殊字符，只保留中英文和数字
        text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text)  # 去除多余空格
        return text.strip()
    def predict_text(self):
        if not hasattr(self, 'classifier'):
            messagebox.showwarning("警告", "请先训练模型！")
            return

        text = self.input_entry.get().strip()
        if not text:
            messagebox.showwarning("警告", "请输入要分类的文本")
            return

        text = self.clean_text(text)
        words = jieba.lcut(text)
        text = ' '.join(words)
        vec = self.vectorizer.transform([text])
        vec_svd = self.svd.transform(vec)
        pred = self.classifier.predict(vec_svd)[0]
        self.result_label.config(text=f"输入的文本预测类别：{pred}")
        proba = self.classifier.predict_proba(vec_svd)[0]
        classes = self.classifier.classes_

        proba = self.classifier.predict_proba(vec_svd)[0]
        classes = self.classifier.classes_

        # 绘制图表
        self.ax.clear()
        bars = self.ax.bar(classes, proba, color='darkred')  # 设置柱状图颜色为深红色
        self.ax.set_ylabel("预测概率")
        self.ax.set_xlabel("类别")

        self.ax.set_ylim(0, 1)

        # 显示概率数值在柱状图上
        for bar, p in zip(bars, proba):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{p:.2f}", ha='center', va='bottom', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1000x600')
    root.resizable(False, False)
    app = TextClassifierApp(root)
    root.mainloop()
