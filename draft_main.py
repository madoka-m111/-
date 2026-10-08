import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. 加载数据 ==========
def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

X_train, y_train, X_test_unlabeled = load_data()

# ========== 2. 数据加载验证 ==========
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# ========== 3. 划分训练/验证集 ==========
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print(f"\n训练集大小: {len(X_tr)}, 验证集大小: {len(X_val)}")
print("="*50)

# ========== 4. TF-IDF 特征 ==========
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)
X_tr_vec = vectorizer.fit_transform(X_tr)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test_unlabeled)

print(f"TF-IDF 特征维度: {X_tr_vec.shape[1]}")
print("="*50)

# ========== 5. 模型训练与验证 ==========
results = {}
models = {}

# --- 朴素贝叶斯 ---
nb = MultinomialNB()
nb.fit(X_tr_vec, y_tr)
val_pred_nb = nb.predict(X_val_vec)
acc_nb = accuracy_score(y_val, val_pred_nb)
results['朴素贝叶斯'] = acc_nb
models['朴素贝叶斯'] = nb
print(f"朴素贝叶斯 验证集准确率: {acc_nb:.4f}")

# --- SVM ---
svm = SVC(kernel='linear', C=5.0, random_state=42)  #
svm.fit(X_tr_vec, y_tr)
val_pred_svm = svm.predict(X_val_vec)
acc_svm = accuracy_score(y_val, val_pred_svm)
results['SVM'] = acc_svm
models['SVM'] = svm
print(f"SVM 验证集准确率: {acc_svm:.4f}")

# --- 逻辑回归 ---
lr = LogisticRegression(
    max_iter=1000, C=10.0, random_state=42  
)
lr.fit(X_tr_vec, y_tr)
val_pred_lr = lr.predict(X_val_vec)
acc_lr = accuracy_score(y_val, val_pred_lr)
results['逻辑回归'] = acc_lr
models['逻辑回归'] = lr
print(f"逻辑回归 验证集准确率: {acc_lr:.4f}")

# --- MLP ---
mlp = MLPClassifier(
    hidden_layer_sizes=(150,), max_iter=500, random_state=42  
)
mlp.fit(X_tr_vec, y_tr)
val_pred_mlp = mlp.predict(X_val_vec)
acc_mlp = accuracy_score(y_val, val_pred_mlp)
results['MLP'] = acc_mlp
models['MLP'] = mlp
print(f"MLP 验证集准确率: {acc_mlp:.4f}")

# ========== 6. 输出分类报告 ==========
print("\n" + "="*50)
print("各模型分类报告（验证集）")
print("="*50)

for name, pred in [('朴素贝叶斯', val_pred_nb), ('SVM', val_pred_svm), 
                   ('逻辑回归', val_pred_lr), ('MLP', val_pred_mlp)]:
    print(f"\n{name} 分类报告:")
    print(classification_report(y_val, pred, zero_division=0))

# ========== 7. 选择最优模型 ==========
best_model_name = max(results, key=results.get)
print("\n" + "="*50)
print("模型性能汇总")
print("="*50)
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
print(f"\n最优模型: {best_model_name}，验证集准确率: {results[best_model_name]:.4f}")

# ========== 8. 用最优模型预测测试集 ==========
best_model = models[best_model_name]
test_preds = best_model.predict(X_test_vec)

# ========== 9. 保存预测结果 ==========
pd.DataFrame(test_preds).to_csv(
    'predictions.csv', index=False, header=False
)
print(f"预测结果已保存至 predictions.csv")
