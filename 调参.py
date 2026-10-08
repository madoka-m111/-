import pandas as pd
import numpy as np
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ========== 1. 加载数据 ==========
def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X = train_df['text'].astype(str).tolist()
    y = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()
    return X, y, X_test

X, y, X_test = load_data()

# ========== 2. 划分训练/验证集 ==========
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_tr)}, 验证集: {len(X_val)}")
print("="*60)

# ========== 3. 定义参数搜索空间 ==========

# TF-IDF 参数组合
tfidf_configs = [
    {'max_features': 5000, 'ngram_range': (1, 1), 'min_df': 2},
    {'max_features': 10000, 'ngram_range': (1, 2), 'min_df': 2},
    {'max_features': 20000, 'ngram_range': (1, 2), 'min_df': 2},
    {'max_features': 20000, 'ngram_range': (1, 3), 'min_df': 2},
]

# SVM 参数组合
svm_configs = [
    {'C': 0.1},
    {'C': 0.5},
    {'C': 1.0},
    {'C': 5.0},
    {'C': 10.0},
]

# 逻辑回归参数组合
lr_configs = [
    {'C': 0.1},
    {'C': 0.5},
    {'C': 1.0},
    {'C': 5.0},
    {'C': 10.0},
]

# MLP 参数组合
mlp_configs = [
    {'hidden_layer_sizes': (50,), 'max_iter': 300},
    {'hidden_layer_sizes': (100,), 'max_iter': 300},
    {'hidden_layer_sizes': (100, 50), 'max_iter': 300},
    {'hidden_layer_sizes': (150,), 'max_iter': 500},
]

# ========== 4. 调参主循环 ==========
all_results = []  # 存储所有结果

for tfidf_idx, tfidf_params in enumerate(tfidf_configs):
    print(f"\n{'='*60}")
    print(f"TF-IDF 配置 {tfidf_idx+1}/{len(tfidf_configs)}: {tfidf_params}")
    print('='*60)
    
    # 构建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"特征维度: {X_tr_vec.shape[1]}")
    
    # --- SVM 调参 ---
    print("\n--- SVM 调参 ---")
    best_svm_acc = 0
    best_svm_params = None
    for svm_params in svm_configs:
        svm = SVC(kernel='linear', C=svm_params['C'], random_state=42)
        svm.fit(X_tr_vec, y_tr)
        acc = accuracy_score(y_val, svm.predict(X_val_vec))
        all_results.append({
            '模型': 'SVM',
            'TF-IDF': f"max_features={tfidf_params['max_features']}, ngram={tfidf_params['ngram_range']}",
            '参数': f"C={svm_params['C']}",
            '准确率': acc
        })
        print(f"  C={svm_params['C']:<5} 准确率: {acc:.4f}")
        
        if acc > best_svm_acc:
            best_svm_acc = acc
            best_svm_params = svm_params
    
    # --- 逻辑回归调参 ---
    print("\n--- 逻辑回归调参 ---")
    best_lr_acc = 0
    best_lr_params = None
    for lr_params in lr_configs:
        lr = LogisticRegression(max_iter=1000, C=lr_params['C'], random_state=42)
        lr.fit(X_tr_vec, y_tr)
        acc = accuracy_score(y_val, lr.predict(X_val_vec))
        all_results.append({
            '模型': '逻辑回归',
            'TF-IDF': f"max_features={tfidf_params['max_features']}, ngram={tfidf_params['ngram_range']}",
            '参数': f"C={lr_params['C']}",
            '准确率': acc
        })
        print(f"  C={lr_params['C']:<5} 准确率: {acc:.4f}")
        
        if acc > best_lr_acc:
            best_lr_acc = acc
            best_lr_params = lr_params
    
    # --- MLP 调参 ---
    print("\n--- MLP 调参 ---")
    best_mlp_acc = 0
    best_mlp_params = None
    for mlp_params in mlp_configs:
        mlp = MLPClassifier(
            hidden_layer_sizes=mlp_params['hidden_layer_sizes'],
            max_iter=mlp_params['max_iter'],
            random_state=42
        )
        mlp.fit(X_tr_vec, y_tr)
        acc = accuracy_score(y_val, mlp.predict(X_val_vec))
        all_results.append({
            '模型': 'MLP',
            'TF-IDF': f"max_features={tfidf_params['max_features']}, ngram={tfidf_params['ngram_range']}",
            '参数': f"hidden={mlp_params['hidden_layer_sizes']}, max_iter={mlp_params['max_iter']}",
            '准确率': acc
        })
        print(f"  hidden={mlp_params['hidden_layer_sizes']}, max_iter={mlp_params['max_iter']} 准确率: {acc:.4f}")
        
        if acc > best_mlp_acc:
            best_mlp_acc = acc
            best_mlp_params = mlp_params
    
    # 记录当前 TF-IDF 配置下的最优结果
    print(f"\n当前 TF-IDF 配置下最优:")
    print(f"  SVM最优: C={best_svm_params['C']}, 准确率: {best_svm_acc:.4f}")
    print(f"  逻辑回归最优: C={best_lr_params['C']}, 准确率: {best_lr_acc:.4f}")
    print(f"  MLP最优: hidden={best_mlp_params['hidden_layer_sizes']}, 准确率: {best_mlp_acc:.4f}")

# ========== 5. 汇总结果 ==========
print("\n" + "="*60)
print("全部调参结果汇总")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('准确率', ascending=False)

# 打印 Top 10
print("\nTop 10 最优配置:")
print(results_df.head(10).to_string(index=False))

# ========== 6. 保存所有结果到CSV ==========
results_df.to_csv('tuning_results.csv', index=False)
print(f"\n所有调参结果已保存至 tuning_results.csv")

# ========== 7. 用全局最优模型预测测试集 ==========
best_row = results_df.iloc[0]
print(f"\n全局最优配置:")
print(f"  模型: {best_row['模型']}")
print(f"  TF-IDF: {best_row['TF-IDF']}")
print(f"  参数: {best_row['参数']}")
print(f"  验证集准确率: {best_row['准确率']:.4f}")

# 根据最优配置重新训练并预测
# 解析 TF-IDF 参数
if 'ngram=(1, 1)' in best_row['TF-IDF']:
    ngram = (1, 1)
elif 'ngram=(1, 3)' in best_row['TF-IDF']:
    ngram = (1, 3)
else:
    ngram = (1, 2)

max_features = int(best_row['TF-IDF'].split('max_features=')[1].split(',')[0])

final_vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=ngram,
    min_df=2
)
X_tr_final = final_vectorizer.fit_transform(X_tr)
X_test_final = final_vectorizer.transform(X_test)

# 根据最优模型类型训练
if best_row['模型'] == 'SVM':
    C = float(best_row['参数'].split('C=')[1])
    final_model = SVC(kernel='linear', C=C, random_state=42)
elif best_row['模型'] == '逻辑回归':
    C = float(best_row['参数'].split('C=')[1])
    final_model = LogisticRegression(max_iter=1000, C=C, random_state=42)
elif best_row['模型'] == 'MLP':
    # 解析 hidden_layer_sizes
    hidden_str = best_row['参数'].split('hidden=')[1].split(',')[0]
    if '100, 50' in hidden_str:
        hidden = (100, 50)
    elif '50' in hidden_str and '100' not in hidden_str:
        hidden = (50,)
    else:
        hidden = (100,)
    final_model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300, random_state=42)
else:  # 朴素贝叶斯
    final_model = MultinomialNB()

final_model.fit(X_tr_final, y_tr)
test_preds = final_model.predict(X_test_final)


# ========== 8. 输出调参总结 ==========
print("\n" + "="*60)
print("="*60)
for model_name in ['SVM', '逻辑回归', 'MLP']:
    model_results = results_df[results_df['模型'] == model_name]
    best_for_model = model_results.iloc[0]
    print(f"\n{model_name}:")
    print(f"  最优准确率: {best_for_model['准确率']:.4f}")
    print(f"  最优参数: {best_for_model['参数']}")
    print(f"  最优TF-IDF: {best_for_model['TF-IDF']}")
