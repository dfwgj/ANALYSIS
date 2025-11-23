"""
贫血数据集分析与建模
步骤3：建模和评估（SMOTE版本）
使用SMOTE处理数据不平衡
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

# 设置中文字体和UTF-8编码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")
output_path.mkdir(exist_ok=True)

print("=" * 70)
print("贫血数据集 - 建模和评估（SMOTE版本）")
print("=" * 70)

# ============================================================
# 【1】加载数据
# ============================================================
print("\n【1】加载数据...")
X_train = pd.read_csv(output_path / "X_train.csv")
X_test = pd.read_csv(output_path / "X_test.csv")
y_train = pd.read_csv(output_path / "y_train.csv").squeeze()
y_test = pd.read_csv(output_path / "y_test.csv").squeeze()

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# ============================================================
# 【2】应用SMOTE过采样
# ============================================================
print("\n【2】数据平衡策略 - 应用SMOTE")
print("=" * 70)

# 查看SMOTE前的类别分布
print("\nSMOTE前的类别分布:")
print("-" * 40)
original_dist = Counter(y_train)
for cls in sorted(original_dist.keys()):
    count = original_dist[cls]
    percentage = count / len(y_train) * 100
    print(f"  类别 {cls}: {count:5d} 样本 ({percentage:5.2f}%)")

print(f"\n不平衡比例: {max(original_dist.values()) / min(original_dist.values()):.1f}:1")

# 应用SMOTE
print("\n应用SMOTE过采样...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE后训练集: {X_train_smote.shape}")

# 查看SMOTE后的类别分布
print("\nSMOTE后的类别分布:")
print("-" * 40)
smote_dist = Counter(y_train_smote)
for cls in sorted(smote_dist.keys()):
    count = smote_dist[cls]
    percentage = count / len(y_train_smote) * 100
    original = original_dist[cls]
    added = count - original
    print(f"  类别 {cls}: {count:5d} 样本 ({percentage:5.2f}%)  [新增 {added:4d}]")

print(f"\nSMOTE后不平衡比例: {max(smote_dist.values()) / min(smote_dist.values()):.1f}:1")
print("✓ SMOTE后数据已完全平衡")

# ============================================================
# 【3】定义模型（使用SMOTE后的数据）
# ============================================================
print("\n【3】定义模型...")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='multinomial'
        # 注意：使用SMOTE后，去掉class_weight='balanced'，因为数据已平衡
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    ),
    'SVM': SVC(
        kernel='rbf',
        random_state=42,
        probability=True
    )
}

# ============================================================
# 【4】模型训练与评估（使用SMOTE数据）
# ============================================================
print("\n【4】交叉验证评估（使用SMOTE数据）...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    print(f"\n训练 {name}...")

    # 交叉验证（在SMOTE数据上）
    cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring='f1_macro')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"  交叉验证 F1 (macro): {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 在训练集上训练（SMOTE数据）
    model.fit(X_train_smote, y_train_smote)

    # 在测试集上预测（注意：测试集保持原样！）
    y_pred = model.predict(X_test)

    # 计算各种指标
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
    weighted_f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

    print(f"  测试集 Accuracy: {accuracy:.4f}")
    print(f"  测试集 Macro F1: {macro_f1:.4f}")

    # 保存结果
    results.append({
        'Model': name,
        'CV_F1_Mean': cv_mean,
        'CV_F1_Std': cv_std,
        'Test_Accuracy': accuracy,
        'Test_Macro_F1': macro_f1,
        'Test_Weighted_F1': weighted_f1,
        'Model_Obj': model,
        'Predictions': y_pred
    })

# ============================================================
# 【5】结果对比（SMOTE vs 原始）
# ============================================================
print("\n【5】模型性能对比（SMOTE vs 原始数据）")
print("=" * 70)

# 加载原始数据的结果
original_results = pd.read_csv(output_path / 'model_results.csv')

comparison_data = []
for result in results:
    name = result['Model']
    smote_f1 = result['Test_Macro_F1']
    smote_acc = result['Test_Accuracy']

    # 查找该模型在原始数据上的结果
    original_row = original_results[original_results['Model'] == name]
    if not original_row.empty:
        original_f1 = original_row.iloc[0]['Test_Macro_F1']
        original_acc = original_row.iloc[0]['Test_Accuracy']

        # 计算提升
        f1_improvement = smote_f1 - original_f1
        acc_improvement = smote_acc - original_acc

        comparison_data.append({
            'Model': name,
            'Original_F1': original_f1,
            'SMOTE_F1': smote_f1,
            'F1_Improvement': f1_improvement,
            'Original_Acc': original_acc,
            'SMOTE_Acc': smote_acc,
            'Acc_Improvement': acc_improvement
        })

comparison_df = pd.DataFrame(comparison_data)
print("\nMacro F1 对比:")
print("-" * 70)
print(f"{'Model':<25} {'Original':<10} {'SMOTE':<10} {'Change':<10}")
print("-" * 70)
for _, row in comparison_df.iterrows():
    change_marker = "↑" if row['F1_Improvement'] > 0 else "↓"
    print(f"{row['Model']:<25} {row['Original_F1']:<10.4f} {row['SMOTE_F1']:<10.4f} {row['F1_Improvement']:<10.4f} {change_marker}")

# 选择最佳模型（SMOTE版本）
results_df = pd.DataFrame(results)
best_model_idx = results_df['Test_Macro_F1'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model = results_df.loc[best_model_idx, 'Model_Obj']
best_predictions = results_df.loc[best_model_idx, 'Predictions']

print(f"\n[SUCCESS] SMOTE最佳模型: {best_model_name}")
print(f"SMOTE Macro F1: {results_df.loc[best_model_idx, 'Test_Macro_F1']:.4f}")

# ============================================================
# 【6】详细评估最佳模型（SMOTE）
# ============================================================
print(f"\n【6】{best_model_name} (SMOTE) 详细分类报告:")
print("=" * 60)
print(classification_report(y_test, best_predictions))

# 混淆矩阵
print(f"\n【7】混淆矩阵（{best_model_name} + SMOTE）:")
print("=" * 60)
cm = confusion_matrix(y_test, best_predictions)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No anemia', 'HGB-anemia', 'Iron deficient', 'Folate deficient', 'B12 deficient'],
            yticklabels=['No anemia', 'HGB-anemia', 'Iron deficient', 'Folate deficient', 'B12 deficient'])
plt.title(f'Confusion Matrix - {best_model_name} (SMOTE)', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(output_path / 'confusion_matrix_smote_best.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[SUCCESS] 混淆矩阵已保存到: {output_path / 'confusion_matrix_smote_best.png'}")

# ============================================================
# 【8】过拟合检查
# ============================================================
print(f"\n【8】过拟合检查 (SMOTE):")
print("=" * 60)
print("比较训练集和测试集的性能差异")
print("-" * 60)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Diff':<10} {'Status'}")
print("-" * 60)

overfit_results = []
for result in results:
    name = result['Model']
    model = result['Model_Obj']

    # 在SMOTE训练集上预测
    y_train_pred = model.predict(X_train_smote)
    train_acc = accuracy_score(y_train_smote, y_train_pred)
    test_acc = result['Test_Accuracy']
    diff = train_acc - test_acc

    if diff > 0.1:
        status = "!! OVERFIT !!"
    elif diff > 0.05:
        status = "! Warning"
    elif diff > 0.02:
        status = "~ Slight"
    else:
        status = "OK"

    print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {diff:<10.4f} {status}")
    overfit_results.append({'Model': name, 'Train_Acc': train_acc, 'Test_Acc': test_acc, 'Diff': diff, 'Status': status})

print("-" * 60)

best_overfit = [r for r in overfit_results if r['Model'] == best_model_name][0]
if best_overfit['Diff'] < 0.02:
    print(f"\n[CONCLUSION] {best_model_name} + SMOTE 没有过拟合! (Gap: {best_overfit['Diff']:.4f})")
else:
    print(f"\n[WARNING] {best_model_name} + SMOTE 存在过拟合! (Gap: {best_overfit['Diff']:.4f})")

# ============================================================
# 【9】特征重要性
# ============================================================
if hasattr(best_model, 'feature_importances_'):
    print(f"\n【9】特征重要性（{best_model_name} + SMOTE）:")
    print("=" * 60)

    importances = best_model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(feature_importance_df.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature')
    plt.title(f'Top 15 Feature Importances - {best_model_name} (SMOTE)', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance_smote_best.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 特征重要性图已保存")

# ============================================================
# 【10】保存结果
# ============================================================
print("\n【10】保存结果...")
print("=" * 60)

# 保存SMOTE结果
results_df.to_csv(output_path / 'model_results_smote.csv', index=False)
comparison_df.to_csv(output_path / 'smote_comparison.csv', index=False)

# 保存最佳SMOTE模型
with open(output_path / 'best_model_smote.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'model_name': best_model_name,
        'predictions': best_predictions,
        'metrics': {
            'accuracy': results_df.loc[best_model_idx, 'Test_Accuracy'],
            'macro_f1': results_df.loc[best_model_idx, 'Test_Macro_F1'],
            'weighted_f1': results_df.loc[best_model_idx, 'Test_Weighted_F1']
        }
    }, f)

print(f"[SUCCESS] SMOTE结果已保存到: {output_path / 'model_results_smote.csv'}")
print(f"[SUCCESS] SMOTE对比已保存到: {output_path / 'smote_comparison.csv'}")
print(f"[SUCCESS] 最佳SMOTE模型已保存到: {output_path / 'best_model_smote.pkl'}")

# ============================================================
# 【11】关键发现总结
# ============================================================
print("\n【11】SMOTE实验总结:")
print("=" * 60)
print(f"1. 最佳模型: {best_model_name}")
print(f"2. SMOTE Macro F1: {results_df.loc[best_model_idx, 'Test_Macro_F1']:.4f}")
print(f"3. SMOTE Accuracy: {results_df.loc[best_model_idx, 'Test_Accuracy']:.4f}")

# 对比提升
original_best_f1 = original_results.iloc[0]['Test_Macro_F1']
smote_best_f1 = results_df.iloc[0]['Test_Macro_F1']
improvement = smote_best_f1 - original_best_f1

print(f"4. 相比原始数据提升: {improvement:+.4f}")
if improvement > 0:
    print("   ✓ SMOTE有效！模型性能提升")
elif improvement < -0.01:
    print("   ✗ SMOTE效果不佳，可能过拟合")
else:
    print("   ~ SMOTE效果不明显，可能数据本身已足够好")

print(f"5. 数据平衡: 从 {max(original_dist.values()) / min(original_dist.values()):.1f}:1 到 1:1")

print("\n" + "=" * 70)
print("SMOTE建模和评估完成！")
print("=" * 70)
