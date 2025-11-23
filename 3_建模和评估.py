"""
贫血数据集分析与建模
步骤3：建模和评估
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和UTF-8编码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")
output_path.mkdir(exist_ok=True)

print("=" * 60)
print("贫血数据集 - 建模和评估")
print("=" * 60)

# 加载处理后的数据
print("\n【1】加载数据...")
X_train = pd.read_csv(output_path / "X_train.csv")
X_test = pd.read_csv(output_path / "X_test.csv")
y_train = pd.read_csv(output_path / "y_train.csv").squeeze()
y_test = pd.read_csv(output_path / "y_test.csv").squeeze()

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 数据平衡策略
print("\n【2】数据平衡策略...")
print("由于数据严重不平衡（类别657:1），采用以下策略:")
print("1. 调整类别权重")
print("2. 使用能处理不平衡数据的算法")
print("3. 使用合适的评估指标（F1, Recall, AUC）")
1
# 定义模型
print("\n【3】定义模型...")

models = {
    # 逻辑回归
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='multinomial',
        class_weight='balanced'  # 调整类别权重
    ),
    # 决策树
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    # 随机森林
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    # 梯度提升树
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42
    ),
    # K-最近邻
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    ),
    # 支持向量机    
    'SVM': SVC(
        kernel='rbf',
        random_state=42,
        probability=True,
        class_weight='balanced'
    )
}

# 交叉验证设置
print("\n【4】交叉验证评估...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    print(f"\n训练 {name}...")

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"  交叉验证 F1 (macro): {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 在训练集上训练
    model.fit(X_train, y_train)

    # 在测试集上预测
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

# 转换为DataFrame
results_df = pd.DataFrame(results)
print("\n【5】模型性能排序（按Macro F1）:")
print("=" * 60)
print(results_df[['Model', 'Test_Accuracy', 'Test_Macro_F1', 'Test_Weighted_F1']].sort_values(
    'Test_Macro_F1', ascending=False
).to_string(index=False))

# 选择最佳模型
best_model_idx = results_df['Test_Macro_F1'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model = results_df.loc[best_model_idx, 'Model_Obj']
best_predictions = results_df.loc[best_model_idx, 'Predictions']

print(f"\n[SUCCESS] 最佳模型: {best_model_name}")
print(f"Macro F1: {results_df.loc[best_model_idx, 'Test_Macro_F1']:.4f}")

# 详细评估最佳模型
print(f"\n【6】{best_model_name} 详细分类报告:")
print("=" * 60)
print(classification_report(y_test, best_predictions))

# 混淆矩阵
print(f"\n【7】混淆矩阵（最佳模型: {best_model_name}）:")
print("=" * 60)
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# 保存混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No anemia', 'HGB-anemia', 'Iron deficient', 'Folate deficient', 'B12 deficient'],
            yticklabels=['No anemia', 'HGB-anemia', 'Iron deficient', 'Folate deficient', 'B12 deficient'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(output_path / 'confusion_matrix_best.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[SUCCESS] 混淆矩阵已保存到: {output_path / 'confusion_matrix_best.png'}")

# ============================================================
# 【8】过拟合检查
# ============================================================
print(f"\n【8】过拟合检查:")
print("=" * 60)
print("比较训练集和测试集的性能差异，判断是否过拟合")
print("-" * 60)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Diff':<10} {'Status'}")
print("-" * 60)

overfit_results = []
for result in results:
    name = result['Model']
    model = result['Model_Obj']

    # 在训练集上预测
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = result['Test_Accuracy']
    diff = train_acc - test_acc

    # 判断过拟合状态
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
print("Interpretation: Diff<0.02=OK, 0.02-0.05=Slight, 0.05-0.1=Warning, >0.1=Overfit")

# 生成学习曲线（最佳模型）
print(f"\n生成学习曲线...")
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, test_scores = learning_curve(
    best_model.__class__(**best_model.get_params()),
    X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes_abs, test_mean, 'o-', color='orange', label='Cross-Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score (Macro)')
plt.title(f'Learning Curve - {best_model_name}\n(Gap between lines indicates overfitting)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
final_gap = train_mean[-1] - test_mean[-1]
plt.annotate(f'Final Gap: {final_gap:.4f}', xy=(train_sizes_abs[-1], (train_mean[-1] + test_mean[-1])/2), fontsize=10, color='red')
plt.tight_layout()
plt.savefig(output_path / 'learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[SUCCESS] 学习曲线已保存到: {output_path / 'learning_curve.png'}")

# 过拟合结论
best_overfit = [r for r in overfit_results if r['Model'] == best_model_name][0]
if best_overfit['Diff'] < 0.02:
    print(f"\n[CONCLUSION] {best_model_name} 没有过拟合! (Train-Test Gap: {best_overfit['Diff']:.4f})")
elif best_overfit['Diff'] < 0.05:
    print(f"\n[CONCLUSION] {best_model_name} 轻微过拟合 (Train-Test Gap: {best_overfit['Diff']:.4f})")
else:
    print(f"\n[WARNING] {best_model_name} 存在过拟合! (Train-Test Gap: {best_overfit['Diff']:.4f})")

# 特征重要性（如果模型支持）
if hasattr(best_model, 'feature_importances_'):
    print(f"\n【9】特征重要性（{best_model_name}）:")
    print("=" * 60)

    importances = best_model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(feature_importance_df.head(10).to_string(index=False))

    # 可视化前15个重要特征
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature')
    plt.title(f'Top 15 Feature Importances - {best_model_name}', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance_best.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] 特征重要性图已保存")

elif hasattr(best_model, 'coef_'):
    print(f"\n【9】特征系数（{best_model_name}）:")
    print("=" * 60)
    coef = best_model.coef_
    feature_names = X_train.columns

    for idx, class_coef in enumerate(coef):
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': class_coef
        }).sort_values('Coefficient', key=abs, ascending=False)

        print(f"\n类别 {idx} 的前10个重要特征:")
        print(coef_df.head(10).to_string(index=False))

# 保存所有模型的预测结果
print("\n【10】保存模型和结果...")
print("=" * 60)

# 保存最佳模型
with open(output_path / 'best_model.pkl', 'wb') as f:
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

# 保存所有结果
results_df.to_csv(output_path / 'model_results.csv', index=False)

# 保存详细的分类报告
class_report = classification_report(y_test, best_predictions, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(output_path / 'classification_report_best.csv')

print(f"[SUCCESS] 最佳模型已保存到: {output_path / 'best_model.pkl'}")
print(f"[SUCCESS] 所有结果已保存到: {output_path / 'model_results.csv'}")
print(f"[SUCCESS] 分类报告已保存到: {output_path / 'classification_report_best.csv'}")

print("\n【11】关键发现总结:")
print("=" * 60)
print(f"1. 最佳模型: {best_model_name}")
print(f"2. 测试集 Macro F1: {results_df.loc[best_model_idx, 'Test_Macro_F1']:.4f}")
print(f"3. 测试集 Accuracy: {results_df.loc[best_model_idx, 'Test_Accuracy']:.4f}")
print(f"4. 主要困难: 数据严重不平衡 (644:1)")
print(f"5. 建议: 尝试SMOTE过采样或集成学习方法")

print("\n" + "=" * 60)
print("建模和评估完成！")
print("=" * 60)
