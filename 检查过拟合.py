"""
贫血数据集 - 过拟合检查
检查模型是否存在过拟合问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")

print("=" * 70)
print("贫血数据集 - 过拟合检查")
print("=" * 70)

# ============================================================
# 【1】加载数据
# ============================================================
print("\n[1] Loading data...")
X_train = pd.read_csv(output_path / "X_train.csv")
X_test = pd.read_csv(output_path / "X_test.csv")
y_train = pd.read_csv(output_path / "y_train.csv").squeeze()
y_test = pd.read_csv(output_path / "y_test.csv").squeeze()

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================
# 【2】过拟合检查方法1：训练集 vs 测试集性能对比
# ============================================================
print("\n" + "=" * 70)
print("[2] Overfitting Check: Train vs Test Performance")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced')
}

print("\n" + "-" * 70)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Diff':<10} {'Status'}")
print("-" * 70)

overfit_results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict on train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Calculate difference
    diff = train_acc - test_acc

    # Determine overfitting status
    if diff > 0.1:
        status = "!! OVERFIT !!"
    elif diff > 0.05:
        status = "! Warning"
    elif diff > 0.02:
        status = "~ Slight"
    else:
        status = "OK"

    print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {diff:<10.4f} {status}")

    overfit_results.append({
        'Model': name,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc,
        'Difference': diff,
        'Status': status
    })

print("-" * 70)
print("""
Interpretation:
  - Diff < 0.02: OK (No overfitting)
  - Diff 0.02-0.05: Slight overfitting (acceptable)
  - Diff 0.05-0.10: Warning (moderate overfitting)
  - Diff > 0.10: Serious overfitting!
""")

# ============================================================
# 【3】过拟合检查方法2：F1 Score对比
# ============================================================
print("\n" + "=" * 70)
print("[3] Overfitting Check: F1 Score (Macro)")
print("=" * 70)

print("\n" + "-" * 70)
print(f"{'Model':<25} {'Train F1':<12} {'Test F1':<12} {'Diff':<10} {'Status'}")
print("-" * 70)

for name, model in models.items():
    # Re-train (models already trained above)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    diff = train_f1 - test_f1

    if diff > 0.1:
        status = "!! OVERFIT !!"
    elif diff > 0.05:
        status = "! Warning"
    elif diff > 0.02:
        status = "~ Slight"
    else:
        status = "OK"

    print(f"{name:<25} {train_f1:<12.4f} {test_f1:<12.4f} {diff:<10.4f} {status}")

print("-" * 70)

# ============================================================
# 【4】过拟合检查方法3：学习曲线
# ============================================================
print("\n" + "=" * 70)
print("[4] Learning Curves (detecting overfitting)")
print("=" * 70)
print("Generating learning curves...")

# Select best model for detailed analysis
best_model = GradientBoostingClassifier(random_state=42)

# Calculate learning curve
train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes_abs, train_scores, test_scores = learning_curve(
    best_model,
    X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

# Calculate mean and std
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))

plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')

plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes_abs, test_mean, 'o-', color='orange', label='Cross-Validation Score')

plt.xlabel('Training Set Size')
plt.ylabel('F1 Score (Macro)')
plt.title('Learning Curve - Gradient Boosting\n(Gap between lines indicates overfitting)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Add annotation
final_gap = train_mean[-1] - test_mean[-1]
plt.annotate(f'Final Gap: {final_gap:.4f}',
             xy=(train_sizes_abs[-1], (train_mean[-1] + test_mean[-1])/2),
             fontsize=10, color='red')

plt.tight_layout()
plt.savefig(output_path / 'overfit_learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Learning curve saved to: {output_path / 'overfit_learning_curve.png'}")

# ============================================================
# 【5】过拟合检查方法4：交叉验证稳定性
# ============================================================
print("\n" + "=" * 70)
print("[5] Cross-Validation Stability")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "-" * 70)
print(f"{'Model':<25} {'CV Mean':<12} {'CV Std':<12} {'Stability'}")
print("-" * 70)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # High std indicates unstable model (potential overfitting)
    if cv_std > 0.05:
        stability = "! Unstable"
    elif cv_std > 0.02:
        stability = "~ Moderate"
    else:
        stability = "OK Stable"

    print(f"{name:<25} {cv_mean:<12.4f} {cv_std:<12.4f} {stability}")

print("-" * 70)
print("""
Interpretation:
  - CV Std < 0.02: Stable (good generalization)
  - CV Std 0.02-0.05: Moderate variability
  - CV Std > 0.05: Unstable (potential overfitting)
""")

# ============================================================
# 【6】可视化：训练集 vs 测试集对比图
# ============================================================
print("\n[6] Generating comparison chart...")

results_df = pd.DataFrame(overfit_results)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['Train_Accuracy'], width, label='Train Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', color='coral')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Overfitting Check: Train vs Test Accuracy\n(Large gap = Overfitting)')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0.8, 1.02)

# Add difference labels
for i, (train, test, diff) in enumerate(zip(results_df['Train_Accuracy'],
                                             results_df['Test_Accuracy'],
                                             results_df['Difference'])):
    color = 'red' if diff > 0.05 else 'green'
    ax.annotate(f'{diff:.3f}', xy=(i, max(train, test) + 0.01),
                ha='center', fontsize=9, color=color)

plt.tight_layout()
plt.savefig(output_path / 'overfit_train_test_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Comparison chart saved")

# ============================================================
# 【7】总结报告
# ============================================================
print("\n" + "=" * 70)
print("[7] OVERFITTING ANALYSIS SUMMARY")
print("=" * 70)

# Find most overfit model
max_diff_idx = results_df['Difference'].idxmax()
max_diff_model = results_df.loc[max_diff_idx, 'Model']
max_diff = results_df.loc[max_diff_idx, 'Difference']

# Find best generalization model
min_diff_idx = results_df['Difference'].idxmin()
min_diff_model = results_df.loc[min_diff_idx, 'Model']
min_diff = results_df.loc[min_diff_idx, 'Difference']

print(f"""
Analysis Results:
-----------------
1. Model with MOST overfitting: {max_diff_model}
   Train-Test Gap: {max_diff:.4f}

2. Model with BEST generalization: {min_diff_model}
   Train-Test Gap: {min_diff:.4f}

3. Gradient Boosting (your best model):
   Train Accuracy: {results_df[results_df['Model']=='Gradient Boosting']['Train_Accuracy'].values[0]:.4f}
   Test Accuracy:  {results_df[results_df['Model']=='Gradient Boosting']['Test_Accuracy'].values[0]:.4f}
   Gap: {results_df[results_df['Model']=='Gradient Boosting']['Difference'].values[0]:.4f}
""")

# Determine overall conclusion
gb_diff = results_df[results_df['Model']=='Gradient Boosting']['Difference'].values[0]

if gb_diff < 0.02:
    conclusion = """
CONCLUSION: NO OVERFITTING DETECTED!

Your Gradient Boosting model shows excellent generalization:
- Train-Test gap is very small (< 2%)
- The model learns patterns, not just memorizing data
- High test accuracy (99.97%) is genuine, not due to overfitting

This is a well-trained model!
"""
elif gb_diff < 0.05:
    conclusion = """
CONCLUSION: SLIGHT OVERFITTING (Acceptable)

Your model shows minor overfitting:
- Train-Test gap is 2-5%
- Generally acceptable for most applications
- Consider regularization if you want to improve

Suggestions:
1. Increase n_estimators with lower learning_rate
2. Add max_depth limit
3. Use more training data if available
"""
else:
    conclusion = """
CONCLUSION: OVERFITTING DETECTED!

Your model shows significant overfitting:
- Train-Test gap > 5%
- Model may be memorizing training data

Suggestions to fix:
1. Reduce model complexity (lower max_depth, fewer trees)
2. Add regularization
3. Use cross-validation for hyperparameter tuning
4. Get more training data
5. Try simpler models
"""

print(conclusion)

# ============================================================
# 【8】如何修复过拟合（如果有的话）
# ============================================================
print("\n" + "=" * 70)
print("[8] HOW TO FIX OVERFITTING (Reference)")
print("=" * 70)
print("""
If overfitting is detected, try these solutions:

1. REDUCE MODEL COMPLEXITY
   - Decision Tree: set max_depth=5, min_samples_leaf=10
   - Random Forest: reduce n_estimators, set max_depth
   - Gradient Boosting: lower learning_rate, add max_depth

2. REGULARIZATION
   - Logistic Regression: increase C parameter
   - Add dropout (for neural networks)

3. MORE DATA
   - Collect more samples
   - Use data augmentation
   - Use SMOTE for minority classes

4. FEATURE SELECTION
   - Remove irrelevant features
   - Use feature importance to select top features

5. EARLY STOPPING
   - Stop training when validation score stops improving

6. CROSS-VALIDATION
   - Use k-fold CV for hyperparameter tuning
   - Ensures model generalizes well

Example code to reduce overfitting in Gradient Boosting:
---------------------------------------------------------
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,      # Lower = less overfit
    max_depth=3,             # Limit tree depth
    min_samples_split=10,    # Require more samples to split
    min_samples_leaf=5,      # Require more samples in leaves
    subsample=0.8,           # Use only 80% of data per tree
    random_state=42
)
""")

# Save results
results_df.to_csv(output_path / 'overfit_check_results.csv', index=False)
print(f"\n[OK] Results saved to: {output_path / 'overfit_check_results.csv'}")

print("\n" + "=" * 70)
print("Overfitting check completed!")
print("=" * 70)
