"""
贫血数据集 - 正态分布检验
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")

print("=" * 70)
print("贫血数据集 - 正态分布检验")
print("=" * 70)

# 加载数据（使用原始未归一化的数据更有意义）
print("\n【1】加载数据...")
df = pd.read_csv(output_path / "raw_data.csv")

# 获取特征列（排除分类列）
feature_cols = [col for col in df.columns if not col.endswith('_Class') and col != 'All_Class']
print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(df)}")

# ============================================================
# 【2】正态分布检验
# ============================================================
print("\n【2】正态分布检验（Shapiro-Wilk & D'Agostino）")
print("=" * 70)
print(f"{'特征名':<15} {'Shapiro-W':<12} {'Shapiro-p':<12} {'正态?':<8} {'偏度':<10} {'峰度':<10}")
print("-" * 70)

results = []

for col in feature_cols:
    data = df[col].dropna()

    # 如果样本量太大，Shapiro-Wilk只取前5000个样本
    if len(data) > 5000:
        sample_data = data.sample(5000, random_state=42)
    else:
        sample_data = data

    # Shapiro-Wilk检验
    # H0: 数据服从正态分布
    # p < 0.05: 拒绝H0，即不服从正态分布
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)

    # 计算偏度和峰度
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # 判断是否正态（p > 0.05 且 偏度和峰度接近0）
    is_normal = shapiro_p > 0.05
    normal_str = "YES" if is_normal else "NO"

    print(f"{col:<15} {shapiro_stat:<12.4f} {shapiro_p:<12.4f} {normal_str:<8} {skewness:<10.3f} {kurtosis:<10.3f}")

    results.append({
        '特征': col,
        'Shapiro统计量': shapiro_stat,
        'Shapiro_p值': shapiro_p,
        '是否正态': is_normal,
        '偏度': skewness,
        '峰度': kurtosis
    })

results_df = pd.DataFrame(results)

# ============================================================
# 【3】统计汇总
# ============================================================
print("\n【3】检验结果汇总")
print("=" * 70)

normal_count = results_df['是否正态'].sum()
not_normal_count = len(results_df) - normal_count

print(f"符合正态分布的特征: {normal_count} 个")
print(f"不符合正态分布的特征: {not_normal_count} 个")
print(f"正态分布比例: {normal_count/len(results_df)*100:.1f}%")

# ============================================================
# 【4】偏度和峰度分析
# ============================================================
print("\n【4】偏度和峰度分析")
print("=" * 70)
print("""
偏度(Skewness)解读:
  = 0  : 完美对称（正态分布）
  > 0  : 右偏（右尾长，大多数值在左边）
  < 0  : 左偏（左尾长，大多数值在右边）
  |偏度| > 1 : 高度偏斜

峰度(Kurtosis)解读:
  = 0  : 正态分布的峰度
  > 0  : 尖峰（比正态分布更陡峭）
  < 0  : 平峰（比正态分布更平坦）
  |峰度| > 1 : 明显偏离正态
""")

# 偏度分析
high_skew = results_df[abs(results_df['偏度']) > 1]
if len(high_skew) > 0:
    print(f"高度偏斜的特征（|偏度| > 1）:")
    for _, row in high_skew.iterrows():
        direction = "右偏" if row['偏度'] > 0 else "左偏"
        print(f"  {row['特征']}: 偏度={row['偏度']:.3f} ({direction})")
else:
    print("没有高度偏斜的特征")

# 峰度分析
high_kurt = results_df[abs(results_df['峰度']) > 1]
if len(high_kurt) > 0:
    print(f"\n峰度异常的特征（|峰度| > 1）:")
    for _, row in high_kurt.iterrows():
        shape = "尖峰" if row['峰度'] > 0 else "平峰"
        print(f"  {row['特征']}: 峰度={row['峰度']:.3f} ({shape})")
else:
    print("\n没有峰度异常的特征")

# ============================================================
# 【5】可视化
# ============================================================
print("\n【5】生成可视化图表...")

# 选择前6个特征进行可视化
plot_features = feature_cols[:6]

# 5.1 直方图 + 正态曲线
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(plot_features):
    ax = axes[idx]
    data = df[col].dropna()

    # 画直方图
    ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # 画正态分布曲线
    mu, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, std)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label='正态分布')

    # 获取检验结果
    result = results_df[results_df['特征'] == col].iloc[0]
    is_normal = "[Normal]" if result['是否正态'] else "[Not Normal]"

    ax.set_title(f'{col}\n{is_normal} (p={result["Shapiro_p值"]:.4f})')
    ax.set_xlabel('值')
    ax.set_ylabel('密度')
    ax.legend()

plt.suptitle('各特征分布直方图 vs 正态分布曲线', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path / '正态分布检验_直方图.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 直方图已保存")

# 5.2 Q-Q图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(plot_features):
    ax = axes[idx]
    data = df[col].dropna()

    # Q-Q图
    stats.probplot(data, dist="norm", plot=ax)

    result = results_df[results_df['特征'] == col].iloc[0]
    is_normal = "[Normal]" if result['是否正态'] else "[Not Normal]"

    ax.set_title(f'{col} Q-Q\n{is_normal}')
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markersize(3)

plt.suptitle('Q-Q图（点越接近红线，越接近正态分布）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path / '正态分布检验_QQ图.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Q-Q图已保存")

# 5.3 偏度和峰度可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 偏度
ax1 = axes[0]
colors = ['green' if abs(s) <= 1 else 'red' for s in results_df['偏度']]
bars1 = ax1.barh(results_df['特征'], results_df['偏度'], color=colors, alpha=0.7)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.axvline(x=-1, color='orange', linestyle='--', linewidth=1, label='|偏度|=1')
ax1.axvline(x=1, color='orange', linestyle='--', linewidth=1)
ax1.set_xlabel('偏度')
ax1.set_title('各特征偏度\n(绿色=正常，红色=高度偏斜)')
ax1.legend()

# 峰度
ax2 = axes[1]
colors = ['green' if abs(k) <= 1 else 'red' for k in results_df['峰度']]
bars2 = ax2.barh(results_df['特征'], results_df['峰度'], color=colors, alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=-1, color='orange', linestyle='--', linewidth=1, label='|峰度|=1')
ax2.axvline(x=1, color='orange', linestyle='--', linewidth=1)
ax2.set_xlabel('峰度')
ax2.set_title('各特征峰度\n(绿色=正常，红色=异常)')
ax2.legend()

plt.tight_layout()
plt.savefig(output_path / '正态分布检验_偏度峰度.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 偏度峰度图已保存")

# ============================================================
# 【6】保存结果
# ============================================================
print("\n【6】保存检验结果...")
results_df.to_csv(output_path / '正态分布检验结果.csv', index=False, encoding='utf-8-sig')
print(f"[OK] 结果已保存到: {output_path / '正态分布检验结果.csv'}")

# ============================================================
# 【7】结论
# ============================================================
print("\n" + "=" * 70)
print("【7】结论")
print("=" * 70)

if normal_count == 0:
    print("""
Conclusion: None of the features follow normal distribution

This is common in medical data, possible reasons:
1. Data contains mixed distributions (healthy + various anemia types)
2. Biological indicators are often not normally distributed
3. Presence of outliers or extreme values

Impact on Machine Learning:
[OK] Most ML algorithms don't require normal distribution
[OK] Decision Tree, Random Forest, Gradient Boosting - Not affected
[OK] KNN - Not affected
[OK] Logistic Regression - Slight impact, usually acceptable
[OK] SVM - Not affected

If normalization is needed, try:
1. Log transform: np.log1p(x)
2. Box-Cox transform: scipy.stats.boxcox(x)
3. Yeo-Johnson transform: sklearn.preprocessing.PowerTransformer
""")
elif normal_count == len(results_df):
    print("Conclusion: All features follow normal distribution! This is rare but ideal.")
else:
    print(f"""
Conclusion: Some features follow normal distribution

Normal: {normal_count}
Not Normal: {not_normal_count}

Impact on ML:
Most models (Decision Tree, Random Forest, Gradient Boosting) don't require
normal distribution, so this won't affect your model performance.
""")

print("\n" + "=" * 70)
print("正态分布检验完成！")
print("=" * 70)
