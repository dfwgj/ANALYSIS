""""
贫血数据集分析与建模
步骤2：数据清洗和特征工程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import sys
warnings.filterwarnings('ignore')

# 设置中文字体和UTF-8编码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")
output_path.mkdir(exist_ok=True)

print("=" * 60)
print("贫血数据集 - 数据清洗和特征工程")
print("=" * 60)

# 加载数据
print("\n【1】加载原始数据...")
df = pd.read_csv(output_path / "raw_data.csv")
print(f"原始数据形状: {df.shape}")

# 确定特征和标签
# 修正版：移除数据泄露特征
feature_cols = [
    col for col in df.columns
    if not col.endswith('_Class')
    and col != 'All_Class'
    and col not in ['Folate_anemia_class', 'B12_Anemia_class']  # ⭐ 移除泄露特征
]
print(f"\n特征列数量: {len(feature_cols)}")
print(f"特征列: {feature_cols}")
print("\n⚠️  已移除泄露特征: Folate_anemia_class, B12_Anemia_class")

# 根据原始论文，All_Class是主要的5类分类标签
label_col = 'All_Class'
print(f"\n目标标签列: {label_col}")
print("标签分布:")
print(df[label_col].value_counts().sort_index())

# 数据清洗
print("\n【2】数据清洗...")

# 2.1 删除重复值
print(f"\n删除重复值前的行数: {len(df)}")
# drop_duplicates()函数用于删除DataFrame中的重复行
df_clean = df.drop_duplicates()
print(f"删除重复值后的行数: {len(df_clean)}")
print(f"删除了 {len(df) - len(df_clean)} 行重复数据")

# 2.2 检查并删除异常值
print("\n【3】检测异常值...")
print("根据论文描述，原始数据集(15300样本)已经是经过专家清洗后的数据。")
print("论文中提到的'剔除参考范围外数据'是指在构建这15300条数据之前的预处理步骤。")
print("如果我们现在再次使用'健康人参考范围'进行过滤，会将所有患病（贫血）样本作为异常值删除！")
print("例如：贫血患者的HGB必然低于正常范围，这是疾病特征而非噪音。")
print("因此，跳过基于参考范围的删除步骤，保留所有样本以供模型学习疾病模式。")

# 仅保留去重后的数据
print(f"最终保留数据形状: {df_clean.shape}")

# 特征分析
print("\n【4】特征相关性分析...")
correlation_matrix = df_clean[feature_cols].corr()

# 找出高度相关的特征对（相关系数 > 0.9）
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr_val
            ))

if high_corr_pairs:
    print("高度相关的特征对（|r| > 0.9）:")
    for col1, col2, corr in high_corr_pairs:
        print(f"  {col1} - {col2}: {corr:.3f}")
else:
    print("没有高度相关的特征对（|r| > 0.9）")

# 与目标变量的相关性
print("\n【5】与目标变量的相关性分析...")
target_corr = df_clean[feature_cols + [label_col]].corr()[label_col].drop(label_col).sort_values(key=abs, ascending=False)
print("与目标变量最相关的10个特征:")
print(target_corr.head(10))

# 数据变换
print("\n【6】数据变换...")

# 6.1 分离特征和标签
X = df_clean[feature_cols]
y = df_clean[label_col]

print(f"特征矩阵形状: {X.shape}")
print(f"标签向量形状: {y.shape}")

# 6.2 Min-Max归一化（根据论文要求）
print("\n进行Min-Max归一化...")
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)
print("归一化完成")

# 6.3 检查数据平衡性
print("\n【7】数据平衡性分析...")
class_distribution = y.value_counts().sort_index()
print("类别分布:")
for cls, count in class_distribution.items():
    print(f"  类别 {cls}: {count:5d} 样本 ({count/len(y)*100:5.2f}%)")

# 计算不平衡率
imbalance_ratio = max(class_distribution) / min(class_distribution)
print(f"\n数据不平衡比例: {imbalance_ratio:.2f}:1")
print(f"多数类是少数类的 {imbalance_ratio:.1f} 倍")

# 数据集划分
print("\n【8】数据集划分...")

# 使用分层抽样确保训练集和测试集中各类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # 分层抽样
    shuffle=True
)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")

print("\n训练集类别分布:")
train_dist = y_train.value_counts().sort_index()
for cls, count in train_dist.items():
    print(f"  类别 {cls}: {count:5d} 样本 ({count/len(y_train)*100:5.2f}%)")

print("\n测试集类别分布:")
test_dist = y_test.value_counts().sort_index()
for cls, count in test_dist.items():
    print(f"  类别 {cls}: {count:5d} 样本 ({count/len(y_test)*100:5.2f}%)")

# 保存处理后的数据
print("\n【9】保存处理后的数据...")

# 保存训练集和测试集
X_train.to_csv(output_path / "X_train.csv", index=False)
X_test.to_csv(output_path / "X_test.csv", index=False)
y_train.to_csv(output_path / "y_train.csv", index=False)
y_test.to_csv(output_path / "y_test.csv", index=False)
# 保存归一化器
import pickle
with open(output_path / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

# 保存特征列名和标签列名
with open(output_path / "feature_cols.txt", 'w', encoding='utf-8') as f:
    for col in feature_cols:
        f.write(col + '\n')

with open(output_path / "label_col.txt", 'w', encoding='utf-8') as f:
    f.write(label_col)

# 保存清洗后的数据报告
with open(output_path / "数据清洗报告.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 60 + '\n')
    f.write("贫血数据集 - 数据清洗和特征工程报告\n")
    f.write("=" * 60 + '\n\n')

    f.write(f"原始数据: {df.shape}\n")
    f.write(f"清洗后数据: {df_clean.shape}\n")
    f.write(f"删除样本数: {len(df) - len(df_clean)}\n\n")

    f.write("数据特征:\n")
    for i, col in enumerate(feature_cols, 1):
        f.write(f"  {i:2d}. {col}\n")
    f.write(f"\n标签: {label_col}\n")

    f.write(f"\n类别分布:\n")
    for cls, count in class_distribution.items():
        f.write(f"  类别 {cls}: {count:5d} 样本 ({count/len(y)*100:5.2f}%)\n")

    f.write(f"\n数据不平衡比例: {imbalance_ratio:.2f}:1\n")

    f.write(f"\n训练集大小: {len(X_train)}\n")
    f.write(f"测试集大小: {len(X_test)}\n")

print("\n[SUCCESS] 数据清洗完成！")
print(f"保存位置: {output_path}")
print("=" * 60)
