"""
贫血数据集分析与建模
步骤1：数据加载与探索
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# 设置中文字体和UTF-8编码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
if sys.version_info.major < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

# 创建输出目录
output_path = Path("E:\\Users\\DF\\Desktop\\机器学习大作业\\output")
output_path.mkdir(exist_ok=True)

print("=" * 60)
print("贫血数据集 - 数据探索阶段")
print("=" * 60)

# 加载数据
print("\n【1】加载数据...")
file_path = "E:\\Users\\DF\\Desktop\\机器学习大作业\\贫血数据集\\SKILICARSLAN_Anemia_DataSet.xlsx"

try:
    df = pd.read_excel(file_path)
    print(f"[SUCCESS] 数据加载成功！")
    print(f"数据形状: {df.shape}")
except Exception as e:
    print(f"[ERROR] 数据加载失败: {e}")
    exit()

print("\n【2】数据基本信息...")
print("-" * 60)
print(f"数据形状: {df.shape}")
print(f"总行数: {df.shape[0]:,}")
print(f"总列数: {df.shape[1]}")

print("\n【3】列名信息...")
print("-" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n【4】数据类型检查...")
print("-" * 60)
print(df.dtypes)

print("\n【5】前5行数据预览...")
print("-" * 60)
print(df.head())

print("\n【6】后5行数据预览...")
print("-" * 60)
print(df.tail())

print("\n【7】检查缺失值...")
print("-" * 60)
missing_info = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing_info / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数': missing_info,
    '缺失率(%)': missing_pct.round(2)
})
missing_df = missing_df[missing_df['缺失数'] > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print("[INFO] 无缺失值")

print("\n【8】检查重复值...")
print("-" * 60)
duplicates = df.duplicated().sum()
print(f"重复行数: {duplicates}")
if duplicates > 0:
    print("重复行详情:")
    print(df[df.duplicated()])

print("\n【9】基本统计信息...")
print("-" * 60)
print(df.describe().round(2))

# count（计数）：非空值的数量。
# 用途：再次确认缺失情况。
# mean（均值）：数据的平均水平。
# 用途：了解数据的中心位置。
# std（标准差）：数据的离散程度。
# 用途：数值越大，说明数据波动越大，越不稳定。
# min（最小值）：数据的下限。
# 用途：检查是否有不合理的负数（例如年龄为 -5）。
# 25%（下四分位数）：低分段的分界线。
# 50%（中位数）：正中间的那个数。
# 用途：非常重要！ 对比 mean 和 50%。如果 mean 远大于 50%，说明数据右偏（有极大的值拉高了平均数，比如“我和马云平均资产过亿”）。
# 75%（上四分位数）：高分段的分界线。
# max（最大值）：数据的上限。
# 用途：检查是否有离谱的极大值（异常值）。
# 保存基本信息到文件

print("\n【10】保存探索结果...")
print("-" * 60)
with open(output_path / "数据探索报告.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 60)
    f.write("\n贫血数据集 - 探索报告\n")
    f.write("=" * 60)
    f.write(f"\n\n数据形状: {df.shape}\n")
    f.write(f"总行数: {df.shape[0]:,}\n")
    f.write(f"总列数: {df.shape[1]}\n\n")

    f.write("列名列表:\n")
    f.write("-" * 60 + "\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:2d}. {col}\n")

    f.write("\n\n数据类型:\n")
    f.write("-" * 60 + "\n")
    f.write(str(df.dtypes))

    f.write("\n\n缺失值统计:\n")
    f.write("-" * 60 + "\n")
    if len(missing_df) > 0:
        f.write(str(missing_df))
    else:
        f.write("无缺失值")

    f.write(f"\n\n重复值: {duplicates} 行\n")

    f.write("\n\n基本统计:\n")
    f.write("-" * 60 + "\n")
    f.write(str(df.describe().round(2)))

print(f"[SUCCESS] 报告已保存到: {output_path / '数据探索报告.txt'}")
print("\n" + "=" * 60)
print("数据探索完成！")
print("=" * 60)

# 保存完整的DataFrame供后续使用
df.to_csv(output_path / "raw_data.csv", index=False)
print(f"[SUCCESS] 原始数据已保存到: {output_path / 'raw_data.csv'}")
