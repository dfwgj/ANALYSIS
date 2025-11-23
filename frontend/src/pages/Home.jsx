import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Activity, Database, Brain, Code, ChevronRight, BarChart2, FileText, Search } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import CodeBlock from '../components/CodeBlock';

const Home = () => {

  const classData = [
    { name: '健康 (Healthy)', value: 63.58 }, // Approx % based on typical balance or 1:1 after SMOTE
    { name: 'HGB贫血', value: 6.67 },
    { name: '缺铁性贫血', value: 27.45 },
    { name: '叶酸缺乏', value: 1.00 },
    { name: 'B12缺乏', value: 1.3 },
  ];
  const COLORS = ['#00f3ff', '#bc13fe', '#ff0055', '#ffaa00', '#00ff66'];

  const mlSteps = [
    { 
      title: "1. 数据探索", 
      icon: <Search />, 
      desc: "加载15,300条样本，检查缺失值与数据分布。",
      detail: "使用 pandas 加载 .xlsx 数据，分析 29 个特征列的统计信息。"
    },
    { 
      title: "2. 特征工程", 
      icon: <Database />, 
      desc: "清洗数据，筛选出15个关键特征，进行归一化。",
      detail: "剔除无关列，使用 MinMaxScaler 将数据缩放到 [0,1] 区间。"
    },
    { 
      title: "3. 模型训练", 
      icon: <Brain />, 
      desc: "应用 SMOTE 解决类别不平衡，训练梯度提升模型。",
      detail: "使用 SMOTE 过采样少数类，训练 GradientBoostingClassifier。"
    },
    { 
      title: "4. 模型评估", 
      icon: <BarChart2 />, 
      desc: "使用混淆矩阵和 F1-Score 评估模型性能。",
      detail: "在测试集上验证，确保高准确率和泛化能力。"
    },
  ];

  const codeSnippet1 = `
# 1. 数据加载与探索 (1_数据探索.py)
import pandas as pd

# 加载数据
file_path = "SKILICARSLAN_Anemia_DataSet.xlsx"
df = pd.read_excel(file_path)

# 检查缺失值
missing_info = df.isnull().sum().sort_values(ascending=False)
print(f"数据形状: {df.shape}") # (15300, 29)
  `;

  const codeSnippet2 = `
# 2. 特征工程 (2_数据清洗和特征工程.py)
from sklearn.preprocessing import MinMaxScaler

# 筛选特征 (排除 _Class 结尾的列)
feature_cols = [col for col in df.columns if not col.endswith('_Class') 
                and col != 'All_Class']

# Min-Max 归一化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(df[feature_cols])
  `;

  const codeSnippet3 = `
# 3. 建模 (3_建模和评估_SMOTE.py)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# SMOTE 过采样解决不平衡
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 训练梯度提升分类器
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)
  `;

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="h-screen flex flex-col items-center justify-center text-center relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1579154204601-01588f351e67?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-10" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-light-bg/80 to-light-bg dark:via-dark-bg/80 dark:to-dark-bg" />
        
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 max-w-4xl px-4"
        >
          <h1 className="text-6xl md:text-8xl font-tech font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-cyan-600 dark:from-cyan-400 dark:to-blue-500 drop-shadow-lg dark:drop-shadow-[0_0_20px_rgba(0,243,255,0.3)]">
            贫血智能诊断系统
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-12 font-light tracking-wide">
            基于机器学习的精准医疗辅助决策平台
          </p>
          
          <Link to="/predict">
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(0, 243, 255, 0.4)" }}
              whileTap={{ scale: 0.95 }}
              className="px-12 py-5 bg-blue-600 hover:bg-blue-700 dark:bg-gradient-to-r dark:from-cyan-500 dark:to-blue-600 rounded-full text-white font-tech font-bold text-xl flex items-center gap-3 mx-auto group shadow-lg transition-all"
            >
              立即体验 <ChevronRight className="group-hover:translate-x-1 transition-transform" />
            </motion.button>
          </Link>
        </motion.div>
      </section>

      {/* Data Visualization Section */}
      <section className="py-20 px-4 bg-white/50 dark:bg-panel-bg/30">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-tech text-blue-600 dark:text-neon-blue mb-12 text-center flex items-center justify-center gap-3">
            <BarChart2 /> 数据集洞察 (Dataset Insights)
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <motion.div 
              whileHover={{ y: -5 }}
              className="glass-panel p-8"
            >
              <h3 className="text-xl font-tech text-gray-900 dark:text-white mb-6 text-center">样本类别分布 (SMOTE平衡前)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={classData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {classData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--tooltip-bg, #13131f)', borderColor: '#333' }}
                      itemStyle={{ color: '#fff' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap justify-center gap-4 mt-4">
                {classData.map((entry, index) => (
                  <div key={entry.name} className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index] }} />
                    {entry.name}
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ y: -5 }}
              className="glass-panel p-8 flex flex-col justify-center"
            >
              <h3 className="text-xl font-tech text-gray-900 dark:text-white mb-6 text-center">数据集统计</h3>
              <div className="space-y-6">
                <div className="flex justify-between items-center border-b border-gray-200 dark:border-white/10 pb-2">
                  <span className="text-gray-600 dark:text-gray-400">总样本数</span>
                  <span className="text-2xl font-bold text-blue-600 dark:text-neon-blue">15,300</span>
                </div>
                <div className="flex justify-between items-center border-b border-gray-200 dark:border-white/10 pb-2">
                  <span className="text-gray-600 dark:text-gray-400">特征数量</span>
                  <span className="text-2xl font-bold text-purple-600 dark:text-neon-purple">24</span>
                </div>
                <div className="flex justify-between items-center border-b border-gray-200 dark:border-white/10 pb-2">
                  <span className="text-gray-600 dark:text-gray-400">诊断类别</span>
                  <span className="text-2xl font-bold text-gray-900 dark:text-white">5</span>
                </div>
                <div className="flex justify-between items-center border-b border-gray-200 dark:border-white/10 pb-2">
                  <span className="text-gray-600 dark:text-gray-400">数据来源</span>
                  <span className="text-sm text-gray-500 dark:text-gray-300">SKILICARSLAN Dataset</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ML Process Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-tech text-purple-600 dark:text-neon-purple mb-16 text-center flex items-center justify-center gap-3">
            <Brain /> 机器学习全流程 (ML Pipeline)
          </h2>
          <div className="grid md:grid-cols-4 gap-6">
            {mlSteps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
                viewport={{ once: true }}
                className="glass-panel p-6 relative group hover:bg-blue-50 dark:hover:bg-white/5 transition-colors"
              >
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-12 h-12 bg-white dark:bg-dark-bg border-2 border-purple-500 dark:border-neon-purple rounded-full flex items-center justify-center text-purple-600 dark:text-neon-purple group-hover:scale-110 transition-transform shadow-lg dark:shadow-[0_0_15px_rgba(188,19,254,0.3)]">
                  {step.icon}
                </div>
                <div className="mt-8 text-center">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{step.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">{step.desc}</p>
                  <p className="text-xs text-gray-500">{step.detail}</p>
                </div>
                {index < mlSteps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-0.5 bg-gradient-to-r from-purple-500 to-transparent dark:from-neon-purple dark:to-transparent" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Code Section */}
      <section className="py-20 px-4 bg-white/50 dark:bg-panel-bg/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-tech text-gray-900 dark:text-white mb-12 text-center flex items-center justify-center gap-3">
            <Code /> 核心代码实现 (Core Code)
          </h2>
          
          <div className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <h3 className="text-xl text-blue-600 dark:text-neon-blue mb-4 font-tech">1. 数据探索与加载</h3>
              <CodeBlock code={codeSnippet1} />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <h3 className="text-xl text-blue-600 dark:text-neon-blue mb-4 font-tech">2. 特征工程与归一化</h3>
              <CodeBlock code={codeSnippet2} />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <h3 className="text-xl text-blue-600 dark:text-neon-blue mb-4 font-tech">3. SMOTE平衡与模型训练</h3>
              <CodeBlock code={codeSnippet3} />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 text-center text-gray-500 dark:text-gray-600 font-tech text-sm border-t border-gray-200 dark:border-white/5">
        <p>贫血智能诊断系统 (ANEMIA INTELLIGENT DIAGNOSIS SYSTEM) • 2025</p>
      </footer>
    </div>
  );
};

export default Home;
