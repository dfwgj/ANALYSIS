import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Droplet, Zap, Shield } from 'lucide-react';

// 英文特征名到显示标签的映射
const FEATURE_LABELS = {
  'HGB': '血红蛋白(HGB)',
  'HCT': '红细胞压积(HCT)',
  'RBC': '红细胞计数(RBC)',
  'RDW': '红细胞分布宽度(RDW)',
  'MCH': '平均红细胞血红蛋白(MCH)',
  'MCHC': '平均红细胞血红蛋白浓度(MCHC)',
  'MCV': '平均红细胞体积(MCV)',
  'PLT': '血小板计数(PLT)',
  'PCT': '血小板压积(PCT)',
  'PDW': '血小板分布宽度(PDW)',
  'LY#': '淋巴细胞计数(LY#)',
  'NE#': '中性粒细胞计数(NE#)',
  'SD': '血清铁(SD)',
  'TSD': '总铁结合力(TSD)',
  'FOLATE': '叶酸(FOLATE)'
};

const FEATURE_GROUPS = {
  "红细胞相关 (Red Blood Cells)": ['HGB', 'HCT', 'RBC', 'RDW', 'MCH', 'MCHC', 'MCV'],
  "白细胞相关 (White Blood Cells)": ['LY#', 'NE#'],
  "血小板相关 (Platelets)": ['PLT', 'PCT', 'PDW'],
  "生化指标 (Serum & Biochemical)": ['SD', 'TSD', 'FOLATE']
};

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    HGB: '', HCT: '', RBC: '', RDW: '', MCH: '', MCHC: '', MCV: '',
    PLT: '', PCT: '', PDW: '',
    'LY#': '', 'NE#': '',
    SD: '', TSD: '', FOLATE: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // 将表单数据转换为数值
    const numericData = {};
    for (const key in formData) {
      if (formData[key] !== '') {
        numericData[key] = parseFloat(formData[key]);
      }
    }
    onSubmit(numericData);
  };

  const getIcon = (group) => {
    switch(group) {
      case "白细胞相关 (White Blood Cells)": return <Shield className="w-5 h-5 text-purple-600 dark:text-neon-purple" />;
      case "红细胞相关 (Red Blood Cells)": return <Droplet className="w-5 h-5 text-red-500" />;
      case "血小板相关 (Platelets)": return <Activity className="w-5 h-5 text-yellow-500 dark:text-yellow-400" />;
      case "生化指标 (Serum & Biochemical)": return <Zap className="w-5 h-5 text-green-500 dark:text-green-400" />;
      default: return <Activity />;
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {Object.entries(FEATURE_GROUPS).map(([group, features]) => (
        <motion.div 
          key={group}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-6 relative overflow-hidden"
        >
          <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-blue-500 to-transparent dark:from-neon-blue dark:to-transparent opacity-50" />
          <h3 className="text-xl font-tech text-blue-600 dark:text-neon-blue mb-4 flex items-center gap-2">
            {getIcon(group)} {group}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map(feature => (
              <div key={feature} className="relative group">
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1 font-tech tracking-wider">{FEATURE_LABELS[feature]}</label>
                <input
                  type="number"
                  step="any"
                  name={feature}
                  value={formData[feature] || ''}
                  onChange={handleChange}
                  required
                  className="w-full bg-gray-50 dark:bg-dark-bg/50 border border-gray-300 dark:border-white/10 rounded px-3 py-2 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-neon-blue focus:outline-none focus:shadow-[0_0_10px_rgba(0,243,255,0.3)] transition-all"
                  placeholder="0.00"
                />
                <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-500 dark:bg-neon-blue group-hover:w-full transition-all duration-300" />
              </div>
            ))}
          </div>
        </motion.div>
      ))}

      <motion.button
        whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(0, 243, 255, 0.5)" }}
        whileTap={{ scale: 0.98 }}
        type="submit"
        disabled={isLoading}
        className="w-full py-4 bg-blue-600 hover:bg-blue-700 dark:bg-gradient-to-r dark:from-cyan-500 dark:to-blue-600 text-white font-tech font-bold text-xl rounded-lg shadow-lg relative overflow-hidden group transition-all"
      >
        <span className="relative z-10">{isLoading ? '正在分析...' : '开始智能诊断'}</span>
        <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
      </motion.button>
    </form>
  );
};

export default PredictionForm;
