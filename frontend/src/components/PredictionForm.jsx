import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Droplet, Zap, Shield } from 'lucide-react';

const FEATURE_GROUPS = {
  "基本信息 (Basic Info)": ["GENDER"],
  "白细胞相关 (White Blood Cells)": ["WBC", "NE#", "LY#", "MO#", "EO#", "BA#"],
  "红细胞相关 (Red Blood Cells)": ["RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW"],
  "血小板相关 (Platelets)": ["PLT", "MPV", "PCT", "PDW"],
  "生化指标 (Serum & Biochemical)": ["SD", "SDTSD", "TSD", "FERRITTE", "FOLATE", "B12"]
};

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    GENDER: '1', WBC: '', NE: '', LY: '', MO: '', EO: '', BA: '',
    RBC: '', HGB: '', HCT: '', MCV: '', MCH: '', MCHC: '', RDW: '',
    PLT: '', MPV: '', PCT: '', PDW: '',
    SD: '', SDTSD: '', TSD: '', FERRITTE: '', FOLATE: '', B12: ''
  });

  // Fix key names to match backend expectation (e.g., NE# -> NE#)
  // React state keys cannot easily contain #, so we map them on submit or use bracket notation
  // For simplicity in this demo, I'll use the exact keys from the backend in the state, 
  // but we need to be careful with input names.
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Convert to numbers
    const numericData = {};
    for (const key in formData) {
      numericData[key] = parseFloat(formData[key]);
    }
    onSubmit(numericData);
  };

  const getIcon = (group) => {
    switch(group) {
      case "基本信息 (Basic Info)": return <Shield className="w-5 h-5 text-blue-600 dark:text-neon-blue" />;
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
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1 font-tech tracking-wider">{feature}</label>
                {feature === 'GENDER' ? (
                  <select
                    name={feature}
                    value={formData[feature]}
                    onChange={handleChange}
                    className="w-full bg-gray-50 dark:bg-dark-bg/50 border border-gray-300 dark:border-white/10 rounded px-3 py-2 text-gray-900 dark:text-white focus:border-blue-500 dark:focus:border-neon-blue focus:outline-none focus:shadow-[0_0_10px_rgba(0,243,255,0.3)] transition-all appearance-none"
                  >
                    <option value="1">男 (Male)</option>
                    <option value="0">女 (Female)</option>
                  </select>
                ) : (
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
                )}
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
