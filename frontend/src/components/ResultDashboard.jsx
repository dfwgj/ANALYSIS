import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, CheckCircle, Activity } from 'lucide-react';

const ResultDashboard = ({ result, onReset }) => {
  if (!result) return null;

  const { class_label, probability, all_probabilities } = result;
  
  const data = Object.entries(all_probabilities).map(([name, value]) => ({
    name: name.replace(' Anemia', '').replace(' Deficiency', ''),
    value: value * 100
  }));

  const isHealthy = class_label === "No Anemia";
  const color = isHealthy ? '#00f3ff' : '#bc13fe';

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-panel p-8 text-center space-y-8 border-2 border-blue-500/30 dark:border-neon-blue/30 shadow-lg dark:shadow-[0_0_30px_rgba(0,243,255,0.1)]"
    >
      <div className="space-y-2">
        <h2 className="text-2xl font-tech text-gray-900 dark:text-white mb-6 flex items-center gap-2">
        <Activity className="text-blue-600 dark:text-neon-blue" /> 诊断结果 (Diagnosis Result)
      </h2>

      <div className="mb-8 p-4 bg-gray-50 dark:bg-white/5 rounded-lg border border-gray-200 dark:border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-600 dark:text-gray-400">预测类别 (Predicted Class)</span>
          <span className="text-xl font-bold text-purple-600 dark:text-neon-purple">{class_label.toUpperCase()}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-600 dark:text-gray-400">置信度 (Confidence)</span>
          <span className="text-xl font-bold text-blue-600 dark:text-neon-blue">{(probability * 100).toFixed(2)}%</span>
        </div>
      </div>
        
        <div className="absolute -top-6 -right-12 animate-pulse-slow">
          {isHealthy ? <CheckCircle className="w-12 h-12 text-blue-600 dark:text-neon-blue" /> : <AlertTriangle className="w-12 h-12 text-purple-600 dark:text-neon-purple" />}
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color, #333)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} stroke="var(--axis-color, #666)" />
            <YAxis dataKey="name" type="category" stroke="var(--text-color, #fff)" width={100} tick={{fontSize: 10}} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'var(--tooltip-bg, #13131f)', borderColor: '#333', color: '#fff' }}
              itemStyle={{ color: '#00f3ff' }}
              formatter={(value) => [`${value.toFixed(2)}%`, 'Probability']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.name === class_label.replace(' Anemia', '').replace(' Deficiency', '') ? color : 'var(--bar-bg, #333)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onReset}
        className="w-full py-3 bg-gray-100 hover:bg-gray-200 dark:bg-white/10 dark:hover:bg-white/20 text-gray-900 dark:text-white rounded-lg transition-colors font-tech"
      >
        进行新的诊断 (New Diagnosis)
      </motion.button>
    </motion.div>
  );
};

export default ResultDashboard;
