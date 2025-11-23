import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import PredictionForm from '../components/PredictionForm';
import ResultDashboard from '../components/ResultDashboard';
import { predictAnemia, checkHealth } from '../api';
import { Activity, ArrowLeft } from 'lucide-react';

function PredictionPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  const handlePrediction = async (data) => {
    setLoading(true);
    try {
      const res = await predictAnemia(data);
      // Simulate a "processing" delay for effect
      setTimeout(() => {
        setResult(res);
        setLoading(false);
      }, 1500);
    } catch (error) {
      alert("Prediction failed. Ensure backend is running.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 relative">
      {/* Header */}
      <header className="flex justify-between items-center mb-12 glass-panel p-4">
        <div className="flex items-center justify-between w-full">
          <Link to="/">
            <motion.button
              whileHover={{ x: -5 }}
              className="flex items-center gap-2 text-blue-600 dark:text-neon-blue hover:text-purple-600 dark:hover:text-neon-purple transition-colors"
            >
              <ArrowLeft size={20} /> 返回首页 (Back)
            </motion.button>
          </Link>
          <div className="text-center">
            <h1 className="text-3xl font-tech font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-cyan-600 dark:from-cyan-400 dark:to-blue-500">
              智能诊断系统
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 tracking-widest">AI ANALYSIS MODULE</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-500 shadow-[0_0_10px_#0f0]' : 'bg-red-500'}`} />
            <span className="text-xs font-mono text-gray-500 dark:text-gray-400">SYSTEM: {backendStatus.toUpperCase()}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto">
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.div
              key="form"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <PredictionForm onSubmit={handlePrediction} isLoading={loading} />
            </motion.div>
          ) : (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <ResultDashboard result={result} onReset={() => setResult(null)} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="mt-16 text-center text-gray-500 dark:text-gray-600 font-tech text-sm">
        <p>POWERED BY MACHINE LEARNING • v1.0.0</p>
      </footer>
    </div>
  );
}

export default PredictionPage;
