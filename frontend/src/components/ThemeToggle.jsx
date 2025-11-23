import React from 'react';
import { motion } from 'framer-motion';
import { Sun, Moon, Github } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <>
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={toggleTheme}
        className="fixed top-6 right-6 z-50 p-3 rounded-full bg-white/10 backdrop-blur-md border border-white/20 shadow-lg hover:bg-white/20 transition-colors dark:bg-black/20 dark:border-white/10"
        aria-label="Toggle Theme"
      >
        <motion.div
          initial={false}
          animate={{ rotate: theme === 'dark' ? 0 : 180 }}
          transition={{ duration: 0.3 }}
        >
          {theme === 'dark' ? (
            <Moon className="w-6 h-6 text-neon-blue" />
          ) : (
            <Sun className="w-6 h-6 text-orange-500" />
          )}
        </motion.div>
      </motion.button>

      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => window.open('https://github.com/dfwgj/ANALYSIS', '_blank')}
        className="fixed top-20 right-6 z-50 p-3 rounded-full bg-white/10 backdrop-blur-md border border-white/20 shadow-lg hover:bg-white/20 transition-colors dark:bg-black/20 dark:border-white/10"
        aria-label="GitHub Repository"
        title="查看GitHub仓库"
      >
        <Github className="w-6 h-6 text-gray-700 dark:text-gray-200" />
      </motion.button>
    </>
  );
};

export default ThemeToggle;
