import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import PredictionPage from './pages/PredictionPage';

import { ThemeProvider } from './context/ThemeContext';
import ThemeToggle from './components/ThemeToggle';

function App() {
  return (
    <ThemeProvider>
      <ThemeToggle />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<PredictionPage />} />
      </Routes>
    </ThemeProvider>
  );
}

export default App;
