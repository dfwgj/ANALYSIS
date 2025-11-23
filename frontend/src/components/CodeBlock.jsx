import React from 'react';

const CodeBlock = ({ code, language = 'python' }) => {
  return (
    <div className="relative rounded-lg overflow-hidden bg-gray-900 dark:bg-[#1e1e1e] border border-gray-700 dark:border-white/10 font-mono text-sm shadow-lg my-4">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 dark:bg-[#252526] border-b border-gray-700 dark:border-white/5">
        <span className="text-xs text-gray-400">{language.toUpperCase()}</span>
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500/20" />
          <div className="w-3 h-3 rounded-full bg-yellow-500/20" />
          <div className="w-3 h-3 rounded-full bg-green-500/20" />
        </div>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-gray-300">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;
