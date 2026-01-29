import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const RAGInterface = () => {
  const [pdfUrl, setPdfUrl] = useState('');
  const [questions, setQuestions] = useState(['', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [responseTime, setResponseTime] = useState(0);

  // Generate floating particles
  const particles = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    top: Math.random() * 100,
    duration: Math.random() * 10 + 5,
    delay: Math.random() * 5
  }));

  const addQuestion = () => {
    if (questions.length >= 15) {
      setError('Maximum 15 questions allowed');
      return;
    }
    setQuestions([...questions, '']);
  };

  const removeQuestion = (index) => {
    if (questions.length <= 5) {
      setError('Minimum 5 questions required');
      return;
    }
    setQuestions(questions.filter((_, i) => i !== index));
  };

  const updateQuestion = (index, value) => {
    const newQuestions = [...questions];
    newQuestions[index] = value;
    setQuestions(newQuestions);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const validQuestions = questions.filter(q => q.trim() !== '');
    
    if (validQuestions.length < 5 || validQuestions.length > 15) {
      setError('Please provide between 5 and 15 questions');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    const startTime = Date.now();

    try {
      const response = await fetch('http://localhost:8000/aibattle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pdf_url: pdfUrl,
          questions: validQuestions
        })
      });

      const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
      setResponseTime(elapsedTime);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults({ answers: data.answers, questions: validQuestions });
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050714] text-white relative overflow-hidden">
      {/* Animated Background */}
      <motion.div
        className="fixed inset-0 -z-10"
        animate={{
          background: [
            'radial-gradient(circle at 20% 50%, rgba(131, 56, 236, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(255, 0, 110, 0.1) 0%, transparent 50%)',
            'radial-gradient(circle at 80% 50%, rgba(131, 56, 236, 0.1) 0%, transparent 50%), radial-gradient(circle at 20% 80%, rgba(255, 0, 110, 0.1) 0%, transparent 50%)'
          ]
        }}
        transition={{ duration: 15, repeat: Infinity, repeatType: 'reverse' }}
      />

      {/* Grid Overlay */}
      <motion.div
        className="fixed inset-0 -z-10 opacity-30"
        style={{
          backgroundImage: 'linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }}
        animate={{ y: [0, 50] }}
        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
      />

      {/* Particles */}
      <div className="fixed inset-0 -z-10">
        {particles.map(particle => (
          <motion.div
            key={particle.id}
            className="absolute w-0.5 h-0.5 bg-cyan-400 rounded-full opacity-30"
            style={{ left: `${particle.left}%`, top: `${particle.top}%` }}
            animate={{
              y: [-100, -200, -100, 0],
              x: [0, 50, -30, 20, 0]
            }}
            transition={{
              duration: particle.duration,
              delay: particle.delay,
              repeat: Infinity
            }}
          />
        ))}
      </div>

      <div className="container mx-auto px-4 py-10 max-w-6xl">
        {/* Header */}
        <motion.header
          className="text-center mb-16"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <motion.div
            className="inline-block mb-6"
            animate={{
              filter: [
                'drop-shadow(0 0 20px rgba(0, 243, 255, 0.8))',
                'drop-shadow(0 0 40px rgba(255, 0, 110, 0.8))',
                'drop-shadow(0 0 20px rgba(0, 243, 255, 0.8))'
              ]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <svg width="120" height="120" viewBox="0 0 100 100">
              <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" style={{ stopColor: '#00f3ff', stopOpacity: 1 }} />
                  <stop offset="100%" style={{ stopColor: '#8338ec', stopOpacity: 1 }} />
                </linearGradient>
              </defs>
              <motion.circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="url(#grad)"
                strokeWidth="3"
                strokeDasharray="0,283"
                animate={{ strokeDasharray: ['0,283', '283,0', '0,283'] }}
                transition={{ duration: 4, repeat: Infinity }}
              />
              <text x="50" y="60" textAnchor="middle" fontFamily="Orbitron" fontSize="40" fontWeight="900" fill="url(#grad)">
                AI
              </text>
            </svg>
          </motion.div>

          <motion.h1
            className="font-['Orbitron'] text-6xl font-black mb-4 tracking-[0.3em]"
            style={{
              background: 'linear-gradient(135deg, #00f3ff, #ff006e, #8338ec)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}
            animate={{
              filter: ['drop-shadow(0 0 10px rgba(0, 243, 255, 0.5))', 'drop-shadow(0 0 25px rgba(255, 0, 110, 0.5))']
            }}
            transition={{ duration: 2, repeat: Infinity, repeatType: 'reverse' }}
          >
            AI BATTLE ARENA
          </motion.h1>

          <motion.p
            className="text-cyan-400 text-lg tracking-[0.2em] font-light"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            ADVANCED RAG SYSTEM • LLAMA 3.1 • LORA OPTIMIZED
          </motion.p>
        </motion.header>

        {/* Main Panel */}
        <motion.div
          className="relative bg-white/5 backdrop-blur-xl rounded-3xl p-10 border border-white/10 shadow-2xl overflow-hidden"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3 }}
        >
          {/* Shimmer Effect */}
          <motion.div
            className="absolute inset-0 -z-10"
            style={{
              background: 'linear-gradient(45deg, transparent, rgba(0, 243, 255, 0.03), transparent)'
            }}
            animate={{
              x: ['-200%', '200%'],
              y: ['-200%', '200%']
            }}
            transition={{ duration: 3, repeat: Infinity }}
          />

          <form onSubmit={handleSubmit}>
            {/* PDF URL Input */}
            <div className="mb-8">
              <motion.label
                className="block mb-3 font-['Orbitron'] font-bold text-sm uppercase tracking-[0.2em] text-cyan-400 pl-4"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <motion.span
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  ▶
                </motion.span>{' '}
                PDF Document URL
              </motion.label>
              <motion.input
                type="url"
                value={pdfUrl}
                onChange={(e) => setPdfUrl(e.target.value)}
                placeholder="https://example.com/document.pdf"
                required
                className="w-full px-5 py-4 bg-black/40 border-2 border-cyan-400/30 rounded-xl text-white font-['JetBrains_Mono'] focus:outline-none focus:border-cyan-400 focus:shadow-[0_0_20px_rgba(0,243,255,0.3)] transition-all"
                whileFocus={{ scale: 1.01 }}
              />
            </div>

            {/* Questions */}
            <div className="mb-8">
              <motion.label
                className="block mb-3 font-['Orbitron'] font-bold text-sm uppercase tracking-[0.2em] text-cyan-400 pl-4"
              >
                <motion.span
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  ▶
                </motion.span>{' '}
                Questions (5-15 required)
              </motion.label>

              <AnimatePresence>
                {questions.map((question, index) => (
                  <motion.div
                    key={index}
                    className="flex gap-3 mb-4"
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 30 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => updateQuestion(index, e.target.value)}
                      placeholder={`Question ${index + 1}`}
                      required
                      className="flex-1 px-5 py-4 bg-black/40 border-2 border-cyan-400/30 rounded-xl text-white font-['JetBrains_Mono'] focus:outline-none focus:border-cyan-400 focus:shadow-[0_0_20px_rgba(0,243,255,0.3)] transition-all"
                    />
                    {questions.length > 5 && (
                      <motion.button
                        type="button"
                        onClick={() => removeQuestion(index)}
                        className="px-5 py-4 bg-red-500/20 border-2 border-red-500 rounded-xl text-red-500 font-['Orbitron'] font-bold hover:bg-red-500 hover:text-white transition-all"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        ✕
                      </motion.button>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            {/* Buttons */}
            <div className="flex gap-4 flex-wrap">
              <motion.button
                type="button"
                onClick={addQuestion}
                className="px-8 py-4 bg-transparent border-2 border-cyan-400 rounded-xl text-cyan-400 font-['Orbitron'] font-bold uppercase tracking-[0.15em] relative overflow-hidden"
                whileHover={{ scale: 1.05, backgroundColor: 'rgba(0, 243, 255, 0.1)' }}
                whileTap={{ scale: 0.95 }}
              >
                + Add Question
              </motion.button>

              <motion.button
                type="submit"
                disabled={loading}
                className="px-10 py-4 rounded-xl text-white font-['Orbitron'] font-bold uppercase tracking-[0.15em] relative overflow-hidden"
                style={{
                  background: 'linear-gradient(135deg, #00f3ff, #8338ec)',
                  boxShadow: '0 10px 30px rgba(0, 243, 255, 0.3)'
                }}
                whileHover={{ y: -3, boxShadow: '0 15px 40px rgba(0, 243, 255, 0.5)' }}
                whileTap={{ y: -1 }}
              >
                <motion.div
                  className="absolute inset-0 bg-white/30 rounded-full"
                  initial={{ scale: 0, x: '-50%', y: '-50%' }}
                  whileHover={{ scale: 3 }}
                  transition={{ duration: 0.6 }}
                />
                <span className="relative z-10">🚀 Analyze Document</span>
              </motion.button>
            </div>
          </form>

          {/* Loading State */}
          <AnimatePresence>
            {loading && (
              <motion.div
                className="text-center py-16"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <motion.div
                  className="w-20 h-20 mx-auto mb-6 border-4 border-cyan-400/20 border-t-cyan-400 rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
                <motion.p
                  className="font-['Orbitron'] text-xl text-cyan-400"
                  animate={{ opacity: [1, 0.5, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  PROCESSING DOCUMENT...
                </motion.p>
                <p className="text-white/50 mt-3">
                  Extracting context • Generating embeddings • Analyzing queries
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error Message */}
          <AnimatePresence>
            {error && (
              <motion.div
                className="mt-6 p-4 bg-red-500/10 border-2 border-red-500 rounded-xl text-red-500"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: [0, -10, 10, 0] }}
                exit={{ opacity: 0 }}
                transition={{ x: { duration: 0.5 } }}
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {results && (
              <motion.div
                className="mt-12"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <h2 className="font-['Orbitron'] text-3xl text-cyan-400 mb-6">
                  📊 ANALYSIS RESULTS
                </h2>

                {results.answers.map((answer, index) => (
                  <motion.div
                    key={index}
                    className="bg-black/30 border-l-4 border-cyan-400 p-6 mb-5 rounded-xl hover:bg-black/50 hover:translate-x-1 transition-all"
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ boxShadow: '-5px 0 20px rgba(0, 243, 255, 0.3)' }}
                  >
                    <div className="font-['Orbitron'] font-bold text-pink-500 mb-3 text-lg">
                      Q{index + 1}: {results.questions[index]}
                    </div>
                    <div className="text-white/90 leading-relaxed pl-5 border-l-2 border-cyan-400/30">
                      {answer}
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Stats Bar */}
        <motion.div
          className="flex justify-around mt-10 p-6 bg-black/40 rounded-xl flex-wrap gap-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          {[
            { label: 'Model', value: 'Llama 3.1' },
            { label: 'Parameters', value: '8B' },
            { label: 'Target Accuracy', value: '95%' },
            { label: 'Response Time', value: responseTime ? `${responseTime}s` : '<8s' }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 + index * 0.1 }}
            >
              <div
                className="font-['Orbitron'] text-4xl font-black mb-2"
                style={{
                  background: 'linear-gradient(135deg, #00f3ff, #00ff9f)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text'
                }}
              >
                {stat.value}
              </div>
              <div className="text-white/50 text-sm uppercase tracking-wider">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;600&display=swap');
      `}</style>
    </div>
  );
};

export default RAGInterface;