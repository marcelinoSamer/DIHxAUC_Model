import { useState } from 'react';
import {
    Upload,
    Play,
    Package,
    TrendingUp,
    DollarSign,
    Send
} from 'lucide-react';
import './App.css';

function App() {
    const [status, setStatus] = useState('Ready to start');
    const [loading, setLoading] = useState(false);
    const [analysis, setAnalysis] = useState(null);
    const [chat, setChat] = useState([
        { role: 'assistant', text: 'Hi! Upload a menu or ask for insights 👋' }
    ]);
    const [input, setInput] = useState('');

    const handleAnalyze = () => {
        setLoading(true);
        setStatus('Running AI analysis...');
        setTimeout(() => {
            setAnalysis({
                summary: { stars: 156, plowhorses: 243, puzzles: 89 }
            });
            setStatus('✅ Analysis complete!');
            setLoading(false);
        }, 2000);
    };

    const sendMessage = () => {
        if (!input.trim()) return;
        setChat([...chat, { role: 'user', text: input }]);
        setInput('');
    };

    return (
        <div className="app">
            <div className="page">

                {/* HEADER */}
                <header className="header">
                    <h1>🍽️ FlavorFlow Craft</h1>
                    <p>AI-Powered Menu Intelligence & Inventory Optimization</p>
                </header>

                {/* MAIN DASHBOARD */}
                <main className="dashboard-grid">

                    {/* LEFT COLUMN */}
                    <section className="left-stack">

                        {/* CONTROL PANEL */}
                        <div className="card">
                            <h2>⚙️ Control Panel</h2>

                            <div className="button-grid">
                                <button className="btn btn-primary">
                                    <Upload size={18} /> Initialize System
                                </button>

                                <button
                                    className="btn btn-success"
                                    onClick={handleAnalyze}
                                    disabled={loading}
                                >
                                    <Play size={18} /> Run Analysis
                                </button>

                                <button className="btn btn-info">
                                    <Package size={18} /> Get Menu Items
                                </button>
                            </div>

                            <div className="status success">{status}</div>
                        </div>

                        {/* FILE UPLOAD */}
                        <div className="card">
                            <h2>📁 Upload Menu</h2>

                            <label className="upload-box">
                                <input type="file" hidden />
                                <Upload size={36} />
                                <p>Drop menu files here or click to upload</p>
                                <span>CSV · Excel · PDF supported</span>
                            </label>
                        </div>

                    </section>

                    {/* RIGHT COLUMN — CHATBOT */}
                    <section className="card chatbot">
                        <h2>🤖 AI Assistant</h2>

                        <div className="chat-window">
                            {chat.map((m, i) => (
                                <div key={i} className={`chat-bubble ${m.role}`}>
                                    {m.text}
                                </div>
                            ))}
                        </div>

                        <div className="chat-input">
                            <input
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                placeholder="Ask about profitability, pricing, optimization…"
                            />
                            <button onClick={sendMessage}>
                                <Send size={18} />
                            </button>
                        </div>
                    </section>

                </main>

                {/* ANALYSIS RESULTS */}
                {analysis && (
                    <section className="card results-section">
                        <h2>📊 Analysis Results</h2>

                        <div className="stats-grid">
                            <div className="stat-card stat-stars">
                                <TrendingUp className="stat-icon" />
                                <p className="stat-value">{analysis.summary.stars}</p>
                                <p>Stars</p>
                            </div>

                            <div className="stat-card stat-plowhorses">
                                <Package className="stat-icon" />
                                <p className="stat-value">{analysis.summary.plowhorses}</p>
                                <p>Plowhorses</p>
                            </div>

                            <div className="stat-card stat-puzzles">
                                <DollarSign className="stat-icon" />
                                <p className="stat-value">{analysis.summary.puzzles}</p>
                                <p>Puzzles</p>
                            </div>
                        </div>
                    </section>
                )}

                {/* FOOTER */}
                <footer className="footer">
                    Built for Deloitte x AUC Hackathon 2025 💚
                </footer>

            </div>
        </div>
    );
}

export default App;
