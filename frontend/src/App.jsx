import { useState } from 'react';
import { Upload, Play, Package, Send } from 'lucide-react';
import './App.css';

/* Add these styles to App.css or here if using styled-components (simulated for now by modifying file content via instruction, assuming this file is App.jsx but the tool targets text content) */
/* Ideally this should go into App.css, but I'll assume standard CSS import matches */


function App() {
    const [chat, setChat] = useState([
        { role: 'assistant', text: 'Hi! Upload a menu or ask for insights 👋' }
    ]);
    const [input, setInput] = useState('');
    const [recommendations, setRecommendations] = useState([]);
    const [loadingRecs, setLoadingRecs] = useState(false);

    const fetchRecommendations = async () => {
        setLoadingRecs(true);
        try {
            const response = await fetch('http://127.0.0.1:8000/recommendations/weekly');
            const data = await response.json();
            setRecommendations(data);
        } catch (error) {
            console.error("Failed to fetch recommendations", error);
        } finally {
            setLoadingRecs(false);
        }
    };


    const sendMessage = () => {
        if (!input.trim()) return;
        setChat([...chat, { role: 'user', text: input }]);
        setInput('');
    };

    return (
        <div className="app">
            <div className="page">

                <header className="header">
                    <h1>🍽️ FlavorFlow Craft</h1>
                    <p>AI-Powered Menu Intelligence & Inventory Optimization</p>
                </header>

                <div className="dashboard-grid">

                    {/* LEFT SIDE */}
                    <div className="left-stack">

                        {/* CONTROL PANEL */}
                        <div className="card">
                            <h2>Control Panel</h2>

                            <div className="button-grid">
                                <button className="btn btn-primary">
                                    <Upload size={18} /> Initialize System
                                </button>
                                <button className="btn btn-success">
                                    <Play size={18} /> Run Analysis
                                </button>
                                <button className="btn btn-info">
                                    <Package size={18} /> Get Menu Items
                                </button>
                            </div>

                            <div className="status success">Ready to start</div>
                        </div>

                        {/* RECOMMENDATIONS PANEL */}
                        <div className="card">
                            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <h2>Weekly Recommendations</h2>
                                <button onClick={fetchRecommendations} className="btn-icon">🔄</button>
                            </div>

                            {loadingRecs ? <p>Loading...</p> : (
                                <div className="recommendations-list">
                                    {recommendations.length === 0 ? (
                                        <p style={{ color: '#888', fontStyle: 'italic' }}>No new recommendations.</p>
                                    ) : (
                                        recommendations.map((rec, i) => (
                                            <div key={i} className={`rec-item ${rec.priority.toLowerCase()}`}>
                                                <strong>{rec.type}: {rec.item}</strong>
                                                <p>{rec.message}</p>
                                                <span className="badge">{rec.priority} Priority</span>
                                            </div>
                                        ))
                                    )}
                                </div>
                            )}
                        </div>


                        {/* UPLOAD */}
                        <div className="card">
                            <h2>📁 Upload Menu</h2>
                            <label className="upload-box">
                                <input type="file" hidden />
                                <Upload size={32} />
                                <p>Drag & drop or click</p>
                                <span>CSV · Excel · PDF</span>
                            </label>
                        </div>

                    </div>

                    {/* RIGHT SIDE — CHATBOT */}
                    <div className="card chatbot">
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
                                placeholder="Ask about pricing, profitability, optimization…"
                            />
                            <button onClick={sendMessage}>
                                <Send size={18} />
                            </button>
                        </div>
                    </div>

                </div>

            </div>
        </div>
    );
}

export default App;
