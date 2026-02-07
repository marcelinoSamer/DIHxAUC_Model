import { useState } from 'react';
import { Upload, Play, Package, Send } from 'lucide-react';
import './App.css';

function App() {
    const [chat, setChat] = useState([
        { role: 'assistant', text: 'Hi! Upload a menu or ask for insights 👋' }
    ]);
    const [input, setInput] = useState('');

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
