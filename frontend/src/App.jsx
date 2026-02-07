import { Upload, Play, Package } from 'lucide-react';
import './App.css';

function App() {
    return (
        <div className="app">
            <div className="page">

                <header className="header">
                    <h1>🍽️ FlavorFlow Craft</h1>
                    <p>AI-Powered Menu Intelligence & Inventory Optimization</p>
                </header>

                <div className="dashboard-grid">
                    <div className="left-stack">
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
                    </div>

                    <div className="card">
                        <h2>🤖 AI Assistant</h2>
                    </div>
                </div>

            </div>
        </div>
    );
}

export default App;
