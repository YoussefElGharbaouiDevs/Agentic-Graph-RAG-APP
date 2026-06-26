import React, { useState, useEffect } from 'react';
import { Database, Network, BrainCircuit, MessageSquareText, Sun, Moon } from 'lucide-react';
import VectorialTab from './tabs/VectorialTab';
import GraphTab from './tabs/GraphTab';
import AgenticTab from './tabs/AgenticTab';
import QueryTab from './tabs/QueryTab';
import './index.css';
type Tab = 'vectorial' | 'graph' | 'agentic' | 'query';

const NAV_ITEMS: { id: Tab; label: string; icon: React.ElementType; sub: string }[] = [
  { id: 'query',     label: 'Query Assistant', icon: MessageSquareText, sub: 'AI Routing' },
  { id: 'vectorial', label: 'Vectorial RAG',   icon: Database,          sub: 'Semantic' },
  { id: 'graph',     label: 'Graph RAG',        icon: Network,           sub: 'Knowledge' },
  { id: 'agentic',   label: 'Agentic RAG',      icon: BrainCircuit,      sub: 'RL Policy' },
];

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('query');
  const [sharedQuery, setSharedQuery] = useState('');
  const [isLightMode, setIsLightMode] = useState(false);

  // Appliquer le thème clair à la racine du document HTML
  useEffect(() => {
    if (isLightMode) {
      document.documentElement.classList.add('light-theme');
    } else {
      document.documentElement.classList.remove('light-theme');
    }
  }, [isLightMode]);

  const renderTab = () => {
    switch (activeTab) {
      case 'vectorial': return <VectorialTab sharedQuery={sharedQuery} setSharedQuery={setSharedQuery} />;
      case 'graph':     return <GraphTab />;
      case 'agentic':   return <AgenticTab sharedQuery={sharedQuery} setSharedQuery={setSharedQuery} />;
      case 'query':     return <QueryTab sharedQuery={sharedQuery} setSharedQuery={setSharedQuery} />;
      default:          return <QueryTab sharedQuery={sharedQuery} setSharedQuery={setSharedQuery} />;
    }
  };

  const activeItem = NAV_ITEMS.find(n => n.id === activeTab);

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* ══════════════════════════════
          SIDEBAR
          ══════════════════════════════ */}
      <aside style={{
        width: '256px',
        minWidth: '256px',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-void)',
        borderRight: '1px solid var(--border-ghost)',
      }}>

        {/* Brand */}
        <div style={{
          padding: '1.375rem 1.125rem',
          borderBottom: '1px solid var(--border-ghost)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.875rem',
        }}>
          {/* Logo mark */}
          <div style={{
            width: '34px', height: '34px',
            borderRadius: '8px',
            background: 'var(--cyan-dim)',
            border: '1px solid var(--cyan-border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}>
            <BrainCircuit size={20} color="#22d3ee" />
          </div>

          <div>
            <div style={{
              fontSize: '0.9375rem', fontWeight: 700,
              color: 'var(--text-head)',
              letterSpacing: '-0.02em', lineHeight: 1.2,
            }}>
              GraphRAG
            </div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '0.65rem',
              color: 'var(--text-muted)',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              marginTop: '2px',
            }}>
              v2.0 · platform
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div style={{ flex: 1, padding: '1rem 0.625rem' }}>
          <div className="section-label" style={{ paddingLeft: '0.625rem', marginBottom: '0.375rem' }}>
            Modules
          </div>

          <nav style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
            {NAV_ITEMS.map(({ id, label, icon: Icon, sub }) => {
              const active = activeTab === id;
              return (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  style={{
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.625rem 0.75rem',
                    borderRadius: 'var(--radius-md)',
                    border: 'none',
                    background: active ? 'rgba(34, 211, 238, 0.06)' : 'transparent',
                    cursor: 'pointer',
                    width: '100%',
                    textAlign: 'left',
                    transition: 'background 0.1s ease',
                  }}
                >
                  {/* Barre active cyan */}
                  {active && (
                    <span style={{
                      position: 'absolute',
                      left: 0,
                      top: '6px', bottom: '6px',
                      width: '2px',
                      background: 'var(--cyan)',
                      borderRadius: '0 2px 2px 0',
                    }} />
                  )}

                  <Icon
                    size={16}
                    color={active ? '#22d3ee' : '#4a6480'}
                    style={{ flexShrink: 0 }}
                  />

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', minWidth: 0 }}>
                    <span style={{
                      fontSize: '0.875rem',
                      fontWeight: active ? 600 : 400,
                      color: active ? 'var(--text-head)' : 'var(--text-secondary)',
                      letterSpacing: active ? '-0.01em' : '0',
                      lineHeight: 1.2,
                    }}>
                      {label}
                    </span>
                    <span style={{
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: '0.625rem',
                      color: active ? 'var(--cyan)' : 'var(--text-muted)',
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                    }}>
                      {sub}
                    </span>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* System status */}
        <div style={{
          padding: '0.875rem 1rem',
          borderTop: '1px solid var(--border-ghost)',
        }}>
          <div className="section-label" style={{ marginBottom: '0.625rem' }}>System</div>
          {[
            { label: 'Backend',     ok: true,  val: ':8000' },
            { label: 'Neo4j Aura', ok: true,  val: 'graph' },
            { label: 'FAISS',      ok: true,  val: 'index' },
          ].map(({ label, ok, val }) => (
            <div key={label} style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              marginBottom: '0.4rem',
            }}>
              <div
                className={ok ? 'status-ping' : ''}
                style={{
                  width: '6px', height: '6px', borderRadius: '50%',
                  background: ok ? 'var(--success)' : 'var(--danger)',
                  flexShrink: 0,
                }}
              />
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', flex: 1 }}>
                {label}
              </span>
              <code style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.02em',
              }}>
                {val}
              </code>
            </div>
          ))}
        </div>
      </aside>

      {/* ══════════════════════════════
          MAIN AREA
          ══════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* Topbar */}
        <header style={{
          height: '50px',
          minHeight: '50px',
          display: 'flex',
          alignItems: 'center',
          padding: '0 1.25rem',
          gap: '0.4rem',
          background: 'var(--bg-surface)',
          borderBottom: '1px solid var(--border-ghost)',
        }}>
          {/* Breadcrumb */}
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            platform
          </span>
          <span style={{ color: 'var(--border-base)', fontSize: '0.85rem', margin: '0 0.1rem' }}>/</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
            {activeItem?.id}
          </span>
          <span style={{ color: 'var(--border-base)', fontSize: '0.85rem', margin: '0 0.1rem' }}>/</span>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--cyan)' }}>
            {activeItem?.sub.toLowerCase()}
          </span>

          {/* Right */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button 
              onClick={() => setIsLightMode(!isLightMode)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '32px', height: '32px', borderRadius: '50%',
                background: 'var(--bg-raised)', border: '1px solid var(--border-base)',
                cursor: 'pointer', transition: 'all 0.1s ease', color: 'var(--text-primary)'
              }}
              title="Toggle Theme"
            >
              {isLightMode ? <Moon size={16} /> : <Sun size={16} />}
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <div
                className="status-ping"
                style={{
                  width: '5px', height: '5px', borderRadius: '50%',
                  background: 'var(--success)',
                }}
              />
              <span style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '0.6rem',
                color: 'var(--text-secondary)',
                letterSpacing: '0.05em',
              }}>
                LIVE
              </span>
            </div>
          </div>
        </header>

        {/* Page */}
        <main style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {renderTab()}
        </main>
      </div>
    </div>
  );
}

export default App;
