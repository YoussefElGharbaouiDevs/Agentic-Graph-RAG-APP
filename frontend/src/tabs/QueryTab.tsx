import React, { useState } from 'react';
import axios from 'axios';
import { MessageSquareText, Send, RotateCcw, Activity, Search, ShieldCheck, Clock, Zap, ChevronRight, ThumbsUp, ThumbsDown } from 'lucide-react';

interface QueryResponse {
  query: string;
  query_type: string;
  decision: string;
  confidence: number;
  routed_to: string;
  answer: string;
  sources: string[];
  policy_path: string[];
  state?: string;
}

interface HistoryItem extends QueryResponse {
  timestamp: Date;
}

const STAT_STYLE: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border-dim)',
  borderRadius: 'var(--radius-lg)',
  padding: '1.125rem 1.375rem',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.625rem',
};

export default function QueryTab({ sharedQuery, setSharedQuery }: { sharedQuery: string, setSharedQuery: (q: string) => void }) {
  const [loading, setLoading] = useState(false);
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleFeedback = async (reward: number) => {
    if (!currentResponse || !currentResponse.state) return;
    try {
      await axios.post('http://localhost:8000/feedback', {
        state: currentResponse.state,
        action: currentResponse.routed_to,
        reward_adjustment: reward
      });
      setFeedbackStatus('success');
      setTimeout(() => setFeedbackStatus('idle'), 3000);
    } catch (e) {
      console.error(e);
      setFeedbackStatus('error');
      setTimeout(() => setFeedbackStatus('idle'), 3000);
    }
  };

  const handleExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sharedQuery.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/query', { query: sharedQuery });
      setCurrentResponse(res.data);
      setHistory(prev => [{ ...res.data, timestamp: new Date() }, ...prev]);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSharedQuery('');
    setCurrentResponse(null);
    setFeedbackStatus('idle');
  };

  const getTypeColor = (type: string) => {
    if (type === 'semantic') return 'var(--primary-color)';
    if (type === 'systematic') return 'var(--success-color)';
    return 'var(--warning-color)';
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>

      {/* ── Page Header ── */}
      <div className="page-header">
        <div className="page-header-left">
          <div className="page-icon">
            <MessageSquareText size={20} color="#22d3ee" />
          </div>
          <div>
            <div className="page-category">Routing Engine</div>
            <h1 className="page-title">Intelligent Query Agent</h1>
            <p className="page-subtitle">Q-Learning optimized routing for Graph or Vectorial RAG.</p>
          </div>
        </div>
      </div>

      {/* ── Content Area (Single native scrollbar on the right) ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

        {/* ── Top Input Bar ── */}
        <div style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-base)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.25rem',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <form onSubmit={handleExecute} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
            <textarea
              className="input-field"
              value={sharedQuery}
              onChange={e => setSharedQuery(e.target.value)}
              placeholder="Ask a question — e.g. What causes the polar vortex? (Shift+Enter for new line)"
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleExecute(e as any);
                }
              }}
              style={{ flex: 1, minHeight: '60px', maxHeight: '180px', resize: 'vertical' }}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flexShrink: 0 }}>
              <button type="submit" className="btn btn-primary" disabled={loading || !sharedQuery.trim()}>
                {loading ? <Activity size={16} className="animate-spin" /> : <Send size={16} />}
                {loading ? 'Routing...' : 'Execute'}
              </button>
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                <RotateCcw size={14} /> Reset
              </button>
            </div>
          </form>
        </div>

        {/* ── Empty & Loading States ── */}
        {!currentResponse && !loading && (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            padding: '4rem 0', gap: '1.25rem', opacity: 0.8
          }}>
            <div style={{
              width: '56px', height: '56px',
              borderRadius: 'var(--radius-lg)',
              background: 'var(--cyan-dim)',
              border: '1px solid var(--cyan-border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Zap size={24} color="#22d3ee" />
            </div>
            <div style={{ textAlign: 'center', maxWidth: '380px' }}>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                Ready to route
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                Enter a question above. The Q-Learning agent will classify it and route it to the optimal retrieval engine.
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '4rem 0', gap: '1rem' }}>
            <div className="status-ping animate-pulse" style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--cyan)' }} />
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: 'var(--cyan)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
              routing query...
            </span>
          </div>
        )}

        {/* ── Results & History Grid ── */}
        {currentResponse && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.5rem', alignItems: 'start' }}>
            
            {/* Left: Results */}
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              
              {/* KPIs */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                <div style={STAT_STYLE}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Search size={14} color={getTypeColor(currentResponse.query_type)} />
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                      Query Type
                    </span>
                  </div>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '1.25rem', fontWeight: 700,
                    color: getTypeColor(currentResponse.query_type), letterSpacing: '-0.01em', lineHeight: 1, marginTop: '0.25rem'
                  }}>
                    {currentResponse.query_type.toUpperCase()}
                  </div>
                </div>

                <div style={STAT_STYLE}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <ShieldCheck size={14} color="#22d3ee" />
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                      Confidence
                    </span>
                  </div>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '2rem', fontWeight: 700,
                    color: '#22d3ee', letterSpacing: '-0.02em', lineHeight: 1, marginTop: '0.25rem'
                  }}>
                    {(currentResponse.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={STAT_STYLE}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Activity size={14} color="var(--success-color)" />
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                      Routed To
                    </span>
                  </div>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '1rem', fontWeight: 700,
                    color: 'var(--success-color)', lineHeight: 1, marginTop: '0.35rem'
                  }}>
                    {currentResponse.routed_to}
                  </div>
                </div>
              </div>

              {/* Policy path */}
              {currentResponse.policy_path?.length > 0 && (
                <div style={{
                  background: 'var(--bg-card)', border: '1px solid var(--border-dim)',
                  borderRadius: 'var(--radius-lg)', padding: '1rem 1.25rem'
                }}>
                  <div style={{ fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '0.75rem' }}>
                    Routing Decision Path
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {currentResponse.policy_path.map((step, i) => (
                      <React.Fragment key={i}>
                        <span style={{
                          fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem',
                          color: i === currentResponse.policy_path.length - 1 ? '#22d3ee' : 'var(--text-secondary)',
                          fontWeight: i === currentResponse.policy_path.length - 1 ? 600 : 400,
                        }}>
                          {step}
                        </span>
                        {i < currentResponse.policy_path.length - 1 && <ChevronRight size={14} color="var(--text-muted)" />}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              )}

              {/* Answer */}
              <div style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-base)',
                borderRadius: 'var(--radius-lg)', overflow: 'hidden'
              }}>
                <div style={{
                  padding: '1rem 1.25rem', borderBottom: '1px solid var(--border-ghost)',
                  display: 'flex', alignItems: 'center', gap: '0.625rem'
                }}>
                  <Activity size={16} color="#22d3ee" />
                  <span style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-primary)' }}>Final Answer</span>
                </div>
                <div style={{ padding: '1.25rem' }}>
                  <p style={{ lineHeight: 1.75, fontSize: '0.9375rem', color: 'var(--text-primary)', whiteSpace: 'pre-line', margin: 0 }}>
                    {currentResponse.answer}
                  </p>

                  {/* RLHF Feedback UI */}
                  {currentResponse.state && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginTop: '1.25rem', padding: '0.75rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-ghost)' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Is this routing optimal?</span>
                      <button onClick={() => handleFeedback(0.5)} style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', color: 'var(--success-color)' }} title="Reward Q-Table">
                        <ThumbsUp size={16} />
                      </button>
                      <button onClick={() => handleFeedback(-0.5)} style={{ background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', color: 'var(--warning-color)' }} title="Penalize Q-Table">
                        <ThumbsDown size={16} />
                      </button>
                      {feedbackStatus === 'success' && <span style={{ fontSize: '0.75rem', color: 'var(--success-color)', marginLeft: 'auto' }}>Q-Table Updated!</span>}
                      {feedbackStatus === 'error' && <span style={{ fontSize: '0.75rem', color: 'var(--warning-color)', marginLeft: 'auto' }}>Error updating</span>}
                    </div>
                  )}

                  {currentResponse.sources && currentResponse.sources.length > 0 && (
                    <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-ghost)' }}>
                      <div style={{ fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '0.75rem' }}>
                        Sources & Evidence
                      </div>
                      {currentResponse.sources.map((src, i) => (
                        <details key={i} style={{
                          marginBottom: '0.5rem',
                          background: 'var(--bg-surface)',
                          border: '1px solid var(--border-ghost)',
                          borderRadius: 'var(--radius-md)'
                        }}>
                          <summary style={{
                            padding: '0.75rem 1rem', cursor: 'pointer', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--text-secondary)',
                            display: 'flex', alignItems: 'center', gap: '0.75rem', outline: 'none'
                          }}>
                            <span style={{ color: 'var(--text-muted)' }}>{i + 1}.</span>
                            <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'inline-block', maxWidth: '80%' }}>
                              {src.split('\n')[0].substring(0, 80)}...
                            </span>
                          </summary>
                          <div style={{ padding: '0 1rem 1rem 1rem', fontSize: '0.85rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
                            <div style={{ background: 'rgba(34,211,238,0.05)', padding: '0.75rem', borderRadius: '4px', borderLeft: '2px solid var(--cyan)', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                              {src}
                            </div>
                          </div>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right: History */}
            <div style={{
              background: 'var(--bg-surface)', border: '1px solid var(--border-ghost)',
              borderRadius: 'var(--radius-lg)', overflow: 'hidden',
              position: 'sticky', top: 0
            }}>
              <div style={{
                padding: '1rem 1.25rem', borderBottom: '1px solid var(--border-ghost)',
                display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-card)'
              }}>
                <Clock size={14} color="#4a6480" />
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>History</span>
                {history.length > 0 && (
                  <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {history.length}
                  </span>
                )}
              </div>
              <div style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {history.map((item, i) => (
                  <button
                    key={i}
                    onClick={() => setSharedQuery(item.query)}
                    style={{
                      width: '100%', display: 'flex', flexDirection: 'column', gap: '0.5rem',
                      padding: '0.875rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-ghost)',
                      background: 'transparent', cursor: 'pointer', textAlign: 'left',
                      transition: 'border-color 0.1s, background 0.1s',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-dim)';
                      (e.currentTarget as HTMLElement).style.background = 'var(--bg-raised)';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-ghost)';
                      (e.currentTarget as HTMLElement).style.background = 'transparent';
                    }}
                  >
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', fontWeight: 500, lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {item.query}
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', padding: '0.15rem 0.4rem', background: 'var(--cyan-dim)', color: 'var(--cyan)', borderRadius: '3px', border: '1px solid var(--cyan-border)' }}>
                        {item.query_type}
                      </span>
                      <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                        {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
            
          </div>
        )}
      </div>
    </div>
  );
}
