import React, { useState } from 'react';
import axios from 'axios';
import { BrainCircuit, Award, GitMerge, FileCheck, RefreshCw, Terminal, Cpu } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface AgenticData {
  query: string;
  state_features: Record<string, number>;
  chosen_action: string;
  confidence: number;
  reward: number;
  q_table_snapshot: { state: string; action_graph: number; action_vectorial: number }[];
  decision_path: string[];
  final_answer: string;
  provenance: string;
  cypher_queries: string[];
}

const PIPELINE_STEPS = [
  { id: 1, label: 'Analyze Query', key: 'analyze' },
  { id: 2, label: 'State → Action', key: 'state' },
  { id: 3, label: 'Route', key: 'route' },
  { id: 4, label: 'Reward', key: 'reward' },
  { id: 5, label: 'Q-Update', key: 'update' },
];

export default function AgenticTab({ sharedQuery, setSharedQuery }: { sharedQuery: string, setSharedQuery: (q: string) => void }) {
  const [data, setData] = useState<AgenticData | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchAgentData = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/agentic', { params: { query: sharedQuery } });
      setData(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>

      {/* ── Page Header with inline query input ── */}
      <div className="page-header">
        <div className="page-header-left">
          <div className="page-icon">
            <BrainCircuit size={18} color="#22d3ee" />
          </div>
          <div>
            <div className="page-category">Reinforcement Learning</div>
            <h1 className="page-title">Agentic Orchestration</h1>
            <p className="page-subtitle">Q-Learning Policy · Reward Monitor · Feedback Loop</p>
          </div>
        </div>

        {/* Inline query input */}
        <form onSubmit={e => { e.preventDefault(); fetchAgentData(); }}
          style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexShrink: 0 }}>
          <input
            type="text"
            className="input-field"
            value={sharedQuery}
            onChange={e => setSharedQuery(e.target.value)}
            placeholder="Enter query to observe agent decision..."
            style={{ width: '300px' }}
          />
          <button type="submit" className="btn btn-primary" disabled={loading} style={{ flexShrink: 0 }}>
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Running...' : 'Execute Agent'}
          </button>
        </form>
      </div>

      {/* ── Scrollable content ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

        {/* Empty / Loading state */}
        {!data && !loading && (
          <div style={{
            flex: 1,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: '1.25rem', paddingTop: '4rem',
          }}>
            <div style={{
              width: '52px', height: '52px',
              borderRadius: 'var(--radius-lg)',
              background: 'var(--cyan-dim)',
              border: '1px solid var(--cyan-border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Cpu size={24} color="#22d3ee" />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                Agent waiting for input
              </div>
              <div style={{ fontSize: '0.775rem', color: 'var(--text-secondary)' }}>
                Enter a query above to trigger Q-Learning routing
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.875rem', paddingTop: '4rem' }}>
            <div className="status-ping animate-pulse" style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--cyan)' }} />
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.7rem', color: 'var(--cyan)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
              executing policy...
            </span>
          </div>
        )}

        {data && (
          <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

            {/* ── KPI Row — big monospace numbers ── */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.875rem' }}>

              {/* Chosen Route */}
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                padding: '1.5rem',
              }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <BrainCircuit size={13} color="#4a6480" /> Chosen Route
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1.25rem', fontWeight: 700, color: '#22d3ee', letterSpacing: '-0.015em', lineHeight: 1, marginBottom: '0.625rem' }}>
                  {data.chosen_action}
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Confidence: <span style={{ color: '#22d3ee' }}>{(data.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Reward */}
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                padding: '1.5rem',
              }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <Award size={13} color="#4a6480" /> Reward Signal
                </div>
                <div style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '2.5rem',
                  fontWeight: 700,
                  letterSpacing: '-0.025em',
                  lineHeight: 1,
                  marginBottom: '0.625rem',
                  color: data.reward > 0 ? 'var(--success-color)' : 'var(--danger-color)',
                }}>
                  {data.reward > 0 ? '+' : ''}{data.reward.toFixed(3)}
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Q-Table updated ✓
                </div>
              </div>

              {/* Provenance */}
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                padding: '1.5rem',
              }}>
                <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <FileCheck size={13} color="#4a6480" /> Provenance
                </div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '1rem', fontWeight: 600, color: 'var(--warning-color)', letterSpacing: '-0.01em', lineHeight: 1.3 }}>
                  {data.provenance}
                </div>
              </div>
            </div>

            {/* ── Feedback Loop Pipeline ── */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-dim)',
              borderRadius: 'var(--radius-lg)',
              overflow: 'hidden',
            }}>
              <div style={{ padding: '0.75rem 1.125rem', borderBottom: '1px solid var(--border-ghost)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <RefreshCw size={13} color="#22d3ee" />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>Hybrid Feedback Loop</span>
              </div>
              <div style={{ padding: '1rem 1.125rem', display: 'flex', alignItems: 'center' }}>
                {PIPELINE_STEPS.map((step, i) => (
                  <React.Fragment key={step.id}>
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '0.375rem',
                      flex: 1,
                    }}>
                      <div style={{
                        width: '28px', height: '28px',
                        borderRadius: '50%',
                        background: 'var(--cyan-dim)',
                        border: '1px solid var(--cyan-border)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontFamily: 'JetBrains Mono, monospace',
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        color: '#22d3ee',
                      }}>
                        {step.id}
                      </div>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', textAlign: 'center', lineHeight: 1.2, maxWidth: '72px' }}>
                        {step.id === 3 ? `Route → ${data.chosen_action}` :
                          step.id === 4 ? `Reward ${data.reward > 0 ? '+' : ''}${data.reward.toFixed(2)}` :
                            step.label}
                      </span>
                    </div>
                    {i < PIPELINE_STEPS.length - 1 && (
                      <div style={{ height: '1px', flex: '0 0 24px', background: 'var(--border-base)' }} />
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* ── Decision Path + Q-Table ── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>

              {/* Decision timeline */}
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                overflow: 'hidden',
              }}>
                <div style={{ padding: '0.75rem 1.125rem', borderBottom: '1px solid var(--border-ghost)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <GitMerge size={13} color="#22d3ee" />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>Decision Path</span>
                </div>
                <div style={{ padding: '0.875rem 1.125rem' }}>
                  <div style={{ position: 'relative', paddingLeft: '1.25rem', borderLeft: '1px solid var(--border-base)' }}>
                    {data.decision_path?.map((step, i) => (
                      <div key={i} style={{ position: 'relative', paddingBottom: i < data.decision_path.length - 1 ? '0.875rem' : 0 }}>
                        {/* Dot */}
                        <div style={{
                          position: 'absolute',
                          left: '-1.375rem',
                          top: '4px',
                          width: '7px', height: '7px',
                          borderRadius: '50%',
                          background: i === data.decision_path.length - 1 ? '#22d3ee' : 'var(--border-strong)',
                          border: i === data.decision_path.length - 1 ? '1px solid rgba(34,211,238,0.4)' : 'none',
                          boxShadow: i === data.decision_path.length - 1 ? '0 0 6px rgba(34,211,238,0.4)' : 'none',
                        }} />
                        <div style={{
                          fontFamily: i === data.decision_path.length - 1 ? 'JetBrains Mono, monospace' : 'inherit',
                          fontSize: '0.8rem',
                          color: i === data.decision_path.length - 1 ? '#22d3ee' : 'var(--text-secondary)',
                          fontWeight: i === data.decision_path.length - 1 ? 600 : 400,
                          lineHeight: 1.4,
                        }}>
                          {step}
                        </div>
                        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.58rem', color: 'var(--text-muted)', marginTop: '2px', letterSpacing: '0.06em' }}>
                          step_{String(i + 1).padStart(2, '0')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Q-Table chart */}
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                overflow: 'hidden',
              }}>
                <div style={{ padding: '0.75rem 1.125rem', borderBottom: '1px solid var(--border-ghost)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <BrainCircuit size={13} color="#22d3ee" />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>Policy Q-Table</span>
                </div>
                <div style={{ padding: '0.875rem', height: '220px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.q_table_snapshot || []} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="var(--border-ghost)" vertical={false} />
                      <XAxis
                        dataKey="state"
                        stroke="var(--border-base)"
                        tick={{ fill: 'var(--text-secondary)', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                      />
                      <YAxis
                        stroke="var(--border-base)"
                        tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
                      />
                      <Tooltip
                        cursor={{ fill: 'rgba(34,211,238,0.04)' }}
                        contentStyle={{
                          backgroundColor: 'var(--bg-raised)',
                          borderColor: 'var(--border-base)',
                          borderRadius: '6px',
                          fontSize: '12px',
                          color: 'var(--text-primary)',
                          fontFamily: 'JetBrains Mono, monospace',
                        }}
                      />
                      <Bar dataKey="action_vectorial" name="Vectorial RAG" fill="#22d3ee" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="action_graph" name="Graph RAG" fill="#60a5fa" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* ── Final Answer ── */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-base)',
              borderRadius: 'var(--radius-lg)',
              overflow: 'hidden',
            }}>
              <div style={{ padding: '0.75rem 1.125rem', borderBottom: '1px solid var(--border-ghost)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--success-color)' }} className="status-ping" />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>Final Answer</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', color: 'var(--text-muted)' }}>
                  via {data.chosen_action}
                </span>
              </div>
              <div style={{ padding: '1.125rem' }}>
                <p style={{ lineHeight: 1.75, fontSize: '0.9rem', color: 'var(--text-primary)', whiteSpace: 'pre-line', margin: 0 }}>
                  {data.final_answer}
                </p>
              </div>
            </div>

            {/* ── Cypher Queries ── */}
            {data.chosen_action === 'Graph RAG' && data.cypher_queries?.length > 0 && (
              <div className="animate-fade-in" style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-lg)',
                overflow: 'hidden',
              }}>
                <div style={{ padding: '0.75rem 1.125rem', borderBottom: '1px solid var(--border-ghost)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Terminal size={13} color="#22d3ee" />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>Cypher Execution Log</span>
                  <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.6rem', color: 'var(--text-muted)' }}>
                    {data.cypher_queries.length} queries
                  </span>
                </div>
                <div style={{ padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.625rem' }}>
                  {data.cypher_queries.map((query, i) => (
                    <div key={i} style={{
                      background: 'var(--bg-void)',
                      border: '1px solid var(--border-ghost)',
                      borderRadius: 'var(--radius-md)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        padding: '0.3rem 0.75rem',
                        borderBottom: '1px solid var(--border-ghost)',
                        fontFamily: 'JetBrains Mono, monospace',
                        fontSize: '0.58rem',
                        color: 'var(--text-muted)',
                        letterSpacing: '0.08em',
                      }}>
                        QUERY_{String(i + 1).padStart(2, '0')}.cypher
                      </div>
                      <pre style={{
                        margin: 0,
                        padding: '0.75rem',
                        fontSize: '0.775rem',
                        color: 'var(--text-code)',
                        whiteSpace: 'pre-wrap',
                        fontFamily: 'JetBrains Mono, monospace',
                        lineHeight: 1.6,
                      }}>
                        {query}
                      </pre>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
