import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import ForceGraph2D from 'react-force-graph-2d';
import { Network, Users, Activity, Route } from 'lucide-react';

const COLORS = ['#22d3ee', '#60a5fa', '#10b981', '#a78bfa', '#f59e0b', '#38bdf8', '#f87171', '#14b8a6', '#fb923c'];

interface GraphData {
  graph_aura: {
    nodes: { id: string; label: string; group: number }[];
    edges: { source: string; target: string; relation: string }[];
  };
  modularity: number;
  num_clusters: number;
  communities: { id: number; members: string[]; size: number }[];
  centrality_top: { node: string; score: number }[];
  semantic_paths: string[];
  density: number;
  fusion_metrics: string;
  contextual_recommendations: string[];
}

export default function GraphTab() {
  const [activeSection, setActiveSection] = useState<'aura' | 'louvain' | 'advanced'>('aura');
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [relFilter, setRelFilter] = useState('ALL');
  const fgRef = useRef<any>(null);

  const uniqueRels = data ? Array.from(new Set(data.graph_aura.edges.map(e => e.relation))) : [];
  const filteredGraphData = data ? {
    nodes: data.graph_aura.nodes,
    links: relFilter === 'ALL' ? data.graph_aura.edges : data.graph_aura.edges.filter(e => e.relation === relFilter)
  } : { nodes: [], links: [] };

  useEffect(() => {
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/graph', { params: { query: 'overview' } });
      setData(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleNodeClick = useCallback((node: any) => {
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 1000);
      fgRef.current.zoom(8, 2000);
    }
  }, [fgRef]);

  return (
    <div className="flex-col" style={{ flex: 1, minHeight: 0 }}>

      {/* ── Page Header ── */}
      <div className="page-header">
        <div className="page-header-left">
          <div className="page-icon">
            <Network size={18} color="#22d3ee" />
          </div>
          <div>
            <div className="page-category">Knowledge Graph</div>
            <h1 className="page-title">Graph Intelligence</h1>
            <p className="page-subtitle">Neo4j Aura visualization, Louvain communities, and advanced metrics.</p>
          </div>
        </div>

        {/* Section switcher */}
        <div className="section-tabs">
          <button
            className={`section-tab${activeSection === 'aura' ? ' active' : ''}`}
            onClick={() => setActiveSection('aura')}
          >
            <Network size={13} /> Graphe Aura
          </button>
          <button
            className={`section-tab${activeSection === 'louvain' ? ' active' : ''}`}
            onClick={() => setActiveSection('louvain')}
          >
            <Users size={13} /> Communautés
          </button>
          <button
            className={`section-tab${activeSection === 'advanced' ? ' active' : ''}`}
            onClick={() => setActiveSection('advanced')}
          >
            <Activity size={13} /> Analyse Avancée
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div style={{ padding: '1.25rem', flex: 1, overflow: 'auto' }}>
      {loading && !data && (
        <div className="flex-row justify-center items-center" style={{ height: '400px', color: 'var(--primary-color)' }}>
          <div className="animate-pulse">Building Knowledge Graph...</div>
        </div>
      )}

      {activeSection === 'aura' && data?.graph_aura && (
        <div className="animate-fade-in glass-card flex-col" style={{ height: '600px', padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '1rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between' }}>
            <h3 className="flex-row items-center gap-2">
              <Network size={20} className="gradient-text"/> Neo4j Interactive Network
            </h3>
            <span className="badge badge-purple">{data.graph_aura.nodes.length} Nodes, {data.graph_aura.edges.length} Edges</span>
          </div>
          <div style={{ flex: 1, position: 'relative' }}>
            <div style={{ position: 'absolute', top: '10px', right: '10px', zIndex: 10, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ background: 'rgba(0,0,0,0.6)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Community Legend</h4>
                <div className="flex-col gap-2">
                  {data.communities.map(c => (
                    <div key={c.id} className="flex-row items-center gap-2">
                      <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: COLORS[c.id % COLORS.length] }}></div>
                      <span style={{ fontSize: '0.85rem' }}>Cluster {c.id}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.6)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Relation Filter</h4>
                <select className="input-field" value={relFilter} onChange={e => setRelFilter(e.target.value)} style={{ padding: '0.5rem', fontSize: '0.85rem' }}>
                  <option value="ALL">All Relations</option>
                  {uniqueRels.map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
                <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Showing {filteredGraphData.links.length} edges
                </div>
              </div>
            </div>
            <ForceGraph2D
              ref={fgRef}
              graphData={filteredGraphData}
              nodeLabel="id"
              nodeColor={(node: any) => COLORS[node.group % COLORS.length]}
              nodeRelSize={6}
              linkColor={() => '#162030'}
              linkDirectionalParticles={2}
              linkDirectionalParticleSpeed={0.004}
              onNodeClick={handleNodeClick}
              backgroundColor="#07090f"
              width={1000}
              height={500}
            />
          </div>
        </div>
      )}

      {activeSection === 'louvain' && data && (
        <div className="animate-fade-in grid-2">
          <div className="glass-card flex-col gap-4">
            <h3 className="flex-row items-center gap-2"><Users size={20} className="gradient-text"/> Louvain Detection</h3>
            
            <div className="flex-row justify-between glass-panel" style={{ padding: '1.5rem' }}>
              <div className="flex-col items-center">
                <span style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--success-color)' }}>{data.modularity.toFixed(3)}</span>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Modularity Score</span>
              </div>
              <div className="flex-col items-center">
                <span style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--accent-color)' }}>{data.num_clusters}</span>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Total Clusters</span>
              </div>
            </div>
          </div>
          
          <div className="glass-card" style={{ overflowY: 'auto', maxHeight: '500px' }}>
            <h3 style={{ marginBottom: '1.5rem' }}>Detected Communities</h3>
            <div className="flex-col gap-4">
              {data.communities.map((comm) => (
                <div key={comm.id} className="glass-panel" style={{ borderLeft: `4px solid ${COLORS[comm.id % COLORS.length]}` }}>
                  <div className="flex-row justify-between items-center" style={{ padding: '1rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <strong>Cluster {comm.id}</strong>
                    <span className="badge" style={{ backgroundColor: COLORS[comm.id % COLORS.length], color: 'white' }}>{comm.size} Members</span>
                  </div>
                  <div style={{ padding: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {comm.members.map(member => (
                      <span key={member} style={{ background: 'rgba(255,255,255,0.05)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                        {member}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeSection === 'advanced' && data && (
        <div className="animate-fade-in grid-2">
          <div className="flex-col gap-4">
            <div className="glass-card flex-row justify-between items-center" style={{ borderLeft: '4px solid var(--warning-color)' }}>
              <div className="flex-col">
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Graph Density</span>
                <strong style={{ fontSize: '1.5rem', color: 'var(--warning-color)' }}>{data.density?.toFixed(4) || '0.0000'}</strong>
              </div>
              <Activity size={32} color="var(--warning-color)" opacity={0.5} />
            </div>

            <div className="glass-card">
              <h3 className="flex-row items-center gap-2" style={{ marginBottom: '1.5rem' }}>
                <Activity size={20} className="gradient-text"/> Degree Centrality
              </h3>
              <div className="flex-col gap-3">
              {data.centrality_top.map((c, i) => (
                <div key={c.node} className="glass-panel flex-row justify-between items-center" style={{ padding: '1rem' }}>
                  <div className="flex-row items-center gap-3">
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 'bold' }}>#{i+1}</span>
                    <strong style={{ color: 'var(--primary-color)' }}>{c.node}</strong>
                  </div>
                  <span className="badge badge-blue">{c.score.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
          </div>

          <div className="glass-card flex-col gap-4">
            <h3 className="flex-row items-center gap-2" style={{ marginBottom: '1.5rem' }}>
              <Route size={20} className="gradient-text"/> Semantic Paths
            </h3>
            <div className="flex-col gap-3">
              {data.semantic_paths.map((path, i) => (
                <div key={i} className="glass-panel" style={{ padding: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {path.split(' → ').map((node, j, arr) => (
                      <React.Fragment key={j}>
                        <span style={{ background: 'rgba(99,102,241,0.1)', padding: '0.2rem 0.6rem', borderRadius: '4px', color: '#818cf8', fontSize: '0.9rem' }}>
                          {node}
                        </span>
                        {j < arr.length - 1 && <span style={{ color: 'var(--text-secondary)' }}>→</span>}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              ))}
              {!data.semantic_paths.length && (
                <div style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '2rem' }}>No semantic paths found in the sample.</div>
              )}
            </div>
          </div>
          
          <div className="glass-card flex-col gap-4" style={{ gridColumn: '1 / -1', border: '1px solid var(--accent-color)' }}>
            <h3 className="flex-row items-center gap-2"><Network size={20} className="gradient-text"/> Graph + Vectorial Fusion</h3>
            <div style={{ padding: '1rem', background: 'rgba(99,102,241,0.05)', borderRadius: '8px' }}>
              <p style={{ lineHeight: 1.6, fontSize: '1rem' }}>{data.fusion_metrics}</p>
            </div>
            
            <h3 className="flex-row items-center gap-2" style={{ marginTop: '1rem' }}><Activity size={20} className="gradient-text"/> Contextual Recommendations</h3>
            <ul style={{ paddingLeft: '1.5rem', lineHeight: 1.6, fontSize: '0.95rem', color: 'var(--text-secondary)' }}>
              {data.contextual_recommendations?.map((rec, i) => (
                <li key={i} style={{ marginBottom: '0.5rem' }}>{rec}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
