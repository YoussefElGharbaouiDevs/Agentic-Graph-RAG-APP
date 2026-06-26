import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Layers, Database, Search, Cpu, BarChart } from 'lucide-react';

const COLORS = ['#22d3ee', '#60a5fa', '#10b981', '#a78bfa', '#f59e0b', '#38bdf8', '#f87171'];

interface ChunkingAnalysis {
  best_method: string;
  total_chunks: number;
  strategies: Record<string, { count: number; avg_len: number; sample: string }>;
}

interface RetrievalDoc {
  doc_id: string;
  excerpt: string;
  score: number;
}

interface VectorialData {
  chunk_info?: {
    method: string;
    total_chunks: number;
    avg_chunk_size: number;
    sample_chunks: string[];
    hierarchy: { doc_id: string; chunks: { id: string; text: string; sub_chunks: string[] }[] }[];
  };
  embedding_info?: {
    total_vectors: number;
    dimensions: number;
    pca_points: { x: number; y: number; chunk: string; preview: string; is_nearest?: boolean }[];
  };
  retrieval_method?: string;
  relevant_docs?: RetrievalDoc[];
  summary?: string;
}

export default function VectorialTab({ sharedQuery, setSharedQuery }: { sharedQuery: string, setSharedQuery: (q: string) => void }) {
  const [activeSection, setActiveSection] = useState<'chunking' | 'embeddings' | 'retrieval'>('chunking');
  const [chunkingData, setChunkingData] = useState<ChunkingAnalysis | null>(null);
  const [vectorialData, setVectorialData] = useState<VectorialData | null>(null);
  
  const [searchMethod, setSearchMethod] = useState('auto');
  const [loading, setLoading] = useState(false);
  const [showNearest, setShowNearest] = useState(false);

  useEffect(() => {
    fetchChunkingData();
    fetchVectorialData();
  }, []);

  const fetchChunkingData = async () => {
    try {
      const res = await axios.get('http://localhost:8000/chunking');
      setChunkingData(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchVectorialData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`http://localhost:8000/vectorial`, {
        params: { query: sharedQuery, method: searchMethod }
      });
      setVectorialData(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    fetchVectorialData();
  };

  return (
    <div className="flex-col" style={{ flex: 1, minHeight: 0 }}>

      {/* ── Page Header ── */}
      <div className="page-header">
        <div className="page-header-left">
          <div className="page-icon">
            <Database size={18} color="#22d3ee" />
          </div>
          <div>
            <div className="page-category">Semantic Retrieval</div>
            <h1 className="page-title">Vectorial RAG Engine</h1>
            <p className="page-subtitle">Advanced document chunking, semantic embeddings, and high-dimensional retrieval.</p>
          </div>
        </div>

        {/* Section switcher */}
        <div className="section-tabs">
          <button
            className={`section-tab${activeSection === 'chunking' ? ' active' : ''}`}
            onClick={() => setActiveSection('chunking')}
          >
            <Layers size={13} /> Chunking
          </button>
          <button
            className={`section-tab${activeSection === 'embeddings' ? ' active' : ''}`}
            onClick={() => setActiveSection('embeddings')}
          >
            <Database size={13} /> FAISS Embeddings
          </button>
          <button
            className={`section-tab${activeSection === 'retrieval' ? ' active' : ''}`}
            onClick={() => setActiveSection('retrieval')}
          >
            <Search size={13} /> Retrieval
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div style={{ padding: '1.25rem', flex: 1, overflow: 'auto' }}>
      {activeSection === 'chunking' && chunkingData && (
        <div className="animate-fade-in grid-2">
          <div className="flex-col gap-4">
            <div className="glass-card">
              <div className="flex-row items-center justify-between" style={{ marginBottom: '1.5rem' }}>
                <h3 className="flex-row items-center gap-2"><Cpu size={20} className="gradient-text"/> Chunking Strategies</h3>
                <span className="badge badge-green">Auto-selected: {chunkingData.best_method}</span>
              </div>
              
              <div className="flex-col gap-2">
                {Object.entries(chunkingData.strategies).map(([method, stats], i) => (
                  <div key={method} className="glass-panel" style={{ padding: '1rem', borderLeft: method === chunkingData.best_method ? '4px solid var(--primary-color)' : '' }}>
                    <div className="flex-row justify-between items-center" style={{ marginBottom: '0.5rem' }}>
                      <strong>{method.replace('_', ' ').toUpperCase()}</strong>
                      <span className="badge badge-blue">{stats.count} chunks</span>
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Avg Length: {stats.avg_len} chars
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="glass-card flex-col">
            <h3 className="flex-row items-center gap-2" style={{ marginBottom: '1.5rem' }}>
              <Layers size={20} className="gradient-text"/> Hierarchy Preview
            </h3>
            <div style={{ overflowY: 'auto', maxHeight: '600px', paddingRight: '0.5rem' }}>
              {vectorialData?.chunk_info?.hierarchy?.map((doc, docIdx) => (
                <div key={docIdx} className="glass-panel flex-col gap-2" style={{ padding: '1rem', marginBottom: '1rem' }}>
                  <div style={{ fontSize: '1rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>{doc.doc_id}</div>
                  {doc.chunks.map((chunk, chunkIdx) => (
                    <div key={chunkIdx} style={{ marginLeft: '1rem', paddingLeft: '1rem', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
                      <div style={{ fontSize: '0.9rem', color: 'var(--success-color)', marginBottom: '0.2rem' }}>{chunk.id}</div>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>{chunk.text}</p>
                      <div style={{ marginLeft: '1rem', paddingLeft: '1rem', borderLeft: '2px dotted rgba(255,255,255,0.1)' }}>
                        {chunk.sub_chunks.map((sub, subIdx) => (
                          <div key={subIdx} style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)', marginBottom: '0.2rem' }}>
                            ↳ Sub-chunk: {sub}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
              {!vectorialData?.chunk_info?.hierarchy && vectorialData?.chunk_info?.sample_chunks.map((sample, i) => (
                <div key={i} className="glass-panel" style={{ padding: '1rem', marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--primary-color)', marginBottom: '0.5rem' }}>Chunk {i + 1}</div>
                  <p style={{ fontSize: '0.9rem', lineHeight: 1.5 }}>{sample}...</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeSection === 'embeddings' && vectorialData?.embedding_info && (
        <div className="animate-fade-in glass-card flex-col" style={{ height: '600px' }}>
          <div className="flex-row justify-between items-center" style={{ marginBottom: '1.5rem' }}>
            <h3 className="flex-row items-center gap-2">
              <BarChart size={20} className="gradient-text"/> PCA 2D Projection
            </h3>
            <div className="flex-row gap-4 items-center">
              <label className="flex-row items-center gap-2" style={{ cursor: 'pointer', fontSize: '0.9rem' }}>
                <input type="checkbox" checked={showNearest} onChange={e => setShowNearest(e.target.checked)} />
                Highlight Nearest to Query
              </label>
              <span className="badge badge-purple">Dim: {vectorialData.embedding_info.dimensions}</span>
              <span className="badge badge-blue">Vectors: {vectorialData.embedding_info.total_vectors}</span>
            </div>
          </div>
          
          <div style={{ flex: 1, minHeight: 0, background: 'var(--bg-elevated)', borderRadius: 'var(--radius-lg)', padding: '1rem', border: '1px solid var(--border-subtle)' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="var(--border-ghost)" vertical={false} />
                <XAxis type="number" dataKey="x" name="PCA 1" stroke="var(--border-base)" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                <YAxis type="number" dataKey="y" name="PCA 2" stroke="var(--border-base)" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-raised)', borderColor: 'var(--border-base)', borderRadius: '6px', fontSize: '12px', color: 'var(--text-primary)', fontFamily: 'JetBrains Mono, monospace' }} 
                  cursor={{ fill: 'rgba(34,211,238,0.04)' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="glass-card" style={{ maxWidth: '300px', padding: '1rem' }}>
                          <p style={{ fontSize: '0.85rem' }}>{data.preview}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Chunks" data={vectorialData.embedding_info.pca_points} fill="#8884d8">
                  {vectorialData.embedding_info.pca_points.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={showNearest && entry.is_nearest ? 'var(--warning-color)' : (!showNearest ? COLORS[index % COLORS.length] : 'rgba(255,255,255,0.1)')} 
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeSection === 'retrieval' && (
        <div className="animate-fade-in grid-2">
          <div className="flex-col gap-4">
            <div className="glass-card">
              <h3 style={{ marginBottom: '1rem' }}>Test Retrieval</h3>
              <form onSubmit={handleSearch} className="flex-col gap-4">
                <textarea 
                  className="input-field" 
                  value={sharedQuery}
                  onChange={e => setSharedQuery(e.target.value)}
                  placeholder="Enter a test query..."
                />
                <div className="flex-row gap-4 items-center">
                  <select 
                    className="input-field" 
                    value={searchMethod} 
                    onChange={e => setSearchMethod(e.target.value)}
                    style={{ flex: 1, padding: '0.75rem' }}
                  >
                    <option value="auto">Auto (Meilleure méthode)</option>
                    <option value="semantic">Semantic (Vectoriel pur)</option>
                    <option value="bm25">BM25 (Mots-clés exacts)</option>
                    <option value="hybrid">Hybride (Semantic + BM25)</option>
                    <option value="mmr">MMR (Diversité maximale)</option>
                  </select>
                  <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? 'Searching...' : 'Search'}
                  </button>
                </div>
              </form>
            </div>

            {vectorialData?.summary && (
              <div className="glass-card" style={{ border: '1px solid var(--accent-color)' }}>
                <h3 className="gradient-text" style={{ marginBottom: '1rem' }}>Generated Summary</h3>
                <p style={{ lineHeight: 1.6 }}>{vectorialData.summary}</p>
              </div>
            )}
          </div>

          <div className="glass-card flex-col gap-4">
            <div className="flex-row justify-between items-center">
              <h3>Top Retrieved Documents</h3>
              {vectorialData?.retrieval_method && (
                <span className="badge badge-blue">Method: {vectorialData.retrieval_method.replace('_', ' ')}</span>
              )}
            </div>
            
            <div className="flex-col gap-4" style={{ overflowY: 'auto', maxHeight: '500px', paddingRight: '0.5rem' }}>
              {vectorialData?.relevant_docs?.map((doc, i) => (
                <div key={i} className="glass-panel" style={{ padding: '1.25rem' }}>
                  <div className="flex-row justify-between items-center" style={{ marginBottom: '0.75rem' }}>
                    <strong style={{ color: 'var(--primary-color)' }}>{doc.doc_id}</strong>
                    <span className="badge badge-green">Score: {doc.score.toFixed(3)}</span>
                  </div>
                  <p style={{ fontSize: '0.9rem', lineHeight: 1.5, color: 'var(--text-secondary)' }}>
                    {doc.excerpt}...
                  </p>
                </div>
              ))}
              {!vectorialData?.relevant_docs?.length && !loading && (
                <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '2rem' }}>
                  No documents found. Try another query.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
