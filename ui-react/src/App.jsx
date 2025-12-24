import React, { useState, useRef, useEffect } from 'react';

const RAGChatApp = () => {
  const [sessionId, setSessionId] = useState('session-' + Date.now().toString(36));
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeFile, setActiveFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [streamEnabled, setStreamEnabled] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [stats, setStats] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const eventSourceRef = useRef(null);

  const API_BASE = ''; // Uses Vite proxy in dev, same origin in prod

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus({ type: 'loading', message: 'Uploading...' });

    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setActiveFile(data.active_file);
        setUploadStatus({
          type: 'success',
          message: `${data.file} (${data.chunks_added} chunks)`,
        });
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            content: `Uploaded "${data.file}" — ${data.chunks_added} chunks indexed`,
          },
        ]);
        fetchStats();
      } else {
        setUploadStatus({ type: 'error', message: data.detail || 'Upload failed' });
      }
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message });
    }

    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const query = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: query }]);

    if (streamEnabled) {
      streamResponse(query);
    } else {
      await syncResponse(query);
    }
  };

  const syncResponse = async (query) => {
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: 'assistant', content: '', loading: true }]);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, query }),
      });

      const data = await res.json();

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
        };
        return updated;
      });
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: `Error: ${err.message}`,
          error: true,
        };
        return updated;
      });
    }

    setIsStreaming(false);
  };

  const streamResponse = (query) => {
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);

    const url = `${API_BASE}/chat/stream?session_id=${encodeURIComponent(sessionId)}&query=${encodeURIComponent(query)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data = event.data;

      if (data === '[START]') return;

      if (data === '[END]') {
        es.close();
        eventSourceRef.current = null;
        setIsStreaming(false);
        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1].streaming = false;
          }
          return updated;
        });
        return;
      }

      if (data.startsWith('[ERROR]')) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: data,
            error: true,
          };
          return updated;
        });
        return;
      }

      const text = data.replace(/\\n/g, '\n');
      setMessages((prev) => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].content += text;
        }
        return updated;
      });
    };

    es.onerror = () => {
      es.close();
      eventSourceRef.current = null;
      setIsStreaming(false);
    };
  };

  const clearChat = () => {
    setMessages([]);
  };

  const clearIndex = async () => {
    try {
      await fetch(`${API_BASE}/index/clear?session_id=${sessionId}`, { method: 'POST' });
      setActiveFile(null);
      setStats(null);
      setUploadStatus(null);
      setMessages((prev) => [...prev, { role: 'system', content: 'Index cleared' }]);
    } catch (err) {
      console.error('Failed to clear index:', err);
    }
  };

  const newSession = () => {
    setSessionId('session-' + Date.now().toString(36));
    setMessages([]);
    setActiveFile(null);
    setStats(null);
    setUploadStatus(null);
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <aside style={{ ...styles.sidebar, ...(sidebarOpen ? {} : styles.sidebarClosed) }}>
        <div style={styles.sidebarHeader}>
          <h1 style={styles.logo}>RAG</h1>
          <button onClick={() => setSidebarOpen(false)} style={styles.closeBtn}>
            ×
          </button>
        </div>

        <div style={styles.section}>
          <label style={styles.label}>Session</label>
          <div style={styles.sessionRow}>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              style={styles.sessionInput}
            />
            <button onClick={newSession} style={styles.iconBtn} title="New Session">
              +
            </button>
          </div>
        </div>

        <div style={styles.section}>
          <label style={styles.label}>Document</label>
          <input
            ref={fileInputRef}
            type="file"
            accept=".docx,.ipynb,.txt,.md"
            onChange={handleUpload}
            style={styles.fileInput}
            id="file-upload"
          />
          <label htmlFor="file-upload" style={styles.uploadBtn}>
            Choose File
          </label>

          {uploadStatus && (
            <div
              style={{
                ...styles.status,
                ...(uploadStatus.type === 'success'
                  ? styles.statusSuccess
                  : uploadStatus.type === 'error'
                  ? styles.statusError
                  : styles.statusLoading),
              }}
            >
              {uploadStatus.message}
            </div>
          )}

          {activeFile && (
            <div style={styles.activeFile}>
              <span style={styles.activeFileIcon}>📄</span>
              {activeFile}
            </div>
          )}
        </div>

        {stats && stats.total_chunks > 0 && (
          <div style={styles.section}>
            <label style={styles.label}>Index Stats</label>
            <div style={styles.statsBox}>
              <div style={styles.statRow}>
                <span>Chunks</span>
                <span style={styles.statValue}>{stats.total_chunks}</span>
              </div>
              {stats.sources?.map((s, i) => (
                <div key={i} style={styles.statRow}>
                  <span style={styles.statSource}>{s.source}</span>
                  <span style={styles.statValue}>{s.chunks}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={styles.section}>
          <label style={styles.label}>Options</label>
          <label style={styles.checkbox}>
            <input
              type="checkbox"
              checked={streamEnabled}
              onChange={(e) => setStreamEnabled(e.target.checked)}
            />
            <span>Stream responses</span>
          </label>
        </div>

        <div style={styles.sidebarFooter}>
          <button onClick={clearChat} style={styles.footerBtn}>
            Clear Chat
          </button>
          <button onClick={clearIndex} style={{ ...styles.footerBtn, ...styles.dangerBtn }}>
            Clear Index
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main style={styles.main}>
        {!sidebarOpen && (
          <button onClick={() => setSidebarOpen(true)} style={styles.menuBtn}>
            ☰
          </button>
        )}

        <div style={styles.messagesContainer}>
          {messages.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>💬</div>
              <h2 style={styles.emptyTitle}>Local RAG Chat</h2>
              <p style={styles.emptyText}>
                Upload a document and start asking questions.
                <br />
                Powered by Mistral-7B + FAISS
              </p>
            </div>
          ) : (
            <div style={styles.messages}>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    ...styles.message,
                    ...(msg.role === 'user'
                      ? styles.userMessage
                      : msg.role === 'system'
                      ? styles.systemMessage
                      : styles.assistantMessage),
                    ...(msg.error ? styles.errorMessage : {}),
                  }}
                >
                  {msg.role === 'assistant' && msg.loading ? (
                    <div style={styles.loading}>
                      <span style={styles.dot}>●</span>
                      <span style={{ ...styles.dot, animationDelay: '0.2s' }}>●</span>
                      <span style={{ ...styles.dot, animationDelay: '0.4s' }}>●</span>
                    </div>
                  ) : (
                    <>
                      <div style={styles.messageContent}>
                        {msg.content}
                        {msg.streaming && <span style={styles.cursor}>▋</span>}
                      </div>
                      {msg.sources && msg.sources.length > 0 && (
                        <div style={styles.sources}>
                          Sources: {msg.sources.map((s) => s.source).join(', ')}
                        </div>
                      )}
                    </>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div style={styles.inputContainer}>
          <div style={styles.inputWrapper}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask a question..."
              style={styles.input}
              disabled={isStreaming}
            />
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              style={{
                ...styles.sendBtn,
                ...(isStreaming || !input.trim() ? styles.sendBtnDisabled : {}),
              }}
            >
              {isStreaming ? '...' : '→'}
            </button>
          </div>
          <p style={styles.hint}>
            Press Enter to send • Supports .docx, .ipynb, .txt, .md
          </p>
        </div>
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        * {
          box-sizing: border-box;
        }
        body {
          margin: 0;
          font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
          background: #0a0a0a;
        }
        input:focus, button:focus {
          outline: none;
        }
        ::-webkit-scrollbar {
          width: 6px;
        }
        ::-webkit-scrollbar-track {
          background: transparent;
        }
        ::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    background: '#0a0a0a',
    color: '#e5e5e5',
  },

  // Sidebar
  sidebar: {
    width: '280px',
    background: '#111',
    borderRight: '1px solid #222',
    display: 'flex',
    flexDirection: 'column',
    transition: 'transform 0.2s ease',
  },
  sidebarClosed: {
    transform: 'translateX(-100%)',
    position: 'absolute',
    zIndex: -1,
  },
  sidebarHeader: {
    padding: '20px',
    borderBottom: '1px solid #222',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    fontSize: '20px',
    fontWeight: '700',
    letterSpacing: '2px',
    color: '#fff',
    margin: 0,
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: '#666',
    fontSize: '24px',
    cursor: 'pointer',
    padding: '0',
    lineHeight: '1',
  },

  section: {
    padding: '16px 20px',
    borderBottom: '1px solid #1a1a1a',
  },
  label: {
    display: 'block',
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: '#666',
    marginBottom: '10px',
  },
  sessionRow: {
    display: 'flex',
    gap: '8px',
  },
  sessionInput: {
    flex: 1,
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: '6px',
    padding: '8px 12px',
    color: '#e5e5e5',
    fontSize: '13px',
  },
  iconBtn: {
    width: '36px',
    height: '36px',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: '6px',
    color: '#e5e5e5',
    fontSize: '18px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },

  fileInput: {
    display: 'none',
  },
  uploadBtn: {
    display: 'block',
    width: '100%',
    padding: '10px',
    background: '#1a1a1a',
    border: '1px dashed #333',
    borderRadius: '6px',
    color: '#888',
    fontSize: '13px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
  status: {
    marginTop: '10px',
    padding: '8px 10px',
    borderRadius: '4px',
    fontSize: '12px',
  },
  statusSuccess: {
    background: 'rgba(34, 197, 94, 0.1)',
    color: '#22c55e',
  },
  statusError: {
    background: 'rgba(239, 68, 68, 0.1)',
    color: '#ef4444',
  },
  statusLoading: {
    background: 'rgba(59, 130, 246, 0.1)',
    color: '#3b82f6',
  },
  activeFile: {
    marginTop: '10px',
    padding: '8px 10px',
    background: '#1a1a1a',
    borderRadius: '4px',
    fontSize: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  activeFileIcon: {
    fontSize: '14px',
  },

  statsBox: {
    background: '#1a1a1a',
    borderRadius: '6px',
    padding: '10px',
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '12px',
    padding: '4px 0',
    color: '#888',
  },
  statSource: {
    maxWidth: '140px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  statValue: {
    color: '#e5e5e5',
    fontWeight: '500',
  },

  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#888',
    cursor: 'pointer',
  },

  sidebarFooter: {
    marginTop: 'auto',
    padding: '16px 20px',
    borderTop: '1px solid #222',
    display: 'flex',
    gap: '8px',
  },
  footerBtn: {
    flex: 1,
    padding: '8px',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: '6px',
    color: '#888',
    fontSize: '12px',
    cursor: 'pointer',
  },
  dangerBtn: {
    borderColor: 'rgba(239, 68, 68, 0.3)',
    color: '#ef4444',
  },

  // Main
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
  },
  menuBtn: {
    position: 'absolute',
    top: '16px',
    left: '16px',
    width: '40px',
    height: '40px',
    background: '#111',
    border: '1px solid #333',
    borderRadius: '8px',
    color: '#e5e5e5',
    fontSize: '18px',
    cursor: 'pointer',
    zIndex: 10,
  },

  messagesContainer: {
    flex: 1,
    overflow: 'auto',
    padding: '20px',
  },
  emptyState: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#666',
  },
  emptyIcon: {
    fontSize: '48px',
    marginBottom: '16px',
    opacity: 0.5,
  },
  emptyTitle: {
    fontSize: '24px',
    fontWeight: '600',
    color: '#e5e5e5',
    margin: '0 0 8px 0',
  },
  emptyText: {
    fontSize: '14px',
    textAlign: 'center',
    lineHeight: '1.6',
    margin: 0,
  },

  messages: {
    maxWidth: '800px',
    margin: '0 auto',
  },
  message: {
    marginBottom: '16px',
    padding: '14px 18px',
    borderRadius: '12px',
    fontSize: '14px',
    lineHeight: '1.6',
  },
  userMessage: {
    background: '#1e3a5f',
    marginLeft: '40px',
    borderBottomRightRadius: '4px',
  },
  assistantMessage: {
    background: '#1a1a1a',
    marginRight: '40px',
    borderBottomLeftRadius: '4px',
  },
  systemMessage: {
    background: 'transparent',
    color: '#666',
    fontSize: '12px',
    textAlign: 'center',
    padding: '8px',
  },
  errorMessage: {
    background: 'rgba(239, 68, 68, 0.1)',
    color: '#ef4444',
  },
  messageContent: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  sources: {
    marginTop: '10px',
    paddingTop: '10px',
    borderTop: '1px solid #333',
    fontSize: '11px',
    color: '#666',
  },
  cursor: {
    animation: 'blink 1s infinite',
    color: '#3b82f6',
  },
  loading: {
    display: 'flex',
    gap: '4px',
  },
  dot: {
    animation: 'pulse 1s infinite',
    color: '#666',
  },

  inputContainer: {
    padding: '16px 20px 24px',
    borderTop: '1px solid #1a1a1a',
  },
  inputWrapper: {
    maxWidth: '800px',
    margin: '0 auto',
    display: 'flex',
    gap: '8px',
  },
  input: {
    flex: 1,
    background: '#111',
    border: '1px solid #333',
    borderRadius: '12px',
    padding: '14px 18px',
    color: '#e5e5e5',
    fontSize: '14px',
  },
  sendBtn: {
    width: '48px',
    height: '48px',
    background: '#3b82f6',
    border: 'none',
    borderRadius: '12px',
    color: '#fff',
    fontSize: '20px',
    cursor: 'pointer',
    transition: 'background 0.15s ease',
  },
  sendBtnDisabled: {
    background: '#1a1a1a',
    color: '#666',
    cursor: 'not-allowed',
  },
  hint: {
    maxWidth: '800px',
    margin: '8px auto 0',
    fontSize: '11px',
    color: '#444',
    textAlign: 'center',
  },
};

export default RAGChatApp;
