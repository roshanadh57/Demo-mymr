import React, { useState, useEffect, useCallback } from 'react';
import '../styles/PatientChat.css';

function PatientChat({ patientName, onBack }) {
  // Existing states
  const [profileData, setProfileData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [profileError, setProfileError] = useState('');
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [isQueryLoading, setIsQueryLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  // Modal and document states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docError, setDocError] = useState('');

  // Fetch patient summary
  const fetchPatientSummary = useCallback(async () => {
    if (!patientName) return;
    setIsLoading(true);
    setProfileError('');
    try {
      const response = await fetch(`http://localhost:8000/summary/${encodeURIComponent(patientName)}`);
      if (!response.ok) {
        if (response.status === 404) throw new Error(`No summary found for "${patientName}".`);
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      setProfileData({
        medication_summary: data.summary?.medication_summary || 'No medication information available.',
        lifestyle_recommendations: data.summary?.lifestyle_recommendations || 'No lifestyle recommendations available.',
        condition_summary: data.summary?.condition_summary || 'No condition information available.'
      });
    } catch (error) {
      console.error('Error fetching patient summary:', error);
      setProfileError(error.message || 'Could not load patient summary.');
      setProfileData(null);
    } finally {
      setIsLoading(false);
    }
  }, [patientName]);

  useEffect(() => { fetchPatientSummary(); }, [fetchPatientSummary]);

  // Chat submit handler
  const handleQuerySubmit = async (e) => {
  e.preventDefault();
  if (!query.trim() || isQueryLoading) return;

  const userMessage = { sender: 'user', text: query, timestamp: new Date() };
  const newMessages = [...messages, userMessage];
  setMessages(newMessages);
  setIsQueryLoading(true);
  setQuery('');

  try {
    const response = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_name: patientName, query }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server responded with ${response.status}`);
    }
    const data = await response.json();
    const botText = typeof data.answer === 'string'
      ? data.answer
      : JSON.stringify(data.answer, null, 2);
    const botMessage = { sender: 'bot', text: botText, timestamp: new Date() };
    setMessages([...newMessages, botMessage]);
  } catch (error) {
    console.error('Chat error:', error);
    const errorMessage = {
      sender: 'bot',
      text: `Sorry, I encountered an error: ${error.message}. Please ensure the backend server is running on port 8000.`,
      timestamp: new Date(),
      isError: true,
    };
    setMessages([...newMessages, errorMessage]);
  } finally {
    setIsQueryLoading(false);
  }
};

  // Open documents modal and fetch document list
  const openDocumentsModal = async () => {
    setIsModalOpen(true);
    if (documents.length > 0) return; // Already have data
    setIsLoadingDocs(true);
    setDocError('');
    try {
      const response = await fetch(`http://localhost:8000/documents/${encodeURIComponent(patientName)}`);
      if (!response.ok) throw new Error(`Failed to fetch documents: ${response.status}`);
      const docsList = await response.json();
      setDocuments(docsList);
      if (docsList.length === 0) setDocError('No documents found for this patient.');
    } catch (error) {
      console.error('Error fetching documents:', error);
      setDocError(error.message || 'Could not load documents. Please check backend.');
    } finally {
      setIsLoadingDocs(false);
    }
  };

  // Load document content
  const viewDocumentContent = async (doc) => {
    setSelectedDoc({ filename: doc.filename, content: 'Loading...', classification: '' });
    setDocError('');
    try {
      const response = await fetch('http://localhost:8000/document_content', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path: doc.path }),
      });
      if (!response.ok) throw new Error(`Failed to load document: ${response.status}`);
      const data = await response.json();
      setSelectedDoc({ filename: doc.filename, content: data.content, classification: data.classification });
    } catch (error) {
      console.error('Error loading content:', error);
      setDocError(error.message || 'Could not load document content.');
      setSelectedDoc({ filename: doc.filename, content: 'Error loading content. Please try again.', classification: '' });
    }
  };

  // Close modal
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedDoc(null);
    setDocError('');
  };

  // Scroll chat messages on new message
  useEffect(() => {
    const messagesContainer = document.querySelector('.patient-chat-messages');
    if (messagesContainer) messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }, [messages]);

  // Format time
  const formatMessageTime = (timestamp) => new Date(timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

  // Group documents by category
  const groupedDocuments = documents.reduce((groups, doc) => {
    const cat = doc.category || 'Other';
    groups[cat] = groups[cat] || [];
    groups[cat].push(doc);
    return groups;
  }, {});

  return (
    <>
      <div className="patient-chat-container">
        <button onClick={onBack} className="back-button">← Back to Patients</button>
        <header className="patient-header">
          <h2>Patient Summary for: {patientName}</h2>
        </header>
        {isLoading ? (
          <p>Loading patient summary...</p>
        ) : profileError ? (
          <div>
            <p>{profileError}</p>
            <button onClick={onBack}>Back to Patients</button>
          </div>
        ) : (
          <>
            <section className="summary-section">
              <div><h3>Medication Summary</h3><p>{profileData.medication_summary}</p></div>
              <div><h3>Lifestyle Recommendations</h3><p>{profileData.lifestyle_recommendations}</p></div>
              <div><h3>Condition Summary</h3><p>{profileData.condition_summary}</p></div>
            </section>
            <section className="documents-section">
              <button className="documents-button" onClick={openDocumentsModal}>View Source Documents</button>
            </section>
            {/* Floating Chat Widget */}
            <div className={`patient-chat-widget ${isExpanded ? 'expanded' : ''}`}>
              {!isExpanded && (
                <button
                  className="chat-widget-toggle"
                  onClick={() => setIsExpanded(true)}
                  aria-label="Open chat"
                  title="Open Chat"
                >💬</button>
              )}
              {isExpanded && (
                <div className="chat-widget-content">
                  <div className="chat-widget-header">
                    <span>Ask About {patientName}</span>
                    <button
                      className="chat-widget-close"
                      onClick={() => setIsExpanded(false)}
                      aria-label="Close chat"
                      title="Close Chat"
                    >&times;</button>
                  </div>
                  <div className="patient-chat-messages">
                    {messages.length === 0 ? (
                      <p>Ask questions about {patientName}'s medical history, conditions, or treatments.</p>
                    ) : (
                      messages.map((msg, idx) => (
                        <div key={idx} style={{
                          textAlign: msg.sender === 'user' ? 'right' : 'left',
                          marginBottom: '4px',
                          color: msg.isError ? 'red' : 'black',
                        }}>
                          <strong>{msg.sender === 'user' ? 'You' : 'AI'}: </strong>{msg.text}<br />
                          <small>{formatMessageTime(msg.timestamp)}</small>
                        </div>
                      ))
                    )}
                  </div>
                  <form onSubmit={handleQuerySubmit} className="chat-form">
                    <input
                      type="text"
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      placeholder="Ask a question..."
                      disabled={isQueryLoading}
                      autoFocus
                    />
                    <button type="submit" disabled={isQueryLoading || !query.trim()}>
                      {isQueryLoading ? 'Sending...' : 'Send'}
                    </button>
                  </form>
                </div>
              )}
            </div>
            {/* Document Modal */}
            {isModalOpen && (
              <div className="modal-overlay" onClick={closeModal}>
                <div className="modal-content" onClick={e => e.stopPropagation()}>
                  <div className="modal-header">
                    <h3>Source Documents for {patientName}</h3>
                    <button className="modal-close-button" onClick={closeModal} aria-label="Close modal">&times;</button>
                  </div>
                  <hr />
                  <div className="modal-body">
                    {isLoadingDocs ? (
                      <p>Loading documents...</p>
                    ) : docError ? (
                      <p className="doc-error">{docError}</p>
                    ) : selectedDoc ? (
                      <div>
                        <button className="back-to-docs" onClick={() => setSelectedDoc(null)}>← Back to document list</button>
                        <h4>{selectedDoc.filename}</h4>
                        <p><strong>Document Type:</strong> {selectedDoc.classification}</p>
                        <pre className="document-content">{selectedDoc.content}</pre>
                      </div>
                    ) : (
                      <div>
                        {Object.entries(groupedDocuments).map(([category, docs]) => (
                          <div key={category} className="document-category-section">
                            <h4>{category}</h4>
                            <ul className="documents-list">
                              {docs.map(doc => (
                                <li
                                  key={doc.path}
                                  className={`doc-item ${doc.category.toLowerCase().replace(/\s+/g, '')}`}
                                  onClick={() => viewDocumentContent(doc)}
                                  title={`View content of ${doc.filename}`}
                                >
                                  {doc.filename}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

export default PatientChat;
