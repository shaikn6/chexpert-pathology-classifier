import React, { useState, useCallback } from 'react'
import type { PatientStudy, PredictionResult, DashboardView } from './types'
import LabelConfidenceChart from './components/LabelConfidenceChart'
import GradCAMOverlay from './components/GradCAMOverlay'
import StudyCard from './components/StudyCard'

// Mock data for demonstration — in production these come from the FastAPI backend
const MOCK_STUDIES: PatientStudy[] = [
  {
    patient_id: 'P-001',
    study_id: 'S-001',
    accession_number: 'ACC-20240101-001',
    study_date: '2024-01-01',
    view_position: 'PA',
    image_url: '/api/studies/S-001/image',
    prediction: {
      study_id: 'S-001',
      timestamp: '2024-01-01T10:30:00Z',
      model_version: 'chexpert-v2.1',
      labels: [
        { label: 'Atelectasis', probability: 0.72, uncertainty: 0.08, sentiment: 'positive' },
        { label: 'Cardiomegaly', probability: 0.15, uncertainty: 0.04, sentiment: 'negative' },
        { label: 'Consolidation', probability: 0.08, uncertainty: 0.03, sentiment: 'negative' },
        { label: 'Edema', probability: 0.61, uncertainty: 0.12, sentiment: 'uncertain' },
        { label: 'Enlarged Cardiomediastinum', probability: 0.22, uncertainty: 0.06, sentiment: 'uncertain' },
        { label: 'Fracture', probability: 0.04, uncertainty: 0.02, sentiment: 'negative' },
        { label: 'Lung Lesion', probability: 0.11, uncertainty: 0.05, sentiment: 'negative' },
        { label: 'Lung Opacity', probability: 0.68, uncertainty: 0.09, sentiment: 'positive' },
        { label: 'No Finding', probability: 0.03, uncertainty: 0.01, sentiment: 'negative' },
        { label: 'Pleural Effusion', probability: 0.55, uncertainty: 0.11, sentiment: 'uncertain' },
        { label: 'Pleural Other', probability: 0.07, uncertainty: 0.03, sentiment: 'negative' },
        { label: 'Pneumonia', probability: 0.18, uncertainty: 0.07, sentiment: 'negative' },
        { label: 'Pneumothorax', probability: 0.05, uncertainty: 0.02, sentiment: 'negative' },
        { label: 'Support Devices', probability: 0.88, uncertainty: 0.04, sentiment: 'positive' },
      ],
      gradcam_url: '/api/studies/S-001/gradcam',
      inference_time_ms: 142,
      device: 'cuda',
    },
    radiologist_labels: {
      Atelectasis: 'positive',
      Edema: 'uncertain',
      'Pleural Effusion': 'positive',
      'Support Devices': 'positive',
    },
  },
  {
    patient_id: 'P-002',
    study_id: 'S-002',
    accession_number: 'ACC-20240102-001',
    study_date: '2024-01-02',
    view_position: 'AP',
    image_url: '/api/studies/S-002/image',
    prediction: {
      study_id: 'S-002',
      timestamp: '2024-01-02T09:15:00Z',
      model_version: 'chexpert-v2.1',
      labels: [
        { label: 'Atelectasis', probability: 0.12, uncertainty: 0.03, sentiment: 'negative' },
        { label: 'Cardiomegaly', probability: 0.78, uncertainty: 0.06, sentiment: 'positive' },
        { label: 'Consolidation', probability: 0.05, uncertainty: 0.02, sentiment: 'negative' },
        { label: 'Edema', probability: 0.42, uncertainty: 0.09, sentiment: 'uncertain' },
        { label: 'Enlarged Cardiomediastinum', probability: 0.81, uncertainty: 0.05, sentiment: 'positive' },
        { label: 'Fracture', probability: 0.02, uncertainty: 0.01, sentiment: 'negative' },
        { label: 'Lung Lesion', probability: 0.06, uncertainty: 0.03, sentiment: 'negative' },
        { label: 'Lung Opacity', probability: 0.14, uncertainty: 0.04, sentiment: 'negative' },
        { label: 'No Finding', probability: 0.01, uncertainty: 0.01, sentiment: 'negative' },
        { label: 'Pleural Effusion', probability: 0.33, uncertainty: 0.08, sentiment: 'uncertain' },
        { label: 'Pleural Other', probability: 0.04, uncertainty: 0.02, sentiment: 'negative' },
        { label: 'Pneumonia', probability: 0.07, uncertainty: 0.04, sentiment: 'negative' },
        { label: 'Pneumothorax', probability: 0.03, uncertainty: 0.01, sentiment: 'negative' },
        { label: 'Support Devices', probability: 0.19, uncertainty: 0.05, sentiment: 'negative' },
      ],
      gradcam_url: '/api/studies/S-002/gradcam',
      inference_time_ms: 138,
      device: 'cuda',
    },
    radiologist_labels: {
      Cardiomegaly: 'positive',
      'Enlarged Cardiomediastinum': 'positive',
      Edema: 'uncertain',
    },
  },
]

const App: React.FC = () => {
  const [view, setView] = useState<DashboardView['current']>('history')
  const [selectedStudy, setSelectedStudy] = useState<PatientStudy | null>(MOCK_STUDIES[0])
  const [uploadState, setUploadState] = useState<{ status: string; progress: number; error: string | null }>({
    status: 'idle',
    progress: 0,
    error: null,
  })

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    const imageFile = files.find(f => f.type.startsWith('image/') || f.name.endsWith('.dcm'))
    if (!imageFile) {
      setUploadState({ status: 'error', progress: 0, error: 'Please drop a PNG, JPEG, or DICOM file.' })
      return
    }
    setUploadState({ status: 'uploading', progress: 0, error: null })
    // Simulate upload + inference
    let progress = 0
    const interval = setInterval(() => {
      progress += 10
      setUploadState(s => ({ ...s, progress }))
      if (progress >= 100) {
        clearInterval(interval)
        setUploadState({ status: 'done', progress: 100, error: null })
        setSelectedStudy(MOCK_STUDIES[0])
        setView('results')
      }
    }, 200)
  }, [])

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => e.preventDefault()

  const handleStudySelect = (study: PatientStudy) => {
    setSelectedStudy(study)
    setView('results')
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-icon">⊕</span>
          <h1>CheXpert Pathology Classifier</h1>
        </div>
        <nav className="header-nav">
          {(['upload', 'results', 'history'] as const).map(v => (
            <button
              key={v}
              className={`nav-btn ${view === v ? 'active' : ''}`}
              onClick={() => setView(v)}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </nav>
        <div className="header-meta">
          <span className="model-badge">chexpert-v2.1</span>
        </div>
      </header>

      <main className="app-main">
        {view === 'upload' && (
          <section className="view-upload" aria-labelledby="upload-heading">
            <h2 id="upload-heading">Upload Chest X-Ray</h2>
            <div
              className={`drop-zone ${uploadState.status === 'uploading' ? 'uploading' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              role="region"
              aria-label="File upload drop zone"
            >
              {uploadState.status === 'idle' && (
                <>
                  <div className="drop-icon">⊕</div>
                  <p className="drop-primary">Drop DICOM or PNG file here</p>
                  <p className="drop-secondary">Supports PA, AP, and Lateral views · Max 50 MB</p>
                </>
              )}
              {uploadState.status === 'uploading' && (
                <div className="upload-progress">
                  <p>Uploading… {uploadState.progress}%</p>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${uploadState.progress}%` }} />
                  </div>
                </div>
              )}
              {uploadState.status === 'done' && (
                <div className="upload-success">
                  <p>Analysis complete — switching to Results</p>
                </div>
              )}
              {uploadState.status === 'error' && (
                <div className="upload-error">
                  <p>{uploadState.error}</p>
                  <button onClick={() => setUploadState({ status: 'idle', progress: 0, error: null })}>
                    Try again
                  </button>
                </div>
              )}
            </div>
          </section>
        )}

        {view === 'results' && selectedStudy?.prediction && (
          <section className="view-results" aria-labelledby="results-heading">
            <h2 id="results-heading">
              Study {selectedStudy.study_id} — {selectedStudy.study_date}
            </h2>
            <div className="results-grid">
              <div className="results-image-panel">
                <GradCAMOverlay
                  imageUrl={selectedStudy.image_url}
                  gradcamUrl={selectedStudy.prediction.gradcam_url}
                  studyId={selectedStudy.study_id}
                />
                <div className="study-meta">
                  <span>View: {selectedStudy.view_position}</span>
                  <span>Inference: {selectedStudy.prediction.inference_time_ms} ms</span>
                  <span>Device: {selectedStudy.prediction.device.toUpperCase()}</span>
                </div>
              </div>
              <div className="results-chart-panel">
                <LabelConfidenceChart
                  labels={selectedStudy.prediction.labels}
                  radiologistLabels={selectedStudy.radiologist_labels}
                />
              </div>
            </div>
          </section>
        )}

        {view === 'results' && !selectedStudy?.prediction && (
          <section className="view-results-empty">
            <p>No study selected. Upload an image or select from History.</p>
          </section>
        )}

        {view === 'history' && (
          <section className="view-history" aria-labelledby="history-heading">
            <h2 id="history-heading">Study History</h2>
            <div className="history-stats">
              <div className="stat-chip">
                <span className="stat-value">{MOCK_STUDIES.length}</span>
                <span className="stat-label">Total Studies</span>
              </div>
              <div className="stat-chip">
                <span className="stat-value">
                  {MOCK_STUDIES.filter(s => s.prediction !== null).length}
                </span>
                <span className="stat-label">Analysed</span>
              </div>
            </div>
            <div className="study-list">
              {MOCK_STUDIES.map(study => (
                <StudyCard
                  key={study.study_id}
                  study={study}
                  isSelected={selectedStudy?.study_id === study.study_id}
                  onSelect={handleStudySelect}
                />
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
