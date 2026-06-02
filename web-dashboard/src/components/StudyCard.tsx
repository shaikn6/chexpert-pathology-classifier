import React from 'react'
import type { PatientStudy, LabelPrediction } from '../types'

interface Props {
  study: PatientStudy
  isSelected: boolean
  onSelect: (study: PatientStudy) => void
}

const TOP_LABELS_COUNT = 4

const sentimentBadge = (prob: number): string => {
  if (prob >= 0.7) return 'badge-positive'
  if (prob >= 0.3) return 'badge-uncertain'
  return 'badge-negative'
}

const topLabels = (labels: LabelPrediction[]): LabelPrediction[] =>
  [...labels]
    .filter(l => l.probability >= 0.3)
    .sort((a, b) => b.probability - a.probability)
    .slice(0, TOP_LABELS_COUNT)

const StudyCard: React.FC<Props> = ({ study, isSelected, onSelect }) => {
  const highlights = study.prediction ? topLabels(study.prediction.labels) : []

  return (
    <article
      className={`study-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(study)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onSelect(study)}
      aria-pressed={isSelected}
      aria-label={`Study ${study.study_id}, ${study.study_date}`}
    >
      <header className="card-header">
        <div className="card-id">
          <span className="study-id">{study.study_id}</span>
          <span className="patient-id">Patient {study.patient_id}</span>
        </div>
        <div className="card-meta">
          <span className="view-badge">{study.view_position}</span>
          <time dateTime={study.study_date} className="study-date">
            {new Date(study.study_date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
            })}
          </time>
        </div>
      </header>

      {highlights.length > 0 && (
        <ul className="label-chips" aria-label="Top predicted findings">
          {highlights.map(label => (
            <li key={label.label} className={`label-chip ${sentimentBadge(label.probability)}`}>
              <span className="chip-name">{label.label}</span>
              <span className="chip-prob">{(label.probability * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      )}

      {study.prediction && (
        <footer className="card-footer">
          <span>Model: {study.prediction.model_version}</span>
          <span>{study.prediction.inference_time_ms} ms</span>
          {study.radiologist_labels && (
            <span className="validated-badge">Radiologist-validated</span>
          )}
        </footer>
      )}

      {!study.prediction && (
        <div className="card-unanalysed">Pending analysis</div>
      )}
    </article>
  )
}

export default StudyCard
