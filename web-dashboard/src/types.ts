// CheXpert 14-label classification types

export type CheXpertLabel =
  | 'Atelectasis'
  | 'Cardiomegaly'
  | 'Consolidation'
  | 'Edema'
  | 'Enlarged Cardiomediastinum'
  | 'Fracture'
  | 'Lung Lesion'
  | 'Lung Opacity'
  | 'No Finding'
  | 'Pleural Effusion'
  | 'Pleural Other'
  | 'Pneumonia'
  | 'Pneumothorax'
  | 'Support Devices'

export type LabelSentiment = 'positive' | 'uncertain' | 'negative' | 'unmentioned'

export interface LabelPrediction {
  label: CheXpertLabel
  probability: number
  uncertainty: number        // ± half-width of 95% credible interval
  sentiment: LabelSentiment
}

export interface PredictionResult {
  study_id: string
  timestamp: string           // ISO-8601
  model_version: string
  labels: LabelPrediction[]
  gradcam_url: string | null  // URL to GradCAM heatmap PNG
  inference_time_ms: number
  device: 'cpu' | 'cuda'
}

export interface PatientStudy {
  patient_id: string
  study_id: string
  accession_number: string
  study_date: string          // YYYY-MM-DD
  view_position: 'PA' | 'AP' | 'LATERAL'
  image_url: string
  prediction: PredictionResult | null
  radiologist_labels: Partial<Record<CheXpertLabel, LabelSentiment>> | null
}

export interface ModelMetrics {
  model_version: string
  training_date: string
  auc_per_label: Record<CheXpertLabel, number>
  mean_auc: number
  calibration_ece: number     // Expected Calibration Error
  dataset: 'chexpert-valid' | 'chexpert-test' | 'external'
  n_samples: number
}

export interface UploadState {
  status: 'idle' | 'uploading' | 'processing' | 'done' | 'error'
  progress: number            // 0-100
  error: string | null
  result: PredictionResult | null
}

export interface DashboardView {
  current: 'upload' | 'results' | 'history'
}

export const CHEXPERT_LABELS: CheXpertLabel[] = [
  'Atelectasis',
  'Cardiomegaly',
  'Consolidation',
  'Edema',
  'Enlarged Cardiomediastinum',
  'Fracture',
  'Lung Lesion',
  'Lung Opacity',
  'No Finding',
  'Pleural Effusion',
  'Pleural Other',
  'Pneumonia',
  'Pneumothorax',
  'Support Devices',
]
