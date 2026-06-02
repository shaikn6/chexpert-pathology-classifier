-- CheXpert Pathology Classifier — Database Schema
-- PostgreSQL 15+

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Enum types ────────────────────────────────────────────────────────────

CREATE TYPE view_position AS ENUM ('PA', 'AP', 'LATERAL');
CREATE TYPE label_sentiment AS ENUM ('positive', 'uncertain', 'negative', 'unmentioned');
CREATE TYPE device_type AS ENUM ('cpu', 'cuda');

-- ── Model versions ────────────────────────────────────────────────────────

CREATE TABLE model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_tag     TEXT NOT NULL UNIQUE,          -- e.g. 'chexpert-v2.1'
    architecture    TEXT NOT NULL,                  -- e.g. 'DenseNet121'
    training_date   DATE NOT NULL,
    dataset         TEXT NOT NULL,                  -- e.g. 'CheXpert-v1.0-small'
    n_train_samples INTEGER NOT NULL,
    mean_auc        NUMERIC(5, 4),
    calibration_ece NUMERIC(6, 5),                  -- Expected Calibration Error
    artifact_uri    TEXT,                           -- S3/MLflow URI to model weights
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure exactly one active model at a time
CREATE UNIQUE INDEX model_versions_active_unique
    ON model_versions (is_active)
    WHERE is_active = TRUE;

-- ── Patients ──────────────────────────────────────────────────────────────

CREATE TABLE patients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id     TEXT NOT NULL UNIQUE,   -- de-identified patient identifier
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Patient studies (X-ray exams) ─────────────────────────────────────────

CREATE TABLE patient_studies (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    study_date          DATE NOT NULL,
    view_position       view_position NOT NULL,
    accession_number    TEXT UNIQUE,
    image_path          TEXT NOT NULL,              -- object-store key / path
    image_width_px      INTEGER,
    image_height_px     INTEGER,
    bits_allocated      SMALLINT,                   -- DICOM tag (0028,0100)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX patient_studies_patient_id_idx ON patient_studies (patient_id);
CREATE INDEX patient_studies_study_date_idx ON patient_studies (study_date);

-- ── Predictions ───────────────────────────────────────────────────────────

CREATE TABLE predictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id            UUID NOT NULL REFERENCES patient_studies(id) ON DELETE CASCADE,
    model_version_id    UUID NOT NULL REFERENCES model_versions(id),
    inference_time_ms   INTEGER NOT NULL,
    device              device_type NOT NULL,
    gradcam_path        TEXT,                       -- object-store key for GradCAM PNG
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX predictions_study_id_idx        ON predictions (study_id);
CREATE INDEX predictions_model_version_id_idx ON predictions (model_version_id);
CREATE INDEX predictions_created_at_idx      ON predictions (created_at DESC);

-- ── Labels (per-label probabilities) ──────────────────────────────────────

CREATE TABLE labels (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id   UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    label_name      TEXT NOT NULL,                  -- one of the 14 CheXpert labels
    probability     NUMERIC(5, 4) NOT NULL,
    uncertainty     NUMERIC(5, 4) NOT NULL,         -- 95% credible interval half-width
    sentiment       label_sentiment NOT NULL,
    UNIQUE (prediction_id, label_name)
);

CREATE INDEX labels_prediction_id_idx ON labels (prediction_id);
CREATE INDEX labels_label_name_idx    ON labels (label_name);
CREATE INDEX labels_probability_idx   ON labels (label_name, probability DESC);

-- ── Radiologist ground truth ───────────────────────────────────────────────

CREATE TABLE radiologist_annotations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id        UUID NOT NULL REFERENCES patient_studies(id) ON DELETE CASCADE,
    label_name      TEXT NOT NULL,
    sentiment       label_sentiment NOT NULL,
    annotator_id    TEXT,                           -- de-identified radiologist ID
    annotated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (study_id, label_name, annotator_id)
);

CREATE INDEX radiologist_annotations_study_id_idx ON radiologist_annotations (study_id);

-- ── AUC metrics per model + label ─────────────────────────────────────────

CREATE TABLE model_auc_metrics (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id    UUID NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    label_name          TEXT NOT NULL,
    auc                 NUMERIC(5, 4) NOT NULL,
    n_positive_samples  INTEGER NOT NULL,
    n_total_samples     INTEGER NOT NULL,
    eval_dataset        TEXT NOT NULL,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_version_id, label_name, eval_dataset)
);
