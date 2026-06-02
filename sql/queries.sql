-- CheXpert Pathology Classifier — Analytics Queries
-- Assumes the schema defined in schema.sql

-- ── 1. Top-5 most prevalent pathologies (positive label rate) ─────────────

SELECT
    l.label_name,
    COUNT(*) FILTER (WHERE l.sentiment = 'positive')    AS positive_count,
    COUNT(*)                                             AS total_predictions,
    ROUND(
        COUNT(*) FILTER (WHERE l.sentiment = 'positive')::NUMERIC / COUNT(*) * 100,
        2
    )                                                    AS prevalence_pct
FROM labels l
JOIN predictions p ON l.prediction_id = p.id
WHERE p.created_at >= NOW() - INTERVAL '90 days'
GROUP BY l.label_name
ORDER BY prevalence_pct DESC
LIMIT 5;

-- ── 2. AUC per label for the active model version ─────────────────────────

SELECT
    mv.version_tag,
    am.label_name,
    am.auc,
    am.n_positive_samples,
    am.n_total_samples
FROM model_auc_metrics am
JOIN model_versions mv ON am.model_version_id = mv.id
WHERE mv.is_active = TRUE
  AND am.eval_dataset = 'chexpert-valid'
ORDER BY am.auc DESC;

-- ── 3. Mean AUC trend across model versions ───────────────────────────────

SELECT
    mv.version_tag,
    mv.training_date,
    mv.mean_auc,
    mv.calibration_ece
FROM model_versions mv
ORDER BY mv.training_date;

-- ── 4. Studies with high-confidence Edema + Pleural Effusion co-occurrence ─

SELECT
    ps.id AS study_id,
    ps.study_date,
    ps.view_position,
    edema.probability   AS edema_prob,
    effusion.probability AS effusion_prob
FROM patient_studies ps
JOIN predictions pred ON pred.study_id = ps.id
JOIN labels edema    ON edema.prediction_id    = pred.id AND edema.label_name = 'Edema'
JOIN labels effusion ON effusion.prediction_id = pred.id AND effusion.label_name = 'Pleural Effusion'
WHERE edema.probability   >= 0.7
  AND effusion.probability >= 0.7
ORDER BY ps.study_date DESC
LIMIT 50;

-- ── 5. Model accuracy vs radiologist for the latest 30-day cohort ─────────

WITH radiologist_ground_truth AS (
    SELECT
        ra.study_id,
        ra.label_name,
        ra.sentiment AS gt_sentiment
    FROM radiologist_annotations ra
    WHERE ra.annotated_at >= NOW() - INTERVAL '30 days'
),
model_predictions AS (
    SELECT
        p.study_id,
        l.label_name,
        l.sentiment AS pred_sentiment
    FROM labels l
    JOIN predictions p ON l.prediction_id = p.id
    JOIN model_versions mv ON p.model_version_id = mv.id
    WHERE mv.is_active = TRUE
)
SELECT
    rgt.label_name,
    COUNT(*)                                                         AS n_compared,
    COUNT(*) FILTER (WHERE rgt.gt_sentiment = mp.pred_sentiment)     AS n_correct,
    ROUND(
        COUNT(*) FILTER (WHERE rgt.gt_sentiment = mp.pred_sentiment)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100,
        2
    )                                                                AS accuracy_pct
FROM radiologist_ground_truth rgt
JOIN model_predictions mp USING (study_id, label_name)
GROUP BY rgt.label_name
ORDER BY accuracy_pct DESC;

-- ── 6. Average inference latency by device type (last 7 days) ────────────

SELECT
    p.device,
    ROUND(AVG(p.inference_time_ms), 1) AS avg_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p.inference_time_ms), 1) AS p95_latency_ms,
    COUNT(*) AS n_inferences
FROM predictions p
WHERE p.created_at >= NOW() - INTERVAL '7 days'
GROUP BY p.device;
