import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  ErrorBar,
} from 'recharts'
import type { LabelPrediction, LabelSentiment } from '../types'

interface Props {
  labels: LabelPrediction[]
  radiologistLabels: Partial<Record<string, LabelSentiment>> | null
}

interface ChartDatum {
  name: string
  probability: number
  uncertainty: number
  sentiment: LabelSentiment
  radiologistLabel: LabelSentiment | null
}

const confidenceColor = (prob: number): string => {
  if (prob < 0.3) return '#2dd4bf'   // teal — low confidence / likely negative
  if (prob < 0.7) return '#fbbf24'   // amber — uncertain
  return '#f87171'                    // red — high confidence positive finding
}

const CustomTooltip: React.FC<{
  active?: boolean
  payload?: Array<{ payload: ChartDatum }>
}> = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{d.name}</p>
      <p>Probability: <strong>{(d.probability * 100).toFixed(1)}%</strong></p>
      <p>Uncertainty: <strong>±{(d.uncertainty * 100).toFixed(1)}%</strong></p>
      <p>Model: <strong>{d.sentiment}</strong></p>
      {d.radiologistLabel && (
        <p>Radiologist: <strong>{d.radiologistLabel}</strong></p>
      )}
    </div>
  )
}

const LabelConfidenceChart: React.FC<Props> = ({ labels, radiologistLabels }) => {
  const sorted = [...labels].sort((a, b) => b.probability - a.probability)

  const data: ChartDatum[] = sorted.map(l => ({
    name: l.label,
    probability: l.probability,
    uncertainty: l.uncertainty,
    sentiment: l.sentiment,
    radiologistLabel: radiologistLabels?.[l.label] ?? null,
  }))

  return (
    <div className="confidence-chart">
      <h3 className="chart-title">Label Confidence — 14 Pathologies</h3>
      <div className="chart-legend">
        <span className="legend-item low">● Low (&lt;30%)</span>
        <span className="legend-item uncertain">● Uncertain (30–70%)</span>
        <span className="legend-item high">● High (&gt;70%)</span>
      </div>
      <ResponsiveContainer width="100%" height={420}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 32, left: 160, bottom: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#334155' }}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#cbd5e1', fontSize: 12 }}
            axisLine={{ stroke: '#334155' }}
            width={155}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <ReferenceLine x={0.3} stroke="#2dd4bf" strokeDasharray="4 4" strokeOpacity={0.5} />
          <ReferenceLine x={0.7} stroke="#f87171" strokeDasharray="4 4" strokeOpacity={0.5} />
          <Bar dataKey="probability" radius={[0, 3, 3, 0]} maxBarSize={18}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={confidenceColor(entry.probability)} />
            ))}
            <ErrorBar
              dataKey="uncertainty"
              width={4}
              strokeWidth={1.5}
              stroke="rgba(255,255,255,0.5)"
              direction="x"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default LabelConfidenceChart
