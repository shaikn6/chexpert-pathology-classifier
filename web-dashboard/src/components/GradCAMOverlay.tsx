import React, { useRef, useEffect, useState } from 'react'

interface Props {
  imageUrl: string
  gradcamUrl: string | null
  studyId: string
}

type OverlayMode = 'original' | 'heatmap' | 'overlay'

const GradCAMOverlay: React.FC<Props> = ({ imageUrl, gradcamUrl, studyId }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [mode, setMode] = useState<OverlayMode>('overlay')
  const [opacity, setOpacity] = useState(0.5)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const baseImage = new Image()
    baseImage.crossOrigin = 'anonymous'

    const drawScene = (base: HTMLImageElement, heat: HTMLImageElement | null) => {
      canvas.width = base.naturalWidth || 512
      canvas.height = base.naturalHeight || 512

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (mode === 'original' || !heat) {
        ctx.drawImage(base, 0, 0, canvas.width, canvas.height)
        return
      }

      if (mode === 'heatmap' && heat) {
        ctx.drawImage(heat, 0, 0, canvas.width, canvas.height)
        return
      }

      // Overlay mode: base + heatmap blended
      ctx.drawImage(base, 0, 0, canvas.width, canvas.height)
      ctx.globalAlpha = opacity
      ctx.drawImage(heat, 0, 0, canvas.width, canvas.height)
      ctx.globalAlpha = 1.0
    }

    // Use a placeholder gradient as fallback when real images aren't available (dev mode)
    const useFallback = () => {
      canvas.width = 512
      canvas.height = 512
      const grad = ctx.createRadialGradient(256, 256, 30, 256, 256, 256)
      grad.addColorStop(0, 'rgba(248, 113, 113, 0.9)')
      grad.addColorStop(0.5, 'rgba(251, 191, 36, 0.5)')
      grad.addColorStop(1, 'rgba(15, 23, 42, 0)')
      ctx.fillStyle = '#0f172a'
      ctx.fillRect(0, 0, 512, 512)
      if (mode !== 'original') {
        ctx.fillStyle = grad
        ctx.fillRect(0, 0, 512, 512)
      }
      // Grid lines to simulate X-ray texture
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.06)'
      ctx.lineWidth = 1
      for (let x = 0; x < 512; x += 32) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 512); ctx.stroke()
      }
      for (let y = 0; y < 512; y += 32) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(512, y); ctx.stroke()
      }
      ctx.fillStyle = 'rgba(148,163,184,0.3)'
      ctx.font = '14px monospace'
      ctx.fillText(`Study: ${studyId}`, 12, 24)
      ctx.fillText('GradCAM Overlay (demo)', 12, 44)
    }

    baseImage.onload = () => {
      if (!gradcamUrl || mode === 'original') {
        drawScene(baseImage, null)
        return
      }
      const heatImage = new Image()
      heatImage.crossOrigin = 'anonymous'
      heatImage.onload = () => drawScene(baseImage, heatImage)
      heatImage.onerror = () => drawScene(baseImage, null)
      heatImage.src = gradcamUrl
    }
    baseImage.onerror = () => {
      setLoadError(null) // suppress error, use fallback
      useFallback()
    }
    baseImage.src = imageUrl
  }, [imageUrl, gradcamUrl, mode, opacity, studyId])

  return (
    <div className="gradcam-container">
      <div className="gradcam-controls" role="group" aria-label="Image overlay controls">
        {(['original', 'heatmap', 'overlay'] as const).map(m => (
          <button
            key={m}
            className={`overlay-btn ${mode === m ? 'active' : ''}`}
            onClick={() => setMode(m)}
          >
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
        {mode === 'overlay' && (
          <label className="opacity-control">
            <span>Opacity</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={e => setOpacity(parseFloat(e.target.value))}
              aria-label="GradCAM overlay opacity"
            />
            <span>{Math.round(opacity * 100)}%</span>
          </label>
        )}
      </div>
      {loadError && <p className="gradcam-error">{loadError}</p>}
      <canvas
        ref={canvasRef}
        className="gradcam-canvas"
        aria-label={`Chest X-ray with GradCAM heatmap for study ${studyId}`}
      />
    </div>
  )
}

export default GradCAMOverlay
