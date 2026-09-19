import { useEffect, useRef, useState } from 'react'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as poseDetection from '@tensorflow-models/pose-detection'
import * as tf from '@tensorflow/tfjs'
import './App.css'

type Prediction = { className: string; probability: number }
type Observation = Prediction & { time: string }

const SKELETON = [
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
]

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const classifierRef = useRef<mobilenet.MobileNet | null>(null)
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationRef = useRef<number | null>(null)
  const previousKeypointsRef = useRef<poseDetection.Keypoint[] | null>(null)
  const previousActiveRef = useRef(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [modelsReady, setModelsReady] = useState(false)
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState('')
  const [movementScore, setMovementScore] = useState(0)
  const [personDetected, setPersonDetected] = useState(false)
  const [reps, setReps] = useState(0)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [observations, setObservations] = useState<Observation[]>([])

  useEffect(() => {
    let active = true
    async function prepare() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs localhost or HTTPS. Open the Vite URL, not the HTML file directly.')
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
        if (!active) { stream.getTracks().forEach((track) => track.stop()); return }
        streamRef.current = stream
        if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
        setCameraReady(true)
        const [classifier, detector] = await Promise.all([
          mobilenet.load({ version: 2, alpha: 1.0 }),
          tf.ready().then(() => poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING })),
        ])
        if (!active) return
        classifierRef.current = classifier
        detectorRef.current = detector
        setModelsReady(true)
      } catch (cameraError) {
        const message = cameraError instanceof DOMException && cameraError.name === 'NotAllowedError'
          ? 'Camera permission was blocked. Click the camera icon in your browser address bar and choose Allow, then reload.'
          : cameraError instanceof DOMException && cameraError.name === 'NotFoundError'
            ? 'No camera was found. Connect a webcam or use a device with a camera, then reload.'
            : cameraError instanceof Error ? cameraError.message : 'Something blocked the camera setup.'
        setError(message)
      } finally { if (active) setLoading(false) }
    }
    prepare()
    return () => {
      active = false
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      detectorRef.current?.dispose()
    }
  }, [])

  useEffect(() => {
    if (!modelsReady || !cameraReady) return
    let active = true
    async function trackFrame() {
      const video = videoRef.current
      const detector = detectorRef.current
      const canvas = canvasRef.current
      if (!active || !video || !detector || !canvas || video.readyState < 2) return
      const poses = await detector.estimatePoses(video)
      const pose = poses[0]
      const visibleKeypoints = pose?.keypoints.filter((keypoint) => (keypoint.score ?? 0) > 0.35) ?? []
      const personVisible = visibleKeypoints.length >= 5
      setPersonDetected(personVisible)
      const context = canvas.getContext('2d')
      if (context) {
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        context.clearRect(0, 0, canvas.width, canvas.height)
        if (pose) {
          const points = new Map(pose.keypoints.map((keypoint) => [keypoint.name, keypoint]))
          context.strokeStyle = '#d5f36f'
          context.lineWidth = 4
          SKELETON.forEach(([startName, endName]) => {
            const start = points.get(startName)
            const end = points.get(endName)
            if (start && end && (start.score ?? 0) > 0.35 && (end.score ?? 0) > 0.35) {
              context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke()
            }
          })
          visibleKeypoints.forEach((keypoint) => { context.fillStyle = '#ff765f'; context.beginPath(); context.arc(keypoint.x, keypoint.y, 6, 0, Math.PI * 2); context.fill() })
        }
      }
      if (personVisible && previousKeypointsRef.current) {
        const previous = new Map(previousKeypointsRef.current.map((keypoint) => [keypoint.name, keypoint]))
        const totalMovement = visibleKeypoints.reduce((sum, keypoint) => {
          const old = previous.get(keypoint.name)
          return old ? sum + Math.hypot(keypoint.x - old.x, keypoint.y - old.y) : sum
        }, 0)
        const averageMovement = totalMovement / Math.max(visibleKeypoints.length, 1)
        const intensity = Math.min(100, Math.round(averageMovement * 3.2))
        setMovementScore((current) => Math.round(current * 0.75 + intensity * 0.25))
        const isActive = intensity > 22
        if (isActive && !previousActiveRef.current) setReps((current) => current + 1)
        previousActiveRef.current = isActive
      } else if (!personVisible) {
        setMovementScore((current) => Math.max(0, current - 2))
        previousActiveRef.current = false
      }
      previousKeypointsRef.current = pose?.keypoints ?? null
      animationRef.current = requestAnimationFrame(() => { void trackFrame() })
    }
    void trackFrame()
    return () => { active = false; if (animationRef.current) cancelAnimationFrame(animationRef.current) }
  }, [cameraReady, modelsReady])

  async function scanScene() {
    if (!classifierRef.current || !videoRef.current || !cameraReady) return
    setScanning(true); setError('')
    try {
      const results = await classifierRef.current.classify(videoRef.current, 3)
      setPredictions(results)
      const top = results[0]
      if (top) setObservations((current) => [{ ...top, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }, ...current].slice(0, 5))
    } catch { setError('The model could not read this frame. Try again with better light.') }
    finally { setScanning(false) }
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code === 'Space' && cameraReady && !scanning) { event.preventDefault(); void scanScene() }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cameraReady, scanning])

  const topPrediction = predictions[0]
  const statusLabel = loading ? 'Opening camera' : !cameraReady ? 'Camera unavailable' : !modelsReady ? 'Loading movement model' : personDetected ? 'Person detected' : 'Show yourself to the camera'

  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand"><span className="brand-mark">◉</span><span>Lens Lab</span></div><div className="model-status"><span className={`status-dot ${modelsReady ? 'ready' : ''}`}></span>{statusLabel}<span className="divider">/</span><span className="model-name">MoveNet + MobileNet</span></div></header>
      <section className="intro"><div><p className="eyebrow">CAMERA + MOVEMENT LEARNING</p><h1>Move your body.<br /><em>See it think.</em></h1></div><p className="intro-copy">Lens Lab watches body landmarks in real time, turns motion into a signal, and teaches you what the model can and cannot know.</p></section>
      <section className="workspace">
        <div className="camera-panel"><div className="camera-frame"><video ref={videoRef} muted playsInline aria-label="Live camera view" /><canvas ref={canvasRef} className="pose-overlay" aria-hidden="true" />{!cameraReady && <div className="camera-placeholder"><span className="loader">◌</span><p>{loading ? 'Opening your camera...' : 'Camera unavailable'}</p><small>{error || 'Allow camera access to begin.'}</small></div>}<div className="frame-corner corner-tl"></div><div className="frame-corner corner-tr"></div><div className="frame-corner corner-bl"></div><div className="frame-corner corner-br"></div><span className="live-label"><i></i> {cameraReady ? 'LIVE TRACKING' : 'WAITING'}</span></div><div className="camera-actions"><div><span className="hint-key">SPACE</span><span className="action-note">Classify the current frame</span></div><button className="scan-button" type="button" onClick={scanScene} disabled={!cameraReady || scanning}>{scanning ? 'Reading...' : 'Scan object ↗'}</button></div>{error && <p className="error-message">{error}</p>}</div>
        <aside className="results-panel"><div className="movement-card"><div className="panel-heading"><div><p className="eyebrow">MOVEMENT SIGNAL</p><h2>{personDetected ? 'You are in frame' : 'Waiting for a person'}</h2></div><span className="result-index">{String(reps).padStart(2, '0')}</span></div><div className="confidence-row"><span>Intensity</span><strong>{movementScore}%</strong></div><div className="confidence-track"><span style={{ width: `${movementScore}%` }}></span></div><div className="movement-meta"><span><b>{reps}</b> motion bursts</span><span className={personDetected ? 'signal-on' : ''}>● {personDetected ? 'Tracking' : 'No pose'}</span></div></div>{topPrediction ? <div className="object-read"><p className="eyebrow">OBJECT READ</p><h2>{topPrediction.className}</h2><div className="confidence-row"><span>Confidence</span><strong>{Math.round(topPrediction.probability * 100)}%</strong></div><div className="confidence-track"><span style={{ width: `${topPrediction.probability * 100}%` }}></span></div><p className="result-explanation">This is a visual guess, not a fact. The model compares pixels with categories from its training data.</p>{predictions.slice(1).map((item) => <div className="possibility" key={item.className}><span>{item.className}</span><span>{Math.round(item.probability * 100)}%</span></div>)}</div> : <div className="empty-result"><span>◎</span><p>Object scans appear here.</p><small>Movement tracking is already live above.</small></div>}<div className="privacy-note"><span>⌁</span><p><strong>Private by design</strong><br />Video is processed in this tab and never uploaded.</p></div></aside>
      </section>
      <section className="learning-grid"><div className="lesson-intro"><p className="eyebrow">THE 60-SECOND LESSON</p><h2>How movement becomes data</h2><p>MoveNet does not see a person as a photo. It estimates the position of 17 body landmarks, then we compare those positions over time.</p></div><div className="lesson-steps"><article><span>01</span><div><h3>Dots become a skeleton</h3><p>Shoulders, elbows, hips, knees, and ankles are represented as points with confidence scores.</p></div></article><article><span>02</span><div><h3>Time creates motion</h3><p>When a point changes position between frames, the app measures that change as movement intensity.</p></div></article><article><span>03</span><div><h3>A burst becomes a rep</h3><p>Each new spike in activity increments the counter. It is a simple learning signal, not a medical measurement.</p></div></article></div></section>
      <section className="history-section"><div className="section-title"><div><p className="eyebrow">YOUR FIELD NOTES</p><h2>Recent observations</h2></div><span>{observations.length}/5 saved</span></div>{observations.length ? <div className="history-list">{observations.map((item, index) => <div className="history-item" key={`${item.time}-${index}`}><span className="history-number">0{index + 1}</span><strong>{item.className}</strong><span className="history-confidence">{Math.round(item.probability * 100)}% confidence</span><time>{item.time}</time></div>)}</div> : <div className="empty-history">Use Scan object to add visual observations here.</div>}</section>
      <footer><span>Lens Lab / a small lesson in seeing machines</span><span>Runs locally in your browser</span></footer>
    </main>
  )
}

export default App
