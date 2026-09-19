import { useEffect, useRef, useState } from 'react'
import * as facemesh from '@tensorflow-models/facemesh'
import * as handpose from '@tensorflow-models/handpose'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as posenet from '@tensorflow-models/posenet'
import * as tf from '@tensorflow/tfjs'
import './App.css'

type Prediction = { className: string; probability: number }
type Observation = Prediction & { time: string }
type HandResult = { handInViewConfidence: number; landmarks: number[][]; annotations: Record<string, number[][]> }
type FaceResult = { scaledMesh: number[][] }
type FaceMeshModel = { estimateFaces: (video: HTMLVideoElement, options: { flipHorizontal: boolean }) => Promise<FaceResult[]>; dispose: () => void }

const SKELETON = [
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
]
const HAND_CONNECTIONS = [['thumb', 0], ['thumb', 1], ['thumb', 2], ['thumb', 3], ['thumb', 4], ['indexFinger', 0], ['indexFinger', 5], ['indexFinger', 6], ['indexFinger', 7], ['indexFinger', 8], ['middleFinger', 0], ['middleFinger', 9], ['middleFinger', 10], ['middleFinger', 11], ['middleFinger', 12], ['ringFinger', 0], ['ringFinger', 13], ['ringFinger', 14], ['ringFinger', 15], ['ringFinger', 16], ['pinky', 0], ['pinky', 17], ['pinky', 18], ['pinky', 19], ['pinky', 20]] as const

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const classifierRef = useRef<mobilenet.MobileNet | null>(null)
  const detectorRef = useRef<posenet.PoseNet | null>(null)
  const handDetectorRef = useRef<handpose.HandPose | null>(null)
  const faceDetectorRef = useRef<FaceMeshModel | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationRef = useRef<number | null>(null)
  const previousKeypointsRef = useRef<posenet.Keypoint[] | null>(null)
  const previousActiveRef = useRef(false)
  const cachedHandsRef = useRef<HandResult[]>([])
  const cachedFacesRef = useRef<FaceResult[]>([])
  const frameCountRef = useRef(0)
  const lastPoseRef = useRef<posenet.Pose | null>(null)
  const processingRef = useRef(false)
  const lastStatusFrameRef = useRef(0)
  const [cameraReady, setCameraReady] = useState(false)
  const [modelsReady, setModelsReady] = useState(false)
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState('')
  const [movementScore, setMovementScore] = useState(0)
  const [personDetected, setPersonDetected] = useState(false)
  const [handsDetected, setHandsDetected] = useState(0)
  const [faceDetected, setFaceDetected] = useState(false)
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
        const detector = await tf.ready().then(async () => {
          try { await tf.setBackend('webgl') } catch { }
          await tf.ready()
          return posenet.load({ architecture: 'MobileNetV1', outputStride: 16, inputResolution: { width: 320, height: 240 }, multiplier: 0.5 })
        })
        if (!active) return
        detectorRef.current = detector
        setModelsReady(true)
        const [classifier, handDetector, faceDetector] = await Promise.allSettled([mobilenet.load({ version: 2, alpha: 1.0 }), handpose.load(), facemesh.load({ maxFaces: 1 })])
        if (!active) return
        if (classifier.status === 'fulfilled') classifierRef.current = classifier.value
        if (handDetector.status === 'fulfilled') handDetectorRef.current = handDetector.value
        if (faceDetector.status === 'fulfilled') faceDetectorRef.current = faceDetector.value as FaceMeshModel
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
      classifierRef.current = null
      faceDetectorRef.current?.dispose()
    }
  }, [])

  useEffect(() => {
    if (!modelsReady || !cameraReady) return
    let active = true
    async function trackFrame() {
      const video = videoRef.current
      const detector = detectorRef.current
      const handDetector = handDetectorRef.current
      const faceDetector = faceDetectorRef.current
      const canvas = canvasRef.current
      if (!active || !video || !detector || !canvas || video.readyState < 2) return
      if (processingRef.current) {
        animationRef.current = requestAnimationFrame(() => { void trackFrame() })
        return
      }
      processingRef.current = true
      frameCountRef.current += 1
      const frameNumber = frameCountRef.current
      const poseUpdated = frameNumber % 2 === 0
      const pose = poseUpdated
        ? await detector.estimateSinglePose(video, { flipHorizontal: false })
        : lastPoseRef.current
      if (pose) lastPoseRef.current = pose
      if (handDetector && frameNumber % 5 === 0) cachedHandsRef.current = await handDetector.estimateHands(video, false) as HandResult[]
      if (faceDetector && frameNumber % 8 === 0) cachedFacesRef.current = await faceDetector.estimateFaces(video, { flipHorizontal: false })
      const hands = cachedHandsRef.current
      const faces = cachedFacesRef.current
      const visibleKeypoints = (pose?.keypoints ?? []).filter((keypoint: posenet.Keypoint) => (keypoint.score ?? 0) > 0.35)
      const personVisible = visibleKeypoints.length >= 5
      const typedHands = hands as HandResult[]
      if (frameNumber - lastStatusFrameRef.current >= 6) {
        setHandsDetected(typedHands.filter((hand) => hand.handInViewConfidence > 0.5).length)
        setFaceDetected(faces.length > 0)
        setPersonDetected(personVisible)
        lastStatusFrameRef.current = frameNumber
      }
      const context = canvas.getContext('2d')
      if (context) {
        const videoWidth = video.videoWidth || 640
        const videoHeight = video.videoHeight || 480
        if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
          canvas.width = videoWidth
          canvas.height = videoHeight
        }
        context.clearRect(0, 0, canvas.width, canvas.height)
        const drawBox = (points: Array<[number, number]>, color: string, label: string) => {
          if (!points.length) return
          const xs = points.map(([x]) => x)
          const ys = points.map(([, y]) => y)
          const minX = Math.max(8, Math.min(...xs) - 16)
          const minY = Math.max(8, Math.min(...ys) - 16)
          const maxX = Math.min(canvas.width - 8, Math.max(...xs) + 16)
          const maxY = Math.min(canvas.height - 8, Math.max(...ys) + 16)
          context.strokeStyle = color
          context.lineWidth = 2
          context.strokeRect(minX, minY, maxX - minX, maxY - minY)
          context.font = '600 12px DM Mono, monospace'
          const tagWidth = context.measureText(label).width + 16
          context.fillStyle = color
          context.fillRect(minX, Math.max(0, minY - 22), tagWidth, 22)
          context.fillStyle = '#111414'
          context.fillText(label, minX + 8, Math.max(15, minY - 7))
        }
        if (pose) {
          const points = new Map(pose.keypoints.map((keypoint: posenet.Keypoint) => [keypoint.part, keypoint]))
          context.strokeStyle = '#d5f36f'
          context.lineWidth = 4
          SKELETON.forEach(([startName, endName]) => {
            const start = points.get(startName)
            const end = points.get(endName)
            if (start && end && (start.score ?? 0) > 0.35 && (end.score ?? 0) > 0.35) {
              context.beginPath(); context.moveTo(start.position.x, start.position.y); context.lineTo(end.position.x, end.position.y); context.stroke()
            }
          })
          visibleKeypoints.forEach((keypoint) => { context.fillStyle = '#ff765f'; context.beginPath(); context.arc(keypoint.position.x, keypoint.position.y, 6, 0, Math.PI * 2); context.fill() })
          drawBox(visibleKeypoints.map((keypoint) => [keypoint.position.x, keypoint.position.y]), '#d5f36f', 'TRACK:BODY')
        }
          typedHands.forEach((hand) => {
          const annotations = hand.annotations
          context.strokeStyle = '#61e7ff'
          context.fillStyle = '#61e7ff'
          context.lineWidth = 2
          Object.values(annotations).forEach((finger) => finger.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, 3, 0, Math.PI * 2); context.fill() }))
          HAND_CONNECTIONS.forEach(([fingerName, index]) => {
            const finger = annotations[fingerName]
            const next = finger[index]
            const previous = index === 0 ? hand.landmarks[0] : finger[index - 1]
            if (next && previous) { context.beginPath(); context.moveTo(previous[0], previous[1]); context.lineTo(next[0], next[1]); context.stroke() }
          })
          drawBox(hand.landmarks.map(([x, y]) => [x, y]), '#61e7ff', 'TRACK:HAND')
        })
        faces.forEach((face) => {
          const mesh = face.scaledMesh as number[][]
          context.strokeStyle = '#ffca6b'
          context.fillStyle = '#ffca6b'
          context.lineWidth = 1.5
          mesh.forEach(([x, y], index) => { if (index % 3 === 0) { context.beginPath(); context.arc(x, y, 1.6, 0, Math.PI * 2); context.fill() } })
          const outline = [10, 67, 103, 127, 234, 323, 356, 389, 454, 10]
          context.beginPath(); outline.forEach((point, index) => { const [x, y] = mesh[point] ?? [0, 0]; if (index === 0) context.moveTo(x, y); else context.lineTo(x, y) }); context.stroke()
          drawBox(mesh.map(([x, y]) => [x, y]), '#ffca6b', 'TRACK:FACE')
        })
      }
      if (poseUpdated && personVisible && previousKeypointsRef.current) {
        const previous = new Map(previousKeypointsRef.current.map((keypoint) => [keypoint.part, keypoint]))
        const totalMovement = visibleKeypoints.reduce((sum: number, keypoint: posenet.Keypoint) => {
          const old = previous.get(keypoint.part)
          return old ? sum + Math.hypot(keypoint.position.x - old.position.x, keypoint.position.y - old.position.y) : sum
        }, 0)
        const averageMovement = totalMovement / Math.max(visibleKeypoints.length, 1)
        const intensity = Math.min(100, Math.round(averageMovement * 3.2))
        if (frameNumber % 3 === 0) setMovementScore((current) => Math.round(current * 0.75 + intensity * 0.25))
        const isActive = intensity > 22
        if (isActive && !previousActiveRef.current) setReps((current) => current + 1)
        previousActiveRef.current = isActive
      } else if (poseUpdated && !personVisible) {
        if (frameNumber % 3 === 0) setMovementScore((current) => Math.max(0, current - 2))
        previousActiveRef.current = false
      }
      previousKeypointsRef.current = pose?.keypoints ?? null
      processingRef.current = false
      animationRef.current = requestAnimationFrame(() => {
        void trackFrame().catch(() => {
          if (active) {
            processingRef.current = false
            setError('One tracking frame was skipped. Keep the camera visible and tracking will continue.')
            animationRef.current = requestAnimationFrame(() => { void trackFrame().catch(() => undefined) })
          }
        })
      })
    }
    void trackFrame().catch(() => {
      if (active) setError('Tracking paused. Keep the camera visible and reload if this continues.')
    })
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
      <header className="topbar"><div className="brand"><span className="brand-mark">◉</span><span>Lens Lab</span></div><div className="model-status"><span className={`status-dot ${modelsReady ? 'ready' : ''}`}></span>{statusLabel}<span className="divider">/</span><span className="model-name">Body + Hands + Face</span></div></header>
      <section className="intro"><div><p className="eyebrow">CAMERA + MOVEMENT LEARNING</p><h1>Move your body.<br /><em>See it think.</em></h1></div><p className="intro-copy">Lens Lab watches body landmarks in real time, turns motion into a signal, and teaches you what the model can and cannot know.</p></section>
      <section className="workspace">
        <div className="camera-panel"><div className="camera-frame"><video ref={videoRef} muted playsInline aria-label="Live camera view" /><canvas ref={canvasRef} className="pose-overlay" aria-hidden="true" />{!cameraReady && <div className="camera-placeholder"><span className="loader">◌</span><p>{loading ? 'Opening your camera...' : 'Camera unavailable'}</p><small>{error || 'Allow camera access to begin.'}</small></div>}<div className="frame-corner corner-tl"></div><div className="frame-corner corner-tr"></div><div className="frame-corner corner-bl"></div><div className="frame-corner corner-br"></div><span className="live-label"><i></i> {cameraReady ? 'LIVE TRACKING' : 'WAITING'}</span><span className="frame-hud">AUTO TRACK / 3 CHANNELS<br /><b>BODY · HANDS · FACE</b></span><span className="frame-corner-target">+</span></div><div className="camera-actions"><div><span className="hint-key">SPACE</span><span className="action-note">Identify the object in frame</span></div><button className="scan-button" type="button" onClick={scanScene} disabled={!cameraReady || scanning}>{scanning ? 'Reading...' : 'Identify object ↗'}</button></div>{error && <p className="error-message">{error}</p>}</div>
        <aside className="results-panel"><div className="movement-card"><div className="panel-heading"><div><p className="eyebrow">MOVEMENT SIGNAL</p><h2>{personDetected ? 'You are in frame' : 'Waiting for a person'}</h2></div><span className="result-index">{String(reps).padStart(2, '0')}</span></div><div className="confidence-row"><span>Intensity</span><strong>{movementScore}%</strong></div><div className="confidence-track"><span style={{ width: `${movementScore}%` }}></span></div><div className="movement-meta"><span><b>{reps}</b> motion bursts</span><span className={personDetected ? 'signal-on' : ''}>● {personDetected ? 'Tracking' : 'No pose'}</span></div><div className="tracking-chips"><span className={faceDetected ? 'active' : ''}>◌ Face {faceDetected ? 'locked' : 'searching'}</span><span className={handsDetected ? 'active' : ''}>✧ {handsDetected} hand{handsDetected === 1 ? '' : 's'}</span><span className={personDetected ? 'active' : ''}>⌁ Body {personDetected ? 'locked' : 'searching'}</span></div></div>{topPrediction ? <div className="object-read"><p className="eyebrow">OBJECT READ</p><h2>{topPrediction.className}</h2><div className="confidence-row"><span>Confidence</span><strong>{Math.round(topPrediction.probability * 100)}%</strong></div><div className="confidence-track"><span style={{ width: `${topPrediction.probability * 100}%` }}></span></div><p className="result-explanation">This is a visual guess, not a fact. The model compares pixels with categories from its training data.</p>{predictions.slice(1).map((item) => <div className="possibility" key={item.className}><span>{item.className}</span><span>{Math.round(item.probability * 100)}%</span></div>)}</div> : <div className="empty-result"><span>◎</span><p>Object scans appear here.</p><small>Movement tracking is already live above.</small></div>}<div className="privacy-note"><span>⌁</span><p><strong>Private by design</strong><br />Video is processed in this tab and never uploaded.</p></div></aside>
      </section>
      <section className="learning-grid"><div className="lesson-intro"><p className="eyebrow">THE 60-SECOND LESSON</p><h2>How movement becomes data</h2><p>MoveNet does not see a person as a photo. It estimates the position of 17 body landmarks, then we compare those positions over time.</p></div><div className="lesson-steps"><article><span>01</span><div><h3>Dots become a skeleton</h3><p>Shoulders, elbows, hips, knees, and ankles are represented as points with confidence scores.</p></div></article><article><span>02</span><div><h3>Time creates motion</h3><p>When a point changes position between frames, the app measures that change as movement intensity.</p></div></article><article><span>03</span><div><h3>A burst becomes a rep</h3><p>Each new spike in activity increments the counter. It is a simple learning signal, not a medical measurement.</p></div></article></div></section>
      <section className="history-section"><div className="section-title"><div><p className="eyebrow">YOUR FIELD NOTES</p><h2>Recent observations</h2></div><span>{observations.length}/5 saved</span></div>{observations.length ? <div className="history-list">{observations.map((item, index) => <div className="history-item" key={`${item.time}-${index}`}><span className="history-number">0{index + 1}</span><strong>{item.className}</strong><span className="history-confidence">{Math.round(item.probability * 100)}% confidence</span><time>{item.time}</time></div>)}</div> : <div className="empty-history">Use Scan object to add visual observations here.</div>}</section>
      <footer><span>Lens Lab / a small lesson in seeing machines</span><span>Runs locally in your browser</span></footer>
    </main>
  )
}

export default App
