import { Classifier, Smoother, normalizeLandmarks } from "./classifier.js";

const MEDIAPIPE_VERSION = "1.0.1";
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Wait for a few frames before answering, and tolerate short detection dropouts.
const MIN_FRAMES = 5;
const MAX_FRAMES_WITHOUT_HAND = 5;

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const startButton = $("start");

let classifier;
let metadata;
let smoother;
let handLandmarker;
let HandLandmarkerClass;
let framesWithoutHand = 0;
let lastVideoTime = -1;
const gestureRows = [];
const fps = { frames: 0, since: performance.now(), value: 0 };

async function loadModel() {
  [metadata, classifier] = await Promise.all([
    fetch("model/gestures.json").then((r) => r.json()),
    Classifier.load("model/classifier.json"),
  ]);
  smoother = new Smoother(metadata.smoothingWindow);
  document.querySelector("[data-window]").textContent = metadata.smoothingWindow;
  document.querySelector("[data-threshold]").textContent = `${Math.round(metadata.minConfidence * 100)}%`;
  renderGestureList();
}

function renderGestureList() {
  const list = $("gestures");
  for (const gesture of metadata.gestures) {
    const item = document.createElement("li");
    item.className = "gesture";
    item.innerHTML = `
      <span class="gesture-name"></span>
      <span class="gesture-score">–</span>
      <span class="gesture-shape"></span>
      <div class="meter" aria-hidden="true"><div class="meter-fill"></div></div>`;
    item.querySelector(".gesture-name").textContent = gesture.name;
    const meaning = document.createElement("small");
    meaning.textContent = gesture.meaning;
    item.querySelector(".gesture-name").append(meaning);
    item.querySelector(".gesture-shape").textContent = gesture.handShape;
    list.append(item);
    gestureRows.push({
      item,
      score: item.querySelector(".gesture-score"),
      fill: item.querySelector(".meter-fill"),
    });
  }
}

async function createHandLandmarker() {
  const vision = await import(`${MEDIAPIPE_URL}/vision_bundle.mjs`);
  HandLandmarkerClass = vision.HandLandmarker;
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE_URL}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  try {
    return await vision.HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return vision.HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

async function start() {
  startButton.disabled = true;
  $("stage-text").textContent = "Starting camera…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    $("stage-text").textContent = "Loading hand model…";
    handLandmarker ??= await createHandLandmarker();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    $("stage-message").hidden = true;
    $("status").hidden = false;
    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    $("stage-text").textContent =
      error.name === "NotAllowedError"
        ? "Camera access was blocked. Allow it in your browser settings and try again."
        : `Could not start: ${error.message}`;
    startButton.disabled = false;
  }
}

function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());
    const hand = result.landmarks[0];
    process(hand);
    draw(hand);
    tickFps();
  }
  requestAnimationFrame(loop);
}

function process(hand) {
  if (hand) {
    framesWithoutHand = 0;
    smoother.push(classifier.predict(normalizeLandmarks(hand, video.videoWidth, video.videoHeight)));
  } else if (++framesWithoutHand > MAX_FRAMES_WITHOUT_HAND) {
    smoother.clear();
  }

  const average = smoother.count >= MIN_FRAMES ? smoother.average() : null;
  let best = -1;
  if (average) {
    best = average.indexOf(Math.max(...average));
  }

  gestureRows.forEach((row, i) => {
    const p = average ? average[i] : 0;
    row.score.textContent = average ? `${Math.round(p * 100)}%` : "–";
    row.fill.style.width = `${p * 100}%`;
    row.item.classList.toggle("active", i === best && average[best] >= metadata.minConfidence);
  });

  if (average && average[best] >= metadata.minConfidence) {
    const gesture = metadata.gestures[best];
    showPrediction(gesture.name, gesture.meaning, average[best]);
  } else if (hand || smoother.count > 0) {
    showPrediction("—", "Hold the sign steady", average ? average[best] : 0);
  } else {
    showPrediction("—", "No hand in view", 0);
  }
}

function showPrediction(name, meaning, confidence) {
  $("prediction-name").textContent = name;
  $("prediction-meaning").textContent = meaning;
  $("prediction-meter").style.width = `${confidence * 100}%`;
  $("prediction-confidence").textContent = confidence ? `${Math.round(confidence * 100)}% confidence` : " ";
}

function draw(hand) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!hand) return;
  const style = getComputedStyle(document.documentElement);
  const w = overlay.width;
  const h = overlay.height;

  ctx.lineWidth = Math.max(2, w / 240);
  ctx.strokeStyle = style.getPropertyValue("--bone");
  ctx.beginPath();
  for (const { start, end } of HandLandmarkerClass.HAND_CONNECTIONS) {
    ctx.moveTo(hand[start].x * w, hand[start].y * h);
    ctx.lineTo(hand[end].x * w, hand[end].y * h);
  }
  ctx.stroke();

  ctx.fillStyle = style.getPropertyValue("--accent");
  const radius = Math.max(3, w / 160);
  for (const point of hand) {
    ctx.beginPath();
    ctx.arc(point.x * w, point.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function tickFps() {
  fps.frames++;
  const now = performance.now();
  if (now - fps.since >= 1000) {
    fps.value = Math.round((fps.frames * 1000) / (now - fps.since));
    fps.frames = 0;
    fps.since = now;
    $("status").textContent = `${fps.value} fps · on-device`;
  }
}

startButton.addEventListener("click", start);
loadModel().catch((error) => {
  console.error(error);
  $("stage-text").textContent = "Could not load the gesture model.";
  startButton.disabled = true;
});
