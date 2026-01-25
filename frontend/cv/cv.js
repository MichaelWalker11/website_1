const width = 320; // We will scale the photo width to this
let height = 0; // This will be computed based on the input stream

let streaming = false;


const video = document.getElementById("video");
const photo = document.getElementById("photo");
const startButton = document.getElementById("start-button");
const allowButton = document.getElementById("permissions-button");
const stopButton = document.getElementById("stop-button");
const canvas = document.createElement('canvas');
      

let minAgeSeen = Infinity;
let maxAgeSeen = 0;


let predictLoopOn = false;
let lastPostAt = 0;
let intervalMs = 500;
let inFlight = false; // prevents piling up requests

function startPredictLoop() {
  if (predictLoopOn) return;
  predictLoopOn = true;
  lastPostAt = 0;

  (function loop() {
    if (!predictLoopOn) return;

    const now = Date.now();
    if (!inFlight && (now - lastPostAt >= intervalMs)) {
      lastPostAt = now;
      postCurrentFrame(); // fire + guarded by inFlight
    }

    window.requestAnimationFrame(loop);
  })();
}

function stopPredictLoop() {
  predictLoopOn = false;
}

function canvasToBlob(mime = "image/jpeg", quality = 0.9) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) return reject(new Error("canvas.toBlob returned null"));
      resolve(blob);
    }, mime, quality);
  });
}

async function postCurrentFrame() {
  inFlight = true;

  try {
    // draw current frame to canvas
    const context = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    // OPTIONAL: show preview (remove if you don't want it)
    // const previewUrl = canvas.toDataURL("image/png");
    // if (photo) photo.setAttribute("src", previewUrl);

    // send to API
    const blob = await canvasToBlob("image/jpeg", 0.9);

    const [ageResult, genderResult] = await Promise.all([
      postImageToEndpoint(blob, "/predict/age"),
      postImageToEndpoint(blob, "/predict/gender")
    ]);

    // update text (reuse your formatting)
    renderOutputs(ageResult, genderResult);

    // OPTIONAL: refresh plots if you still want those PNGs
    refreshPlots();
  } catch (err) {
    console.error("Prediction failed:", err);
  } finally {
    inFlight = false;
  }
}

function renderOutputs(ageResult, genderResult) {
  const out = document.getElementById("age-output");

  const predictedAge =
    (ageResult.predicted_age !== undefined) ? ageResult.predicted_age
    : (ageResult.expected_age !== undefined) ? Math.round(ageResult.expected_age)
    : null;

  if (predictedAge !== null) {
    minAgeSeen = Math.min(minAgeSeen, predictedAge);
    maxAgeSeen = Math.max(maxAgeSeen, predictedAge);
  }

  const minText = (minAgeSeen === Infinity) ? "—" : String(minAgeSeen);
  const maxText = (maxAgeSeen === 0) ? "—" : String(maxAgeSeen);

  let ageText = "";
  if (predictedAge !== null) {
    if (ageResult.expected_age !== undefined) {
      ageText = `Min Age: ${minText} | Max Age: ${maxText} | Current Prediction: ${predictedAge} | Expected: ${ageResult.expected_age.toFixed(1)}`;
    } else {
      ageText = `Min Age: ${minText} | Max Age: ${maxText} | Current Prediction: ${predictedAge}`;
    }
  } else {
    ageText = `Min Age: ${minText} | Max Age: ${maxText} | Predicted: ${ageResult.predicted_class ?? "—"}`;
  }

  const g = genderResult.predicted_gender ?? "—";
  const conf = (genderResult.confidence !== undefined)
    ? `${(genderResult.confidence * 100).toFixed(1)}%`
    : "—";

  if (out) out.textContent = `${ageText} | Gender: ${g} (${conf})`;
}

    
stopButton.addEventListener("click", () => {
    const stream = video.srcObject; 
    stopPredictLoop();
  if (stream) {
    stream.getTracks().forEach(track => track.stop()); 
  }
  video.srcObject = null;
  video.pause();
});

allowButton.addEventListener("click", () => {
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    })
    .catch((err) => {
      console.error(`An error occurred: ${err}`);
    });
});

video.addEventListener("canplay", (ev) => {
  if (!streaming) {
    height = video.videoHeight / (video.videoWidth / width);

    video.setAttribute("width", width);
    video.setAttribute("height", height);
    
    streaming = true;

    startPredictLoop();
  }
});

startButton.addEventListener("click", (ev) => {
  takePicture();
  ev.preventDefault();
});

function clearPhoto() {
  const context = canvas.getContext("2d");
  context.fillStyle = "#aaaaaa";
  context.fillRect(0, 0, canvas.width, canvas.height);

  const data = canvas.toDataURL("image/png");
  photo.setAttribute("src", data);
}

clearPhoto();

function dataUrlToBlob(dataUrl) {
  const [meta, base64] = dataUrl.split(",");
  const mime = meta.match(/data:(.*);base64/)[1];
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}

function filenameForMime(mime) {
  if (mime === "image/jpeg") return "capture.jpg";
  if (mime === "image/webp") return "capture.webp";
  return "capture.png";
}


async function postImageToEndpoint(blob, endpointPath) {
  const formData = new FormData();
  formData.append("file", blob, filenameForMime(blob.type));

  const res = await fetch(`http://localhost:8000${endpointPath}`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return await res.json();
}
function refreshPlots() {
  document.getElementById("genderPlot").src = "plots/plot_gender.png?t=" + Date.now();
  document.getElementById("agePlot").src = "plots/plot_age.png?t=" + Date.now();
}





