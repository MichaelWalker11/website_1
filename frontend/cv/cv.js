const width = 320; // We will scale the photo width to this
let height = 0; // This will be computed based on the input stream

let streaming = false;


const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const photo = document.getElementById("photo");
const startButton = document.getElementById("start-button");
const allowButton = document.getElementById("permissions-button");
const stopButton = document.getElementById("stop-button");

let minAgeSeen = Infinity
let maxAgeSeen = 0

stopButton.addEventListener("click", () => {
    const stream = video.srcObject; 
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
    canvas.setAttribute("width", width);
    canvas.setAttribute("height", height);
    streaming = true;
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
async function sendToApi(dataUrl) {
  // Convert the dataURL to a Blob
  const blob = await (await fetch(dataUrl)).blob();

  const formData = new FormData();
  formData.append("file", blob, "capture.png"); // key must be "file" to match FastAPI

  const res = await fetch("http://localhost:8000/predict/age", {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }

  return await res.json();
}

async function takePicture() {
  const context = canvas.getContext("2d");
  if (width && height) {
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    const data = canvas.toDataURL("image/png");
    if (photo) photo.setAttribute("src", data);

    try {
      console.log("Sending to API...");
      const result = await sendToApi(data);
      console.log("API result:", result);

      const out = document.getElementById("age-output");

      // prefer integer predicted_age; fallback to rounded expected_age; fallback to null
      const predictedAge =
        (result.predicted_age !== undefined) ? result.predicted_age
        : (result.expected_age !== undefined) ? Math.round(result.expected_age)
        : null;

      if (predictedAge !== null) {
        minAgeSeen = Math.min(minAgeSeen, predictedAge);
        maxAgeSeen= Math.max(maxAgeSeen, predictedAge)
      }

      if (out) {
        const minText = (minAgeSeen === Infinity) ? "—" : String(minAgeSeen);
        const maxText = (maxAgeSeen === 0) ? "-" : String(maxAgeSeen);
        if (predictedAge !== null) {
          // show min first, then predicted + expected
          if (result.expected_age !== undefined) {
            out.textContent = `Min Age: ${minText} | Max Age: ${maxText} | Current Prediction: ${predictedAge} | Expected: ${result.expected_age.toFixed(1)}`;
          } else {
            out.textContent = `Min Age: ${minText} | Max Age: ${maxText} | Current Prediction: ${predictedAge}`;
          }
        } else {
          // if API doesn't return numeric ages
          out.textContent = `Min Age: ${minText} | Predicted: ${result.predicted_class}`;
        }
      }

    } catch (err) {
      console.error("Prediction failed:", err);
    }

  } else {
    clearPhoto();
  }
}




