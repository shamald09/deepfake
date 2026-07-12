let fileInput = document.getElementById("fileElem");
let dropArea = document.getElementById("drop-area");
let preview = document.getElementById("preview");
let dropEmpty = document.getElementById("drop-empty");
let scanLine = document.getElementById("scan-line");
let analyzeBtn = document.getElementById("analyze-btn");
let btnLabel = document.getElementById("btn-label");

let idleState = document.getElementById("idle-state");
let resultState = document.getElementById("result-state");

dropArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFiles);

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("drag-active");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("drag-active");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("drag-active");
  fileInput.files = e.dataTransfer.files;
  handleFiles();
});

function handleFiles() {
  let file = fileInput.files[0];
  if (!file) return;

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  dropEmpty.style.display = "none";

  // reset previous result when a new image is chosen
  resultState.classList.add("hidden");
  idleState.classList.remove("hidden");
}

function uploadImage() {
  let file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first.");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  // loading state
  analyzeBtn.disabled = true;
  btnLabel.innerText = "Analyzing...";
  scanLine.classList.remove("hidden");
  idleState.classList.add("hidden");
  resultState.classList.add("hidden");

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze image";

    if (data.error) {
      alert(data.error);
      idleState.classList.remove("hidden");
      return;
    }

    renderResult(data.result, data.confidence);
  })
  .catch(err => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze image";
    idleState.classList.remove("hidden");
    alert("Error connecting to server");
    console.error(err);
  });
}

function renderResult(resultText, confidence) {
  let pct = (confidence * 100).toFixed(1);
  let isReal = resultText.toLowerCase().includes("real");

  document.getElementById("result-text").innerText = resultText;
  document.getElementById("confidence-text").innerText = pct + "%";

  let dot = document.getElementById("verdict-dot");
  let bar = document.getElementById("bar-fill");
  let color = isReal ? "var(--verdict-real)" : "var(--verdict-ai)";

  dot.style.background = color;
  bar.style.background = color;
  bar.style.width = pct + "%";

  resultState.classList.remove("hidden");
}