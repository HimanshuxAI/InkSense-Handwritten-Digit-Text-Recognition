/**
 * InkSense — Frontend JavaScript
 * Handles canvas drawing, file uploads, API calls, and UI interactions
 */

const API_BASE = window.location.origin;

// ─── Navbar Scroll Effect ───────────────────────
window.addEventListener('scroll', () => {
    document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 10);
});

// ─── Smooth Nav Links ───────────────────────────
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

// ═══════════════════════════════════════════════
//  DIGIT RECOGNITION — Canvas Drawing
// ═══════════════════════════════════════════════

const digitCanvas = document.getElementById('digit-canvas');
const ctx = digitCanvas.getContext('2d');
let isDrawing = false;
let hasDrawn = false;

function initCanvas() {
    // Set actual canvas resolution
    const rect = digitCanvas.getBoundingClientRect();
    digitCanvas.width = 280;
    digitCanvas.height = 280;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

function getCanvasCoords(e) {
    const rect = digitCanvas.getBoundingClientRect();
    const scaleX = digitCanvas.width / rect.width;
    const scaleY = digitCanvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

digitCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    hasDrawn = true;
    const coords = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
});

digitCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
});

digitCanvas.addEventListener('mouseup', () => { isDrawing = false; });
digitCanvas.addEventListener('mouseleave', () => { isDrawing = false; });

// Touch events
digitCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    const coords = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
});

digitCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
});

digitCanvas.addEventListener('touchend', () => { isDrawing = false; });

// Clear canvas
document.getElementById('digit-clear').addEventListener('click', () => {
    initCanvas();
    hasDrawn = false;
    showPlaceholder('digit');
});

// Predict from canvas
document.getElementById('digit-predict-canvas').addEventListener('click', async () => {
    if (!hasDrawn) return;
    const dataURL = digitCanvas.toDataURL('image/png');
    await predictDigit(dataURL, 'canvas');
});

// ─── Digit Tab Switching ────────────────────────
document.getElementById('digit-tab-canvas').addEventListener('click', () => {
    document.getElementById('digit-tab-canvas').classList.add('active');
    document.getElementById('digit-tab-upload').classList.remove('active');
    document.getElementById('digit-canvas-area').classList.remove('hidden');
    document.getElementById('digit-upload-area').classList.add('hidden');
});

document.getElementById('digit-tab-upload').addEventListener('click', () => {
    document.getElementById('digit-tab-upload').classList.add('active');
    document.getElementById('digit-tab-canvas').classList.remove('active');
    document.getElementById('digit-upload-area').classList.remove('hidden');
    document.getElementById('digit-canvas-area').classList.add('hidden');
});

// ═══════════════════════════════════════════════
//  FILE UPLOAD HANDLERS
// ═══════════════════════════════════════════════

let digitFile = null;
let textFile = null;

function setupDropZone(dropZoneId, fileInputId, previewContainerId, previewImgId, removeId, predictBtnId, fileVar) {
    const dropZone = document.getElementById(dropZoneId);
    const fileInput = document.getElementById(fileInputId);
    const previewContainer = document.getElementById(previewContainerId);
    const previewImg = document.getElementById(previewImgId);
    const removeBtn = document.getElementById(removeId);
    const predictBtn = document.getElementById(predictBtnId);

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file, dropZone, previewContainer, previewImg, predictBtn, fileVar);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, dropZone, previewContainer, previewImg, predictBtn, fileVar);
        }
    });

    removeBtn.addEventListener('click', () => {
        if (fileVar === 'digit') digitFile = null;
        else textFile = null;
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        predictBtn.disabled = true;
        fileInput.value = '';
    });
}

function handleFileSelect(file, dropZone, previewContainer, previewImg, predictBtn, fileVar) {
    if (fileVar === 'digit') digitFile = file;
    else textFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('hidden');
        dropZone.classList.add('hidden');
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Setup drop zones
setupDropZone('digit-drop-zone', 'digit-file-input', 'digit-preview-container', 'digit-preview-img', 'digit-remove-img', 'digit-predict-upload', 'digit');
setupDropZone('text-drop-zone', 'text-file-input', 'text-preview-container', 'text-preview-img', 'text-remove-img', 'text-predict-upload', 'text');

// ─── Digit Upload Predict ───────────────────────
document.getElementById('digit-predict-upload').addEventListener('click', async () => {
    if (!digitFile) return;
    const formData = new FormData();
    formData.append('image', digitFile);
    await predictDigitFromFile(formData);
});

// ─── Text Upload Predict ────────────────────────
document.getElementById('text-predict-upload').addEventListener('click', async () => {
    if (!textFile) return;
    const formData = new FormData();
    formData.append('image', textFile);
    formData.append('line_mode', document.getElementById('line-mode-toggle').checked);
    await predictTextFromFile(formData);
});

// ═══════════════════════════════════════════════
//  API CALLS
// ═══════════════════════════════════════════════

async function predictDigit(dataURL, source) {
    showLoading('digit');
    try {
        const res = await fetch(`${API_BASE}/api/predict-canvas`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        const data = await res.json();
        if (data.success) {
            showDigitResult(data.prediction);
        } else {
            showError('digit', data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('digit', 'Server not available. Please start the backend.');
    }
}

async function predictDigitFromFile(formData) {
    showLoading('digit');
    try {
        const res = await fetch(`${API_BASE}/api/predict-digit`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            showDigitResult(data.prediction);
        } else {
            showError('digit', data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('digit', 'Server not available. Please start the backend.');
    }
}

async function predictTextFromFile(formData) {
    showLoading('text');
    try {
        const res = await fetch(`${API_BASE}/api/predict-text`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.success) {
            showTextResult(data.prediction);
        } else {
            showError('text', data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('text', 'Server not available. Please start the backend.');
    }
}

// ═══════════════════════════════════════════════
//  UI STATE MANAGEMENT
// ═══════════════════════════════════════════════

function showLoading(type) {
    document.getElementById(`${type}-placeholder`).classList.add('hidden');
    document.getElementById(`${type}-result`).classList.add('hidden');
    document.getElementById(`${type}-loading`).classList.remove('hidden');
}

function showPlaceholder(type) {
    document.getElementById(`${type}-placeholder`).classList.remove('hidden');
    document.getElementById(`${type}-result`).classList.add('hidden');
    document.getElementById(`${type}-loading`).classList.add('hidden');
}

function showError(type, message) {
    document.getElementById(`${type}-loading`).classList.add('hidden');
    document.getElementById(`${type}-placeholder`).classList.remove('hidden');
    document.getElementById(`${type}-result`).classList.add('hidden');
    const ph = document.getElementById(`${type}-placeholder`);
    ph.querySelector('p').textContent = message;
}

function showDigitResult(prediction) {
    document.getElementById('digit-loading').classList.add('hidden');
    document.getElementById('digit-placeholder').classList.add('hidden');
    document.getElementById('digit-result').classList.remove('hidden');

    document.getElementById('digit-predicted-value').textContent = prediction.digit;
    document.getElementById('digit-confidence').textContent = prediction.confidence + '%';

    // Build probability bars
    const container = document.getElementById('digit-probabilities');
    container.innerHTML = '';
    for (let i = 0; i <= 9; i++) {
        const prob = prediction.probabilities[String(i)] || 0;
        const isActive = i === prediction.digit;
        container.innerHTML += `
            <div class="prob-row">
                <span class="prob-label">${i}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar ${isActive ? 'active' : ''}" style="width: ${prob}%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            </div>`;
    }
}

function showTextResult(prediction) {
    document.getElementById('text-loading').classList.add('hidden');
    document.getElementById('text-placeholder').classList.add('hidden');
    document.getElementById('text-result').classList.remove('hidden');

    document.getElementById('text-predicted-value').textContent = `"${prediction.text}"`;
    document.getElementById('text-confidence').textContent = prediction.probability + '%';
}

// ─── Init ───────────────────────────────────────
initCanvas();
