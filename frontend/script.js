// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const previewTitle = document.getElementById('preview-title');
const resultsSection = document.getElementById('results-section');
const consoleEl = document.getElementById('console');
const viewTabs = document.getElementById('view-tabs');
const scanOverlay = document.getElementById('scan-overlay');
const exportBtn = document.getElementById('export-btn');
const themeToggle = document.getElementById('theme-toggle');

let currentFile = null;
let currentModality = null;
let lastResults = null;
let imageData = {
    original: null,
    neural: null,
    fft: null,
    ela: null
};

const imageExtensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
const videoExtensions = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
const audioExtensions = ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'];

// Theme Toggle
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);

    log(`Theme switched to ${newTheme} mode`);
});

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// Logging
function log(message) {
    const div = document.createElement('div');
    div.className = 'log';
    div.textContent = message;
    consoleEl.appendChild(div);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function setPlaceholderText(message) {
    const placeholder = previewContainer.querySelector('.placeholder');
    if (!placeholder) return;

    placeholder.classList.remove('hidden');
    const label = placeholder.querySelector('p');
    if (label) label.textContent = message;
}

function resetViewState() {
    resultsSection.classList.add('hidden');
    analyzeBtn.disabled = false;
    lastResults = null;
    imageData.neural = null;
    imageData.fft = null;
    imageData.ela = null;

    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.tab[data-view="original"]').classList.add('active');
}

// Demo Sample Loading
document.querySelectorAll('.demo-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
        const sample = btn.dataset.sample;
        log(`Loading ${sample} sample...`);

        try {
            const response = await fetch(`/samples/${sample}.png`);
            const blob = await response.blob();
            const file = new File([blob], `${sample}.png`, { type: 'image/png' });
            handleFile(file);
            log('Sample loaded successfully');
        } catch (error) {
            log('ERROR: Failed to load sample');
        }
    });
});

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    handleFile(e.dataTransfer.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

function handleFile(file) {
    if (!file) return;

    const extension = (file.name.split('.').pop() || '').toLowerCase();
    const inferredType = file.type.startsWith('image/')
        ? 'image'
        : file.type.startsWith('video/')
            ? 'video'
            : file.type.startsWith('audio/')
                ? 'audio'
                : imageExtensions.includes(extension)
                    ? 'image'
                    : videoExtensions.includes(extension)
                        ? 'video'
                        : audioExtensions.includes(extension)
                            ? 'audio'
                            : null;

    if (!inferredType) {
        alert('Unsupported file type');
        return;
    }

    currentFile = file;
    currentModality = inferredType;

    resetViewState();
    log(`Loaded: ${file.name} (${file.type || currentModality})`);

    if (currentModality === 'image') {
        previewTitle.textContent = 'Image Analysis';
        viewTabs.style.display = '';

        const reader = new FileReader();
        reader.onload = (e) => {
            imageData.original = e.target.result;
            showImage('original');
        };
        reader.readAsDataURL(file);
        return;
    }

    imageData.original = null;
    previewTitle.textContent = currentModality === 'video' ? 'Video Analysis' : 'Audio Analysis';
    viewTabs.style.display = 'none';
    previewImage.classList.add('hidden');
    setPlaceholderText(`Loaded ${currentModality}: ${file.name}`);
}

function showImage(type) {
    if (!imageData[type]) return;

    const placeholder = previewContainer.querySelector('.placeholder');
    if (placeholder) placeholder.classList.add('hidden');

    previewImage.src = imageData[type];
    previewImage.classList.remove('hidden');
}

// Tab switching
viewTabs.addEventListener('click', (e) => {
    if (!e.target.matches('.tab')) return;

    const view = e.target.dataset.view;
    if (!imageData[view]) {
        log(`${view} view not available yet`);
        return;
    }

    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    e.target.classList.add('active');
    showImage(view);
});

// Analysis
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('span').textContent = 'Analyzing...';
    scanOverlay.classList.remove('hidden');
    log(`Starting ${currentModality || 'media'} analysis...`);

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'Server error');
        }

        lastResults = data;

        if ((data.modality || currentModality) === 'image') {
            imageData.neural = `data:image/jpeg;base64,${data.overlay}`;
            imageData.fft = `data:image/jpeg;base64,${data.fft}`;
            imageData.ela = `data:image/jpeg;base64,${data.ela}`;
            log('Neural analysis complete');
            log('Spectral analysis complete');
            log('Artifact detection complete');
        } else if ((data.modality || currentModality) === 'video') {
            log(`Video sampling complete (${data.frames_analyzed || 0} frames)`);
        } else {
            log('Audio inference complete');
        }

        displayResults(data);
        log('Analysis complete');
    } catch (error) {
        console.error(error);
        log(`ERROR: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('span').textContent = 'Analyze File';
        scanOverlay.classList.add('hidden');
    }
});

function displayResults(data) {
    const modality = data.modality || currentModality || 'image';
    resultsSection.classList.remove('hidden');

    const verdict = document.getElementById('verdict');
    verdict.textContent = data.prediction;
    verdict.className = 'verdict ' + data.prediction.toLowerCase();

    const faceIndicator = document.getElementById('face-indicator');
    if (modality === 'audio') {
        faceIndicator.textContent = 'Audio analysis completed';
        faceIndicator.style.color = 'var(--text-secondary)';
    } else if (modality === 'video') {
        const framesAnalyzed = data.frames_analyzed ? ` | ${data.frames_analyzed} frames analyzed` : '';
        faceIndicator.textContent = data.face_detected
            ? `Face detected in sampled frames${framesAnalyzed}`
            : `No face detected in sampled frames${framesAnalyzed}`;
        faceIndicator.style.color = data.face_detected ? 'var(--success)' : 'var(--warning)';
    } else {
        faceIndicator.textContent = data.face_detected ? 'Face detected' : 'No face detected';
        faceIndicator.style.color = data.face_detected ? 'var(--success)' : 'var(--warning)';
    }

    document.getElementById('threat-fill').style.width = data.threat_level + '%';
    document.getElementById('threat-value').textContent = data.threat_level.toFixed(1) + '%';
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('real-prob').textContent = (data.real_prob * 100).toFixed(1) + '%';
    document.getElementById('fake-prob').textContent = (data.fake_prob * 100).toFixed(1) + '%';

    if (modality === 'image' && imageData.neural) {
        viewTabs.style.display = '';
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelector('.tab[data-view="neural"]').classList.add('active');
        showImage('neural');
    } else {
        viewTabs.style.display = 'none';
        previewImage.classList.add('hidden');
        setPlaceholderText(
            modality === 'video'
                ? `Video analyzed: ${currentFile.name}`
                : `Audio analyzed: ${currentFile.name}`
        );
    }

    log(`Verdict: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`);
    log(`Threat Level: ${data.threat_level.toFixed(1)}%`);
}

// Export Functionality
exportBtn.addEventListener('click', () => {
    if (!lastResults) {
        log('No results to export');
        return;
    }

    const report = {
        timestamp: new Date().toISOString(),
        filename: currentFile ? currentFile.name : 'unknown',
        modality: lastResults.modality || currentModality,
        analysis: {
            verdict: lastResults.prediction,
            confidence: `${(lastResults.confidence * 100).toFixed(2)}%`,
            threat_level: `${lastResults.threat_level.toFixed(2)}%`,
            face_detected: lastResults.face_detected
        },
        probabilities: {
            real: `${(lastResults.real_prob * 100).toFixed(2)}%`,
            fake: `${(lastResults.fake_prob * 100).toFixed(2)}%`
        },
        generated_by: 'DeepTrace AI Detector',
        version: '3.0'
    };

    if (typeof lastResults.frames_analyzed !== 'undefined') {
        report.analysis.frames_analyzed = lastResults.frames_analyzed;
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deeptrace_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    log('Report exported successfully');
});
