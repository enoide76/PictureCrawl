// Gemäldeagent Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';

let selectedFile = null;
let analysisResult = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImageBtn = document.getElementById('removeImage');
const notesInput = document.getElementById('notes');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const generateReportBtn = document.getElementById('generateReportBtn');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
removeImageBtn.addEventListener('click', clearImage);
analyzeBtn.addEventListener('click', analyzeArtwork);
generateReportBtn.addEventListener('click', generateReport);
analyzeAnotherBtn.addEventListener('click', reset);

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        document.querySelector('.upload-content').style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    fileInput.value = '';
    document.querySelector('.upload-content').style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
}

async function analyzeArtwork() {
    if (!selectedFile) return;

    // Show loading
    loadingIndicator.style.display = 'block';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('notes', notesInput.value);

        const response = await fetch(`${API_BASE_URL}/api/analyze-image`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success && data.result) {
            analysisResult = data.result;
            displayResults(data.result);
        } else {
            alert(`Analysis failed: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to API. Make sure the backend is running on ' + API_BASE_URL);
    } finally {
        loadingIndicator.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function displayResults(result) {
    // Style & Epoch
    document.getElementById('epoch').textContent = result.style_estimation.epoch;
    document.getElementById('style').textContent = result.style_estimation.style;
    document.getElementById('styleConfidence').textContent =
        (result.style_estimation.confidence * 100).toFixed(1) + '%';

    // Artist Candidates
    const artistList = document.getElementById('artistList');
    artistList.innerHTML = '';

    if (result.artist_candidates && result.artist_candidates.length > 0) {
        result.artist_candidates.forEach(artist => {
            const item = document.createElement('div');
            item.className = 'artist-item';
            item.innerHTML = `
                <div class="artist-name">${artist.name}</div>
                <div class="artist-similarity">${(artist.similarity * 100).toFixed(1)}%</div>
            `;
            artistList.appendChild(item);
        });
    } else {
        artistList.innerHTML = '<p>No artist matches found</p>';
    }

    // Authenticity
    const authScore = result.authenticity_score;
    const authCircle = document.getElementById('authenticityCircle');
    document.getElementById('authenticityScore').textContent = authScore;

    authCircle.classList.remove('high', 'medium', 'low');
    if (authScore >= 80) {
        authCircle.classList.add('high');
    } else if (authScore >= 60) {
        authCircle.classList.add('medium');
    } else {
        authCircle.classList.add('low');
    }

    // Condition
    setBadge('craquele', result.condition.craquele);
    setBadge('yellowing', result.condition.yellowing);
    setBadge('stains', result.condition.stains);
    document.getElementById('damageScore').textContent =
        (result.condition.damage_score * 100).toFixed(1) + '%';
    document.getElementById('conditionNotes').textContent = result.condition.notes;

    // Valuation
    document.getElementById('estimatedValue').textContent =
        '€' + result.valuation.estimated_value.toLocaleString('en-US', {maximumFractionDigits: 0});
    document.getElementById('valueRange').textContent =
        `€${result.valuation.min.toLocaleString('en-US', {maximumFractionDigits: 0})} - €${result.valuation.max.toLocaleString('en-US', {maximumFractionDigits: 0})}`;

    const confBadge = document.getElementById('valuationConfidence');
    confBadge.textContent = result.valuation.confidence;
    confBadge.className = result.valuation.confidence;

    document.getElementById('valuationRationale').textContent = result.valuation.rationale;

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function setBadge(elementId, value) {
    const element = document.getElementById(elementId);
    element.textContent = value ? 'Yes' : 'No';
    element.className = 'badge ' + (value ? 'yes' : 'no');
}

async function generateReport() {
    if (!selectedFile || !analysisResult) return;

    generateReportBtn.disabled = true;
    generateReportBtn.textContent = 'Generating...';

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('notes', notesInput.value);
        formData.append('format', 'pdf');

        const response = await fetch(`${API_BASE_URL}/api/analyze-and-report`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success && data.download_url) {
            // Download the report
            window.open(API_BASE_URL + data.download_url, '_blank');
            alert('Report generated successfully!');
        } else {
            alert('Failed to generate report');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate report');
    } finally {
        generateReportBtn.disabled = false;
        generateReportBtn.textContent = 'Generate PDF Report';
    }
}

function reset() {
    clearImage();
    notesInput.value = '';
    resultsSection.style.display = 'none';
    analysisResult = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
