document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const dropzoneContent = document.getElementById('dropzone-content');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');

    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const predictionBox = document.getElementById('prediction-box');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const resetBtn = document.getElementById('reset-btn');

    let currentFile = null;

    // --- Drag and Drop Handling ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // --- Click Upload Handling ---
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Validate file type
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropzoneContent.style.opacity = '0';
            analyzeBtn.disabled = false;
            
            // Reset results if a new image is loaded
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    // --- API Interaction ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading state
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        loader.classList.remove('hidden');
        resultsPlaceholder.classList.remove('hidden');
        resultsContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let msg = 'Analysis failed.';
                try {
                    const errBody = await response.json();
                    if (errBody.detail) {
                        msg = typeof errBody.detail === 'string'
                            ? errBody.detail
                            : JSON.stringify(errBody.detail);
                    }
                } catch (_) { /* ignore */ }
                throw new Error(msg);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error(error);
            alert(error.message || 'An error occurred during analysis. Make sure the backend server is running.');
        } finally {
            btnText.textContent = 'Analyze Image';
            loader.classList.add('hidden');
            if (currentFile) {
                analyzeBtn.disabled = false;
            }
        }
    });

    function displayResults(data) {
        resultsPlaceholder.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const isMalignant = data.prediction.toLowerCase() === 'malignant';
        const confidencePercentage = (data.confidence * 100).toFixed(1) + '%';

        // Update Prediction Box
        predictionLabel.textContent = data.prediction;
        predictionBox.className = 'prediction-box ' + (isMalignant ? 'malignant' : 'benign');

        // Update Confidence Bar
        confidenceValue.textContent = confidencePercentage;
        confidenceBar.style.width = confidencePercentage;
        confidenceBar.className = 'progress-bar-fill ' + (isMalignant ? 'malignant' : 'benign');
    }

    // --- Reset ---
    resetBtn.addEventListener('click', () => {
        resetResults();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        dropzoneContent.style.opacity = '1';
        analyzeBtn.disabled = true;
    });

    function resetResults() {
        resultsPlaceholder.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        predictionBox.className = 'prediction-box';
        confidenceBar.style.width = '0%';
    }
});
