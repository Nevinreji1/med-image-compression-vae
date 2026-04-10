document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const processingZone = document.getElementById('processing-zone');
    const resultsZone = document.getElementById('results-zone');

    // UI Elements
    const originalImg = document.getElementById('original-img');
    const reconstructedImg = document.getElementById('reconstructed-img');
    const latentImg = document.getElementById('latent-img');
    const originalSize = document.getElementById('original-size');
    const compressedSize = document.getElementById('compressed-size');
    const compRatio = document.getElementById('comp-ratio');
    const psnrVal = document.getElementById('psnr-val');

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }

    // --- Click to Upload Logic ---
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Basic validation
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Please upload a valid PNG or JPEG medical scan.');
            return;
        }

        uploadFile(file);
    }

    // --- API Communication ---
    async function uploadFile(file) {
        // Switch UI to processing mode
        uploadZone.classList.add('hidden');
        processingZone.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/compress', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'API Error occurred');
            }

            // Populate the UI with real data
            originalImg.src = data.original_image;
            reconstructedImg.src = data.reconstructed_image;
            latentImg.src = data.latent_image;
            
            originalSize.innerText = data.original_size;
            compressedSize.innerText = data.compressed_size;
            
            // Animate Numbers
            animateValue(compRatio, 0, data.compression_ratio, 1000);
            animateValue(psnrVal, 0, data.psnr, 1000);

            // Switch UI to results mode
            processingZone.classList.add('hidden');
            resultsZone.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert('Compression failed. See console for details.');
            
            // Reset UI on failure
            processingZone.classList.add('hidden');
            uploadZone.classList.remove('hidden');
        }
    }

    // --- Reset Logic ---
    resetBtn.addEventListener('click', () => {
        resultsZone.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileInput.value = ''; // Clear file input
    });

    // --- Utility: Number Animation ---
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(2);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
