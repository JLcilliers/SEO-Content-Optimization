// API Base URL - uses relative path for same-origin requests
const API_BASE = '';

// DOM Elements
const optimizerForm = document.getElementById('optimizer-form');
const resultsCard = document.getElementById('results');
const errorCard = document.getElementById('error-message');
const downloadBtn = document.getElementById('download-btn');
const submitBtn = document.getElementById('submit-btn');

// Content source toggle buttons
const sourceToggleBtns = document.querySelectorAll('.toggle-btn[data-source]');
const urlSource = document.getElementById('url-source');
const fileSource = document.getElementById('file-source');

// Keywords toggle buttons
const keywordsToggleBtns = document.querySelectorAll('.toggle-btn[data-keywords]');
const keywordsManualSource = document.getElementById('keywords-manual-source');
const keywordsTextSource = document.getElementById('keywords-text-source');
const keywordsFileSource = document.getElementById('keywords-file-source');

// File upload elements
const contentUpload = document.getElementById('content-upload');
const contentFileInput = document.getElementById('content-file');
const keywordsUpload = document.getElementById('keywords-upload');
const keywordsFileInput = document.getElementById('keywords-file');

// State
let currentDocumentBase64 = null;
let currentFileName = 'optimized-content.docx';
let currentSourceType = 'url';
let currentKeywordsType = 'text';

// Content source type toggle
sourceToggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const sourceType = btn.dataset.source;
        currentSourceType = sourceType;

        // Update button states
        sourceToggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update source input visibility
        if (sourceType === 'url') {
            urlSource.classList.add('active');
            fileSource.classList.remove('active');
        } else {
            urlSource.classList.remove('active');
            fileSource.classList.add('active');
        }

        // Hide results and errors when switching
        hideResults();
        hideError();
    });
});

// Keywords source type toggle
keywordsToggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const keywordsType = btn.dataset.keywords;
        currentKeywordsType = keywordsType;

        // Update button states
        keywordsToggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update keywords input visibility
        keywordsManualSource.classList.remove('active');
        keywordsTextSource.classList.remove('active');
        keywordsFileSource.classList.remove('active');

        if (keywordsType === 'manual') {
            keywordsManualSource.classList.add('active');
        } else if (keywordsType === 'text') {
            keywordsTextSource.classList.add('active');
        } else {
            keywordsFileSource.classList.add('active');
        }

        // Hide results and errors when switching
        hideResults();
        hideError();
    });
});

// File upload handling
function setupFileUpload(uploadEl, inputEl) {
    uploadEl.addEventListener('click', () => inputEl.click());

    uploadEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadEl.classList.add('dragover');
    });

    uploadEl.addEventListener('dragleave', () => {
        uploadEl.classList.remove('dragover');
    });

    uploadEl.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadEl.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            inputEl.files = e.dataTransfer.files;
            updateFileDisplay(uploadEl, inputEl.files[0]);
        }
    });

    inputEl.addEventListener('change', () => {
        if (inputEl.files.length) {
            updateFileDisplay(uploadEl, inputEl.files[0]);
        }
    });
}

function updateFileDisplay(uploadEl, file) {
    uploadEl.classList.add('has-file');
    uploadEl.querySelector('.upload-file-name').textContent = file.name;
}

setupFileUpload(contentUpload, contentFileInput);
setupFileUpload(keywordsUpload, keywordsFileInput);

// Parse keywords from text input
function parseKeywords(text) {
    const lines = text.trim().split('\n').filter(line => line.trim());
    return lines.map(line => {
        const parts = line.split(',').map(p => p.trim());
        const keyword = {
            phrase: parts[0]
        };

        if (parts[1] && !isNaN(parseInt(parts[1]))) {
            keyword.volume = parseInt(parts[1]);
        }
        if (parts[2] && !isNaN(parseInt(parts[2]))) {
            keyword.difficulty = parseInt(parts[2]);
        }
        if (parts[3]) {
            keyword.intent = parts[3];
        }

        return keyword;
    });
}

// Get manual keywords from form
function getManualKeywords() {
    const primary = document.getElementById('manual-primary').value.trim();
    const secondary1 = document.getElementById('manual-secondary-1').value.trim();
    const secondary2 = document.getElementById('manual-secondary-2').value.trim();
    const secondary3 = document.getElementById('manual-secondary-3').value.trim();

    if (!primary) {
        throw new Error('Please enter a primary keyword');
    }

    return {
        primary: primary,
        secondary: [secondary1, secondary2, secondary3].filter(s => s.length > 0)
    };
}

// Unified form submission
optimizerForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const faqCount = parseInt(document.getElementById('faq-count').value);
    const maxSecondary = parseInt(document.getElementById('max-secondary').value);

    // Show loading state
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    hideError();
    hideResults();

    try {
        // Handle manual keyword mode
        if (currentKeywordsType === 'manual') {
            const manualKeywords = getManualKeywords();

            if (currentSourceType === 'url') {
                await optimizeFromUrlWithManualKeywords(manualKeywords, faqCount, maxSecondary);
            } else {
                throw new Error('Manual keyword mode currently only supports URL content source');
            }
        }
        // Determine if we're using file upload for keywords
        else if (currentKeywordsType === 'file') {
            // File-based keywords - must use file endpoint
            await optimizeWithKeywordsFile(faqCount, maxSecondary);
        } else {
            // Text-based keywords
            const keywordsText = document.getElementById('keywords-text').value;
            const keywords = parseKeywords(keywordsText);

            if (keywords.length === 0) {
                throw new Error('Please enter at least one keyword');
            }

            if (currentSourceType === 'url') {
                await optimizeFromUrl(keywords, faqCount, maxSecondary);
            } else {
                await optimizeFromFileWithTextKeywords(keywords, faqCount, maxSecondary);
            }
        }
    } catch (error) {
        showError(error.message);
    } finally {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

// Optimize from URL with text keywords
async function optimizeFromUrl(keywords, faqCount, maxSecondary) {
    const sourceUrl = document.getElementById('source-url').value;

    if (!sourceUrl) {
        throw new Error('Please enter a URL');
    }

    const response = await fetch(`${API_BASE}/api/optimize/url`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            source_url: sourceUrl,
            keywords: keywords,
            generate_faq: faqCount > 0,
            faq_count: faqCount,
            max_secondary: maxSecondary
        })
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Failed to optimize content');
    }

    // Store document for download
    currentDocumentBase64 = data.document_base64;
    currentFileName = `optimized-${new URL(sourceUrl).hostname}.docx`;

    // Display results
    displayResults(data);
}

// Optimize from URL with manual keywords (bypasses auto-selection)
async function optimizeFromUrlWithManualKeywords(manualKeywords, faqCount, maxSecondary) {
    const sourceUrl = document.getElementById('source-url').value;

    if (!sourceUrl) {
        throw new Error('Please enter a URL');
    }

    const response = await fetch(`${API_BASE}/api/optimize/url`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            source_url: sourceUrl,
            keywords: [],  // Empty - using manual mode
            manual_keywords: manualKeywords,  // Pass manual keywords
            generate_faq: faqCount > 0,
            faq_count: faqCount,
            max_secondary: maxSecondary
        })
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Failed to optimize content');
    }

    // Store document for download
    currentDocumentBase64 = data.document_base64;
    currentFileName = `optimized-${new URL(sourceUrl).hostname}.docx`;

    // Display results
    displayResults(data);
}

// Optimize from file with text keywords
async function optimizeFromFileWithTextKeywords(keywords, faqCount, maxSecondary) {
    const contentFile = contentFileInput.files[0];

    if (!contentFile) {
        throw new Error('Please select a content document');
    }

    // Create form data with file and keywords as JSON
    const formData = new FormData();
    formData.append('file', contentFile);

    // Create a keywords file from the parsed keywords
    const keywordsJson = JSON.stringify(keywords);
    const keywordsBlob = new Blob([keywordsJson], { type: 'application/json' });
    formData.append('keywords_file', keywordsBlob, 'keywords.json');

    formData.append('generate_faq', faqCount > 0);
    formData.append('faq_count', faqCount);
    formData.append('max_secondary', maxSecondary);

    const response = await fetch(`${API_BASE}/api/optimize/file`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to optimize content');
    }

    // For file upload, response is the document directly
    const blob = await response.blob();
    currentDocumentBase64 = null;
    currentFileName = `optimized-${contentFile.name}`;

    // Create download link and trigger download
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentFileName;
    a.click();
    URL.revokeObjectURL(url);

    // Show success message
    showSuccess('Document optimized and downloaded successfully!');
}

// Optimize with keywords file (Excel/CSV)
async function optimizeWithKeywordsFile(faqCount, maxSecondary) {
    const keywordsFile = keywordsFileInput.files[0];

    if (!keywordsFile) {
        throw new Error('Please select a keywords file');
    }

    const formData = new FormData();
    formData.append('keywords_file', keywordsFile);
    formData.append('generate_faq', faqCount > 0);
    formData.append('faq_count', faqCount);
    formData.append('max_secondary', maxSecondary);

    if (currentSourceType === 'url') {
        // URL content with file keywords
        const sourceUrl = document.getElementById('source-url').value;
        if (!sourceUrl) {
            throw new Error('Please enter a URL');
        }
        formData.append('source_url', sourceUrl);

        const response = await fetch(`${API_BASE}/api/optimize/url-with-keywords-file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to optimize content');
        }

        const data = await response.json();

        // Store document for download
        currentDocumentBase64 = data.document_base64;
        currentFileName = `optimized-${new URL(sourceUrl).hostname}.docx`;

        // Display results
        displayResults(data);
    } else {
        // File content with file keywords
        const contentFile = contentFileInput.files[0];
        if (!contentFile) {
            throw new Error('Please select a content document');
        }

        formData.append('file', contentFile);

        const response = await fetch(`${API_BASE}/api/optimize/file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to optimize content');
        }

        // For file upload, response is the document directly
        const blob = await response.blob();
        currentDocumentBase64 = null;
        currentFileName = `optimized-${contentFile.name}`;

        // Create download link and trigger download
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = currentFileName;
        a.click();
        URL.revokeObjectURL(url);

        // Show success message
        showSuccess('Document optimized and downloaded successfully!');
    }
}

// Display results
function displayResults(data) {
    // Update summary
    document.getElementById('primary-keyword').textContent = data.primary_keyword || '-';
    document.getElementById('secondary-keywords').textContent =
        data.secondary_keywords?.join(', ') || '-';

    // Build meta table
    const metaTable = document.getElementById('meta-table');
    metaTable.innerHTML = `
        <div class="meta-row header">
            <div class="meta-cell">Element</div>
            <div class="meta-cell">Current</div>
            <div class="meta-cell">Optimized</div>
        </div>
    `;

    if (data.meta_elements) {
        data.meta_elements.forEach(meta => {
            const row = document.createElement('div');
            row.className = 'meta-row';
            row.innerHTML = `
                <div class="meta-cell element" data-label="Element">${meta.element_name}</div>
                <div class="meta-cell" data-label="Current">${meta.current || '-'}</div>
                <div class="meta-cell optimized" data-label="Optimized">${meta.optimized || '-'}</div>
            `;
            metaTable.appendChild(row);
        });
    }

    // Build FAQ section
    const faqSection = document.getElementById('faq-section');
    const faqList = document.getElementById('faq-list');
    faqList.innerHTML = '';

    if (data.faq_items && data.faq_items.length > 0) {
        faqSection.style.display = 'block';
        data.faq_items.forEach(faq => {
            const item = document.createElement('div');
            item.className = 'faq-item';
            item.innerHTML = `
                <div class="faq-question">${faq.question}</div>
                <div class="faq-answer">${faq.answer}</div>
            `;
            faqList.appendChild(item);
        });
    } else {
        faqSection.style.display = 'none';
    }

    // Show results card
    resultsCard.classList.remove('hidden');
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Download document
downloadBtn.addEventListener('click', () => {
    if (currentDocumentBase64) {
        // Convert base64 to blob
        const byteCharacters = atob(currentDocumentBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], {
            type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        });

        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = currentFileName;
        a.click();
        URL.revokeObjectURL(url);
    }
});

// Error handling
function showError(message) {
    errorCard.querySelector('.error-text').textContent = message;
    errorCard.classList.remove('hidden');
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
    errorCard.classList.add('hidden');
}

function hideResults() {
    resultsCard.classList.add('hidden');
}

function showSuccess(message) {
    // Create temporary success message
    const successDiv = document.createElement('div');
    successDiv.className = 'error-card';
    successDiv.style.background = '#ecfdf5';
    successDiv.style.borderColor = '#a7f3d0';
    successDiv.innerHTML = `
        <svg class="error-icon" style="color: #10b981" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span class="error-text" style="color: #065f46">${message}</span>
    `;

    const container = document.querySelector('.container');
    container.appendChild(successDiv);

    // Remove after 5 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}
