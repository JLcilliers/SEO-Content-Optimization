// API Base URL - uses relative path for same-origin requests
const API_BASE = '';

// DOM Elements
const urlForm = document.getElementById('url-form');
const fileForm = document.getElementById('file-form');
const tabs = document.querySelectorAll('.tab');
const resultsCard = document.getElementById('results');
const errorCard = document.getElementById('error-message');
const downloadBtn = document.getElementById('download-btn');

// File upload elements
const contentUpload = document.getElementById('content-upload');
const contentFileInput = document.getElementById('content-file');
const keywordsUpload = document.getElementById('keywords-upload');
const keywordsFileInput = document.getElementById('keywords-file');

// State
let currentDocumentBase64 = null;
let currentFileName = 'optimized-content.docx';

// Tab switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;

        // Update tab states
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update form visibility
        document.querySelectorAll('.form').forEach(form => {
            form.classList.remove('active');
        });
        document.getElementById(`${targetTab}-form`).classList.add('active');

        // Hide results and errors when switching tabs
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

// URL Form submission
urlForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('url-submit');
    const sourceUrl = document.getElementById('source-url').value;
    const keywordsText = document.getElementById('keywords-text').value;
    const faqCount = parseInt(document.getElementById('faq-count').value);
    const maxSecondary = parseInt(document.getElementById('max-secondary').value);

    // Parse keywords
    const keywords = parseKeywords(keywordsText);
    if (keywords.length === 0) {
        showError('Please enter at least one keyword');
        return;
    }

    // Show loading state
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    hideError();
    hideResults();

    try {
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

    } catch (error) {
        showError(error.message);
    } finally {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

// File Form submission
fileForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('file-submit');
    const contentFile = contentFileInput.files[0];
    const keywordsFile = keywordsFileInput.files[0];
    const faqCount = parseInt(document.getElementById('file-faq-count').value);
    const maxSecondary = parseInt(document.getElementById('file-max-secondary').value);

    if (!contentFile) {
        showError('Please select a content document');
        return;
    }
    if (!keywordsFile) {
        showError('Please select a keywords file');
        return;
    }

    // Show loading state
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    hideError();
    hideResults();

    try {
        const formData = new FormData();
        formData.append('file', contentFile);
        formData.append('keywords_file', keywordsFile);
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

    } catch (error) {
        showError(error.message);
    } finally {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

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
