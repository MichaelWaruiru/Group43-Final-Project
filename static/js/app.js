// DOM Elements
const fileInput = document.getElementById('file');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const uploadForm = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
function initializeApp() {
    console.log('Plant Disease Detection App initialized');
    
    // Setup file input listener
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Setup form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
    
    // Setup drag and drop
    setupDragAndDrop();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Add fade-in animation to elements
    addAnimations();
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (file) {
        // Validate file type
        if (!isValidImageFile(file)) {
            showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WEBP)', 'error');
            clearFileInput();
            return;
        }
        
        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showAlert('File size must be less than 16MB', 'error');
            clearFileInput();
            return;
        }
        
        // Show image preview
        showImagePreview(file);
        
        // Update submit button
        updateSubmitButton(true);
    } else {
        hideImagePreview();
        updateSubmitButton(false);
    }
}

// Validate image file type
function isValidImageFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    return allowedTypes.includes(file.type);
}

// Show image preview
function showImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        imagePreview.classList.add('fade-in');
    };
    
    reader.readAsDataURL(file);
}

// Hide image preview
function hideImagePreview() {
    imagePreview.style.display = 'none';
    imagePreview.classList.remove('fade-in');
    previewImg.src = '';
}

// Clear file input
function clearFileInput() {
    fileInput.value = '';
    hideImagePreview();
    updateSubmitButton(false);
}

// Update submit button state
function updateSubmitButton(hasFile) {
    if (hasFile) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Plant';
        submitBtn.classList.add('btn-success');
        submitBtn.classList.remove('btn-secondary');
    } else {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-image me-2"></i>Select Image First';
        submitBtn.classList.add('btn-secondary');
        submitBtn.classList.remove('btn-success');
    }
}

// Handle form submission
function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(uploadForm);
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select an image file', 'error');
        return;
    }
    
    // Show loading modal
    showLoadingModal();
    
    // Update submit button to loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    
    // Submit form
    uploadForm.submit();
}

// Show loading modal
function showLoadingModal() {
    loadingModal.show();
    
    // Hide modal after 10 seconds as fallback
    setTimeout(() => {
        loadingModal.hide();
    }, 10000);
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const dropZone = document.querySelector('.upload-section .card');
    
    if (!dropZone) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
}

// Prevent default drag behaviors
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone
function highlight(e) {
    e.currentTarget.classList.add('border-success', 'bg-success-subtle');
}

// Remove highlight from drop zone
function unhighlight(e) {
    e.currentTarget.classList.remove('border-success', 'bg-success-subtle');
}

// Handle file drop
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        
        // Set file to input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Trigger change event
        fileInput.dispatchEvent(new Event('change'));
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert alert at the top of the container
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add animations to elements
function addAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
    
    // Add slide-up animation to hero section
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        heroSection.classList.add('slide-up');
    }
}

// Toggle treatment recommendations
function toggleTreatments(index) {
    const treatmentsDiv = document.getElementById(`treatments-${index}`);
    const button = event.target;
    
    if (treatmentsDiv.style.display === 'none' || treatmentsDiv.style.display === '') {
        treatmentsDiv.style.display = 'block';
        treatmentsDiv.classList.add('fade-in');
        button.innerHTML = '<i class="fas fa-eye-slash me-1"></i>Hide Treatments';
        button.classList.remove('btn-outline-success');
        button.classList.add('btn-success');
    } else {
        treatmentsDiv.style.display = 'none';
        treatmentsDiv.classList.remove('fade-in');
        button.innerHTML = '<i class="fas fa-medical-kit me-1"></i>View Treatments';
        button.classList.remove('btn-success');
        button.classList.add('btn-outline-success');
    }
}

// Share result functionality
function shareResult(disease, confidence) {
    const text = `Plant Disease Detection Result: ${disease.replace('_', ' ')} with ${confidence}% confidence. Analyzed using AI-powered plant disease detection system.`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Plant Disease Detection Result',
            text: text,
            url: window.location.href
        }).then(() => {
            console.log('Result shared successfully');
        }).catch((error) => {
            console.log('Error sharing result:', error);
            fallbackShare(text);
        });
    } else {
        fallbackShare(text);
    }
}

// Fallback share function
function fallbackShare(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showAlert('Result copied to clipboard!', 'success');
        }).catch(() => {
            showAlert('Unable to copy result. Please copy manually.', 'error');
        });
    } else {
        // Create temporary textarea for copying
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        
        try {
            document.execCommand('copy');
            showAlert('Result copied to clipboard!', 'success');
        } catch (err) {
            showAlert('Unable to copy result. Please copy manually.', 'error');
        }
        
        document.body.removeChild(textarea);
    }
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden
        console.log('Page is hidden');
    } else {
        // Page is visible
        console.log('Page is visible');
    }
});

// Handle online/offline status
window.addEventListener('online', function() {
    showAlert('Internet connection restored', 'success');
});

window.addEventListener('offline', function() {
    showAlert('Internet connection lost. Some features may not work.', 'error');
});

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showAlert('An unexpected error occurred. Please try again.', 'error');
});

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        isValidImageFile,
        formatFileSize,
        showAlert
    };
}

// Theme Toggle
document.addEventListener('DOMContentLoaded', () => {
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const htmlEl = document.documentElement;

    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        htmlEl.setAttribute('data-bs-theme', savedTheme);
    }

    // Toggle theme on click
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = htmlEl.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            htmlEl.setAttribute('data-bs-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
});
