// Handle reference file selection
document.getElementById('reference').addEventListener('change', function () {
    const selectedFile = this.files[0];
    if (selectedFile) {
        const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
        if (fileExtension !== 'json') {
            alert('Please select a JSON file.');
            this.value = null;
        }
    }
});

// Handle order file selection
document.getElementById('orderfile').addEventListener('change', function () {
    const selectedFile = this.files[0];
    if (selectedFile) {
        const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
        if (fileExtension !== 'xlsx') {
            alert('Please select an Excel file.');
            this.value = null;
        }
    }
});

// Handle artwork file selection
document.getElementById('art').addEventListener('change', function () {
    const selectedFile = this.files[0];
    if (selectedFile) {
        const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
        if (fileExtension !== 'pdf') {
            alert('Please select a PDF file.');
            this.value = null;
        }
    }
});

// Handle the form submission
document.getElementById('artbtn').addEventListener('click', async function () {
    const referenceFile = document.getElementById('reference').files[0];
    const orderFile = document.getElementById('orderfile').files[0];
    const artFile = document.getElementById('art').files[0];

    if (!referenceFile || !orderFile || !artFile) {
        alert('Please select files for all three input fields.');
        return;
    }

    const formData = new FormData();
    formData.append('reference', referenceFile);
    formData.append('orderfile', orderFile);
    formData.append('art', artFile);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Files uploaded successfully:', result);
        } else {
            console.error('File upload failed.');
        }
    } catch (error) {
        console.error('An error occurred during file upload:', error);
    }
});
