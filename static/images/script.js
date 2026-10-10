document.addEventListener('DOMContentLoaded', function () {
    const referenceInput = document.getElementById('reference');
    const orderInput = document.getElementById('orderfile');
    const artInput = document.getElementById('art');
    const uploadButton = document.getElementById('uploadButton');

    let referenceFile = null;
    let orderFile = null;
    let artFile = null;

    function updateButtonState() {
        if (referenceFile && orderFile && artFile) {
            uploadButton.disabled = false;
        } else {
            uploadButton.disabled = true;
        }
    }

    function checkFileExtension(file, allowedExtensions) {
        return allowedExtensions.includes(file.type);
    }

    function checkFilesAndSubmit() {
        if (referenceFile && orderFile && artFile) {
            const formData = new FormData();
            formData.append('reference', referenceFile);
            formData.append('orderfile', orderFile);
            formData.append('art', artFile);
            formData.append('apiKey', 'YOUR_API_KEY_HERE');  // Add API Key

            fetch('/upload', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Files uploaded successfully');
                } else {
                    console.error('Error uploading files:', data.error);
                    alert('Error uploading files: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
    }

    referenceInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file && checkFileExtension(file, ['application/json'])) {
            referenceFile = file;
            updateButtonState();
        } else {
            alert('Please select a JSON file');
            referenceInput.value = '';
        }
    });

    orderInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file && checkFileExtension(file, ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])) {
            orderFile = file;
            updateButtonState();
        } else {
            alert('Please select an Excel file');
            orderInput.value = '';
        }
    });

    artInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file && checkFileExtension(file, ['application/pdf'])) {
            artFile = file;
            updateButtonState();
        } else {
            alert('Please select a PDF file');
            artInput.value = '';
        }
    });

    uploadButton.addEventListener('click', function () {
        checkFilesAndSubmit();
    });
});