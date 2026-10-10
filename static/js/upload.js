let filesUploaded = {
    orderfile: false,
    art: false
};

function checkFiles() {
    const fileInput2 = document.getElementById('orderfile');
    const fileInput3 = document.getElementById('art');

    filesUploaded.orderfile = fileInput2.files.length > 0;
    filesUploaded.art = fileInput3.files.length > 0;

    // Display the file info for the order file
    displayFileInfo();
}

async function uploadFiles() {
    const fileInput2 = document.getElementById('orderfile');
    const fileInput3 = document.getElementById('art');

    const formData = new FormData();
    if (fileInput2.files.length > 0) formData.append('orderfile', fileInput2.files[0]);
    if (fileInput3.files.length > 0) formData.append('art', fileInput3.files[0]);

    try {
        const response = await fetch('/upload_files', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (result.r) {
            alert('Files Uploaded Successfully!');
            return true;
        } else {
            alert('Error: ' + result.error);
            return false;
        }
    } catch (error) {
        alert('Error uploading files: ' + error.message);
        return false;
    }
}

async function startInspection() {
    checkFiles();
    if (filesUploaded.orderfile && filesUploaded.art) {
        const uploadSuccess = await uploadFiles();
        if (uploadSuccess) {
            // Show the loader
            document.getElementById('loader').style.display = 'flex';

            await fetchPreprocessedData(); // Fetch preprocessed data after successful upload
            setTimeout(async () => {
                await fetchFinalResults(); // Fetch final results after a delay
                // Hide the loader
                document.getElementById('loader').style.display = 'none';
            }, 3000); // Adjust the delay as needed
        }
    } else {
        alert('Please upload all required files before starting inspection.');
    }
}

function refreshPage() {
    window.location.reload();
}

function displayFileInfo() {
    const fileInput2 = document.getElementById('orderfile');
    
    // Get the order file name
    const orderFileName = fileInput2.files.length > 0 ? fileInput2.files[0].name : 'No file chosen';

    // Update the UI with the file information
    document.getElementById('file-info').textContent = ` ${orderFileName}`;
}

// Add event listener to update file info when the order file is selected
document.getElementById('orderfile').addEventListener('change', checkFiles);
