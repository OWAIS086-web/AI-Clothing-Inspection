// Function to fetch and display PDF
async function fetchAndDisplayPDF(pdfId) {
    try {
        const response = await fetch(`/pdf/${pdfId}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        // Create an iframe to display the PDF
        const iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.width = '100%';
        iframe.height = '500px';

        document.getElementById('pdfContainer').appendChild(iframe);
    } catch (error) {
        console.error('Failed to fetch and display PDF:', error);
    }
}

// Fetch and display a PDF when the document is ready
document.addEventListener('DOMContentLoaded', function () {
    const pdfId = 'your-pdf-id'; // Replace with your actual PDF ID or logic to get it
    fetchAndDisplayPDF(pdfId);
});
