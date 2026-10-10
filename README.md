# AI Clothing Inspection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)

## Overview

The AI Clothing Inspection System is an advanced automated quality control solution for garment and textile inspection. It leverages cutting-edge artificial intelligence technologies including multilingual OCR, deep learning vision models, and real-time object detection to ensure comprehensive product quality assurance. The system automates the detection of defects, verifies compliance with order specifications, and generates detailed inspection reports.

## Features

- **Multilingual OCR**: PaddleOCR-based text extraction supporting multiple languages and character sets
- **AI Defect Detection**: Vision Transformer-powered image classification for garment analysis
- **Symbol Recognition**: YOLO-based detection of care symbols and regulatory marks
- **Real-time Progress**: WebSocket-powered live updates during inspection processing
- **History Tracking**: Complete inspection history with filterable results
- **Export Functionality**: Generate and download detailed inspection reports
- **Multi-user Support**: Role-based access with secure authentication
- **Configuration Management**: Flexible settings for production and development environments

## AI Process Explanation

### Step 1: Document Upload
Users upload two files:
- **Order Form**: Excel file containing product specifications and requirements
- **Product Artwork**: PDF or image file of the garment to be inspected

### Step 2: OCR Text Extraction
- PaddleOCR processes the product artwork to extract all text
- Supports multilingual text recognition
- Handles various text orientations and sizes
- Confidence scores provided for each extracted element

### Step 3: AI Analysis
- Vision Transformer model analyzes garment structure
- Classifies different garment parts (sleeves, collar, hem, etc.)
- Detects material composition and fabric types
- Generates feature vectors for comparison

### Step 4: Symbol Recognition
- YOLO model detects care symbols, washing instructions, and certifications
- Identifies regulatory marks and country-of-origin labels
- Maps symbols to standardized care instruction codes

### Step 5: Cross-Validation
- Compares extracted data against order form specifications
- Identifies discrepancies in materials, colors, sizes, and markings
- Flags non-compliance issues
- Generates defect list with severity levels

### Step 6: Report Generation
- Creates comprehensive inspection report
- Provides pass/fail verdict based on tolerance settings
- Includes detailed defect analysis
- Offers export in multiple formats

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- SQLite3
- 4GB RAM minimum

### Setup Instructions

1. **Clone the Repository**
```bash
git clone <repository-url>
cd InspectionAI_Avery_Dennison-AD_frontend
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Settings**
- Edit `setting/reference.json` with your configuration
- Update database connection strings if needed
- Configure API endpoints for backend services

5. **Initialize Database**
```bash
python
>>> from app import app, db
>>> with app.app_context():
...     db.create_all()
>>> exit()
```

6. **Run the Application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Starting the Application
```bash
python app.py
```

### Accessing the Web Interface
1. Open browser and navigate to `http://localhost:5000`
2. Login with your credentials (default setup required)
3. Navigate to Home to upload documents
4. View inspection results in real-time
5. Access History to review past inspections

### API Endpoints

#### Upload Files
```
POST /upload_files
Content-Type: multipart/form-data
- orderfile: Excel file
- art: PDF/Image file
```

#### Start Inspection
```
GET /start_inspection
```

#### Get Inspection Results
```
POST /api/inspection/finalresults
Authorization: Bearer <token>
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OCR** | PaddleOCR | Multilingual text recognition |
| **Vision** | Vision Transformer (ViT) | Image classification and analysis |
| **Detection** | YOLO | Object detection for symbols |
| **Backend** | Flask | Web application framework |
| **Database** | SQLite | Data persistence |
| **Real-time** | WebSocket | Live updates and streaming |
| **Auth** | JWT | Token-based authentication |
| **ORM** | SQLAlchemy | Database abstraction |

## Project Structure

```
InspectionAI_Avery_Dennison-AD_frontend/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── setting/              # Configuration files
│   └── reference.json    # System settings
├── uploads/              # Temporary file storage
├── static/               # Static assets
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── images/           # Image assets
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   ├── dashboard.html    # Server dashboard
│   ├── settings.html     # User settings
│   ├── history.html      # Inspection history
│   ├── about.html        # About page
│   ├── login.html        # Login page
│   └── signup.html       # Registration page
└── README.md             # This file
```

## Configuration

### Configuration File (setting/reference.json)

Key configuration parameters:

```json
{
  "production_mode": false,
  "ppta": true,
  "searchmode": true,
  "size_base_sheet": false,
  "us_size_translation": false,
  "adidas_size_translation": false,
  "body_measurements_languages": []
}
```

### Environment Variables

- `FLASK_ENV`: Environment mode (development/production)
- `FLASK_DEBUG`: Debug mode flag
- `SECRET_KEY`: Application secret key
- `DATABASE_URL`: Database connection string

## Troubleshooting

### Issue: Database Connection Error
**Solution**: Ensure SQLite3 is installed and the database file has write permissions.

### Issue: OCR Text Not Extracted
**Solution**: Check image quality and ensure PaddleOCR is properly installed. Verify PDF conversion if using PDF files.

### Issue: Port Already in Use
**Solution**: Change Flask port with:
```bash
python app.py --port 5001
```

### Issue: WebSocket Connection Failed
**Solution**: Ensure Flask-SocketIO is installed and the server is running with SocketIO enabled.

### Issue: YOLO Model Loading Error
**Solution**: Verify model files are in the correct directory and have proper read permissions.

## Performance Optimization

- Batch process multiple files for improved throughput
- Adjust image preprocessing resolution based on requirements
- Configure database query limits for history retrieval
- Enable caching for frequently accessed settings

## Security Considerations

- All authentication uses JWT tokens with expiration
- Database passwords stored in environment variables
- Input validation on all file uploads
- CORS properly configured for cross-origin requests
- SQL injection prevention through ORM usage



## Support & Contact

For issues, questions, or suggestions, please contact the development team through the project repository.

---

**Last Updated**: 2024
