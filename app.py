import socket
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
import psutil
import datetime
import platform
import os
import base64
import json
from flask_socketio import SocketIO, emit
import requests
import logging
from werkzeug.utils import secure_filename

# Initialize Flask app and configurations
app = Flask(__name__)
app.secret_key = 'aspaul20'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visionrd.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SETTINGS'] = 'setting'
app.config['CERTIFICATE_PATH'] = os.path.join(app.config['UPLOAD_FOLDER'], 'server.crt')
app.config['JWT_SECRET_KEY'] = 'visionrd321'

# Initialize extensions
CORS(app)
socketio = SocketIO(app, async_mode="threading")
jwt = JWTManager(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create directories if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    orderfile_name = db.Column(db.String(200), nullable=True)

    user = db.relationship('User', backref=db.backref('files', lazy=True))

class InspectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    inspection_status = db.Column(db.Integer, nullable=False)
    defect_list = db.Column(db.Text, nullable=False)
    time_taken = db.Column(db.String(50), nullable=False)
    defect_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    user = db.relationship('User', backref=db.backref('inspection_results', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        for file in files:
            try:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except Exception as e:
                logging.error(f'File save error: {str(e)}')
                return jsonify({'r': False, 'error': f'File save error: {str(e)}'}), 500
        return redirect(url_for('index'))
    return render_template('index.html', username=current_user.username)

@app.route('/upload_files', methods=['POST'])
@login_required
def upload_files():
    orderfile = request.files.get('orderfile')
    art = request.files.get('art')

    if not (orderfile and art):
        return jsonify({'r': False, 'error': 'Missing files'}), 400

    try:
        orderfile_filename = secure_filename(orderfile.filename)
        art_filename = secure_filename(art.filename)

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        os.makedirs(user_folder, exist_ok=True)

        orderfile_path = os.path.join(user_folder, orderfile_filename)
        art_path = os.path.join(user_folder, art_filename)

        orderfile.save(orderfile_path)
        art.save(art_path)

        reference_base64 = convert_file_to_base64(os.path.join(app.config['SETTINGS'], 'reference.json'))
        orderfile_base64 = convert_file_to_base64(orderfile_path)
        art_base64 = convert_file_to_base64(art_path)
        json_content = extract_and_map_json_content(os.path.join(app.config['SETTINGS'], 'reference.json'))

        token = create_access_token(identity=current_user.id)

        model = {
            "excel": {"order_form": orderfile_base64},
            "pdf": {"pdf_file": art_base64},
            "item_name": orderfile_filename,
            "ref": json_content,
            "token": token
        }

        api_url = "http://127.0.0.1:10002/api/avery/input/"
        headers = {'Authorization': f'Bearer {token}'}

        response = requests.post(api_url, json=model, headers=headers)
        response.raise_for_status()

        new_file = File(filename=orderfile_filename, filepath=orderfile_path, user_id=current_user.id, orderfile_name=orderfile_filename)
        db.session.add(new_file)
        db.session.commit()

        return jsonify({'r': True, 'file': orderfile_filename})
    except requests.exceptions.RequestException as e:
        logging.error(f'POST request failed: {str(e)}')
        return jsonify({'r': False, 'error': f'POST request failed: {str(e)}'}), 500
    except Exception as e:
        logging.error(f'Error processing files: {str(e)}')
        return jsonify({'r': False, 'error': f'Error processing files: {str(e)}'}), 500



@app.route('/start_inspection', methods=['GET'])
def start_inspection():
    api_url = "http://127.0.0.1:10002/api/avery/input/"

    try:
        response = requests.get(api_url, cert=(app.config['CERTIFICATE_PATH'], app.config['CERTIFICATE_PATH']))
        response.raise_for_status()
        return jsonify({'r': True})
    except requests.exceptions.RequestException as e:
        logging.error(f'GET request failed: {str(e)}')
        return jsonify({'r': False, 'error': f'GET request failed: {str(e)}'}), 500

def convert_file_to_base64(file_path):
    try:
        with open(file_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding file to base64: {str(e)}")
        raise

def extract_and_map_json_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error reading JSON content: {str(e)}")
        raise




@app.route('/api/inspection/Preprocessed', methods=['POST'])
def handle_preprocessed_results():
    try:
        data = request.get_json()
        with open("Preprocessed.json", "w") as json_file:
            json.dump(data, json_file)
            socketio.emit('preprocessed_results', data)
        return jsonify({'r': True}), 200
    except Exception as e:
        logging.error(f'Error handling Preprocessed results: {str(e)}')
        return jsonify({'r': False, 'error': f'Error handling Preprocessed results: {str(e)}'}), 500

@app.route('/api/inspection/finalresults', methods=['POST'])
@jwt_required()
def handle_final_results():
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        inspection_status = data.get('inspection_status')
        defect_list = data.get('defect_list', [])
        time_taken = data.get('time_taken')
        logs = data.get('logs', '')

        defect_count = len(defect_list)

        new_result = InspectionResult(
            user_id=current_user_id,
            inspection_status=inspection_status,
            defect_list=json.dumps(defect_list),
            time_taken=time_taken,
            defect_count=defect_count
        )
        db.session.add(new_result)
        db.session.commit()

        with open("finalresults.json", "w") as json_file:
            json.dump(data, json_file)
            socketio.emit('finalresults', data)

        return jsonify({'r': True}), 200
    except Exception as e:
        logging.error(f'Error handling final results: {str(e)}')
        return jsonify({'r': False, 'error': f'Error handling final results: {str(e)}'}), 500




@app.route('/settings')
@login_required
def settings_page():
    return render_template('settings.html', username=current_user.username)

@app.route('/about')
@login_required
def about_page():
    return render_template('about.html', username=current_user.username)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.get_json()
    try:
        with open('setting/reference.json', 'w') as f:
            json.dump(data, f, indent=4)
        return jsonify({"message": "Settings saved successfully"}), 200
    except Exception as e:
        logging.error(f"Error saving settings: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route('/get_default_settings', methods=['GET'])
def get_default_settings():
    try:
        with open('setting/reference.json') as f:
            settings = json.load(f)
        return jsonify(settings), 200
    except Exception as e:
        logging.error(f"Error retrieving default settings: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route('/upload_settings', methods=['POST'])
def upload_settings():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file and file.filename.endswith('.json'):
        try:
            data = json.load(file)
            with open('setting/reference.json', 'w') as f:
                json.dump(data, f, indent=4)
            return jsonify({"message": "Import successful"}), 200
        except Exception as e:
            logging.error(f"Error uploading settings: {str(e)}")
            return jsonify({"message": "An error occurred", "error": str(e)}), 500
    return jsonify({"message": "Invalid file type"}), 400




@app.route('/dev_setting')
@login_required
def dev_setting_page():
    return render_template('dev_setting.html', username=current_user.username)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/server')
def get_metrics():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        storage = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
        load_avg = psutil.getloadavg()
        open_files = len(psutil.Process().open_files())
        connections = len(psutil.net_connections())

        metrics = {
            'cpu_usage': cpu_usage,
            'memory_total': memory.total,
            'memory_used': memory.used,
            'storage_total': storage.total,
            'storage_used': storage.used,
            'net_sent': net_io.bytes_sent,
            'net_recv': net_io.bytes_recv,
            'uptime': str(uptime),
            'load_avg': load_avg,
            'open_files': open_files,
            'connections': connections,
            'hostname': socket.gethostname(),
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version()
        }
        return jsonify(metrics)
    except Exception as e:
        logging.error(f"Error retrieving server metrics: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        access_token = create_access_token(identity=user.id)
        return render_template('index.html', success=True), 200
    else:
        flash('Wrong username or password!', 'error')
        return redirect(url_for('login'))

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Passwords do not match!', 'error')
        return redirect(url_for('signup'))

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists!', 'error')
        return redirect(url_for('signup'))

    password_hash = generate_password_hash(password)
    new_user = User(username=username, password_hash=password_hash)
    db.session.add(new_user)
    db.session.commit()
    flash('User created successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    inspection_results = InspectionResult.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', username=current_user.username, inspection_results=inspection_results)

@app.route('/api/inspections', methods=['GET'])
def get_inspections():
    inspections_query = InspectionResult.query
    inspections = inspections_query.all()
    
    inspections_data = []

    for inspection in inspections:
        try:
            defect_images = json.loads(inspection.defect_list)
            
        except Exception as e:
            logging.error(f"Error decoding defect list: {e}")
            defect_images = []

        defect_images_base64 = []
        for img_base64 in defect_images:
            try:
                defect_images_base64.append(img_base64)
            except Exception as e:
                logging.error(f"Error processing Base64 image: {e}")

        # Default values if no file is found
        filename = 'No filename available'
        filepath = 'No filepath available'

        # Attempt to find the related file in the File table using user_id
        try:
            if hasattr(inspection, 'user_id') and inspection.user_id:
                # Fetch the file related to the user_id
                file_info = File.query.filter_by(user_id=inspection.user_id).first()
                if file_info:
                    filename = file_info.filename
                    filepath = file_info.filepath
                else:
                    logging.error(f"No matching file found for user_id: {inspection.user_id}")
            else:
                logging.error(f"No user_id attribute or value for inspection ID: {inspection.id}")
        except Exception as e:
            logging.error(f"Error fetching file for inspection ID {inspection.id}: {e}")

        inspection_data = {
            'id': inspection.id,
            'articleNumber': filename,  
            'filepath': filepath,
            'created_at': inspection.created_at.isoformat(),
            'inspection_type': inspection.inspection_status,
            'time_taken': inspection.time_taken,
            'defect_count': inspection.defect_count,
            'defect_status': inspection.inspection_status,
            'defect_images': defect_images_base64
        }

        inspections_data.append(inspection_data)

    return jsonify({'inspections': inspections_data})



@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful.', 'success')
    return redirect(url_for('login'))

@app.errorhandler(404)
def page_not_found(error):
    username = current_user.username if current_user.is_authenticated else ''
    return render_template('404.html', username=username), 404

@app.errorhandler(500)
def internal_server_error(error):
    username = current_user.username if current_user.is_authenticated else ''
    return render_template('500.html', username=username), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)