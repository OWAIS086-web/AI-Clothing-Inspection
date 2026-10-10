from flask import Flask, request, jsonify, render_template
import threading
import queue
import os
import logging
import psutil
import flask
import platform
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from utils.config import dev_cfg, gen_cfg
if dev_cfg['dev_mode']:
    from inspection_dev import * 
else:
    from inspection import *
from utils.insp_logger import user_logger, dev_logger

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret_key')
app.config['SESSION_COOKIE_SECURE'] = True  # Secure session cookies
app.config['REMEMBER_COOKIE_SECURE'] = True  # Secure remember-me cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True  # HttpOnly session cookies
app.config['REMEMBER_COOKIE_HTTPONLY'] = True  # HttpOnly remember-me cookies
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Set SameSite attribute for session cookies

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

temporary_email_storage = None

# Health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "running"}), 200

# Resource check route
@app.route('/resource_check')
def resource_check():
    checks = {
        "filesystem": check_filesystem(),
        "cpu_usage": check_cpu_usage(),
        "memory_usage": check_memory_usage(),
        "disk_usage": check_disk_usage(),
        "network_usage": check_network_usage()
    }
    return jsonify(checks), 200

# Security check route
@app.route('/security_check')
def security_check():
    checks = {
        "https_enabled": check_https(),
        "http_secured": check_http_secured(),
        "secure_headers": check_secure_headers(),
        "server_info": check_server_info()
    }
    return jsonify(checks), 200

# Check file functions
def check_filesystem():
    try:
        root_directory = os.path.abspath(os.sep)
        is_writable = os.access(root_directory, os.W_OK)
        return {
            "status": "ok" if is_writable else "read-only",
            "details": f"Root directory is {'writable' if is_writable else 'not writable'}."
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

def check_cpu_usage():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        return {
            "status": "ok",
            "details": f"Current CPU usage is {cpu_usage}%."
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

def check_memory_usage():
    try:
        memory_info = psutil.virtual_memory()
        return {
            "status": "ok",
            "details": f"Memory usage is {memory_info.percent}% with {memory_info.used / (1024 * 1024):.2f} MB used."
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

def check_disk_usage():
    try:
        disk_usage = psutil.disk_usage('/')
        return {
            "status": "ok",
            "details": f"Disk usage is {disk_usage.percent}% with {disk_usage.used / (1024 * 1024 * 1024):.2f} GB used."
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

def check_network_usage():
    try:
        net_io = psutil.net_io_counters()
        return {
            "status": "ok",
            "details": f"Network received: {net_io.bytes_recv / (1024 * 1024):.2f} MB, Transmitted: {net_io.bytes_sent / (1024 * 1024):.2f} MB."
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

def check_https():
    return {
        "status": "ok" if request.is_secure else "not enabled",
        "details": "HTTPS is enabled." if request.is_secure else "HTTPS is not enabled."
    }

def check_http_secured():
    return {
        "status": "ok" if request.scheme == 'https' else "not secured",
        "details": "HTTP is secured." if request.scheme == 'https' else "HTTP is not secured."
    }

def check_secure_headers():
    headers = [
        'Content-Security-Policy', 
        'X-Content-Type-Options', 
        'X-Frame-Options', 
        'X-XSS-Protection'
    ]
    missing_headers = [header for header in headers if header not in request.headers]
    return {
        "status": "ok" if not missing_headers else "missing",
        "details": "Headers are correctly set." if not missing_headers else f"Missing headers: {', '.join(missing_headers)}."
    }

def check_server_info():
    try:
        python_version = platform.python_version()
        system_info = platform.system()
        machine_info = platform.machine()
        processor_info = platform.processor()
        server_info = {
            "server_name": request.host,
            "server_software": request.headers.get('Server', 'N/A'),
            "flask_version": flask.__version__,
            "python_version": python_version,
            "system_info": system_info,
            "machine_info": machine_info,
            "processor_info": processor_info
        }
        return server_info
    except Exception as e:
        return {"status": "error", "details": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

# Error handlers and security headers
@app.errorhandler(404)
def page_not_found(error):
    logging.error('Page Not Found: %s', error)
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    logging.error('Server Error: %s', error)
    return render_template('500.html'), 500

def run_flask_app():
    context = (f"{gen_cfg['ssl_path']}.crt", f"{gen_cfg['ssl_path']}.key")
    app.run(host=gen_cfg['domain'], port=gen_cfg['port'])

@app.route('/api/avery/input/', methods=['POST'])
@limiter.limit("5 per minute")
def handle_input():
    global temporary_token
    data = request.get_json()
    token = data.get('token', None)
    
    if token:
        temporary_token = token
    
    if data:
        response = {
            'status': 'success',
            'data_received': data,
            'temporary_token': temporary_token
        }
        with open("temp/input.json", "w") as json_file:
            json.dump(response, json_file)
        input_queue.put(response)
    else:
        response = {
            'status': 'failure',
            'data_received': {},
            'temporary_token': temporary_token
        }
   
    return jsonify(response), 200

input_queue = queue.Queue()
output_queue = queue.Queue()

flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

while True:
    try:
        print("Waiting for request...")
        received = input_queue.get(timeout=1)
        if received['status'] == 'success':
            dev_logger.info(f"INFO: Input Response: SUCCESS - 200")
            data_received = received['data_received']
            if 'item_name' not in data_received:
                data_received['item_name'] = 'default_item_name'
            try:
                inspection_dict = run_pipeline(data_received, output_queue=output_queue)
                headers = {
                    "X-Api-Key": gen_cfg['api_key'],
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {temporary_token}"  # Correctly set the Authorization header
                }
                with open("temp/output.json", "w") as json_file:
                    json.dump(inspection_dict, json_file)
                
                response = requests.post(
                    f"{gen_cfg['protocol']}://{gen_cfg['frontend_domain']}:{gen_cfg['frontend_port']}/api/inspection/finalresults",
                    json=inspection_dict,
                    headers=headers,
                    verify=True, #Verification for SSL certificate,
                    timeout=30,
                )
                
                dev_logger.info(f"INFO: Output Response Status Code: {response.status_code}")
                dev_logger.info(f"INFO: Output Response Content: {response.text}")
                
                if response.status_code == 200:
                    dev_logger.info(f"INFO: Output Response: SUCCESS - {response.status_code}")
                    dev_logger.info(f"INFO: Output Response: SUCCESS for token - {temporary_token}")
                    user_logger.info(f"INFO: Output Response: SUCCESS for token - {temporary_token}")
                else:
                    user_logger.error(f"INFO: Something went wrong while processing the outputs. Status Code: {response.status_code}, Response: {response.text}")
                    dev_logger.error(f"INFO: Output Response: FAILED - {response.status_code}")
            except KeyError as e:
                user_logger.error(f"Missing key in input data: {e}")
            except requests.exceptions.RequestException as e:
                user_logger.error(f"Error during requests to the frontend API: {e}")
        else:
            user_logger.info(f"INFO: Input Response: FAILED - {received['status']}")
    except queue.Empty:
        continue
