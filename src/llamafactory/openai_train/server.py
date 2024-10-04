from flask import Flask, request, jsonify, send_file
from typing import List, Dict, Any
import json
import subprocess
import os
from io import BytesIO
from openai.pagination import SyncPage
from openai.types import FileObject, FileDeleted
import threading
from .utils import load_api_keys

app = Flask(__name__)
def pydantic_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, (FileObject, SyncPage, FileDeleted)):
        return obj.model_dump()
    return obj

# Import file operations
from .file_ops import (
    create_file,
    retrieve_file,
    list_files,
    delete_file,
    get_file_content
)

# Import fine-tuning operations
from .oai_train import (
    create_fine_tuning_job,
    retrieve_fine_tuning_job,
    list_fine_tuning_jobs,
    cancel_fine_tuning_job,
    list_fine_tuning_job_events

)


# stand in for db
API_KEYS = load_api_keys()

def validate_api_key():
    auth_header = request.headers.get('Authorization')
    api_key = auth_header.split('Bearer ')[1]

    if api_key not in API_KEYS:
        return jsonify({"error": "Invalid or missing API key"}), 401
    return None

@app.before_request
def before_request():
    if request.endpoint != 'healthcheck':  # Skip API key check for healthcheck
        error_response = validate_api_key()
        if error_response:
            return error_response


@app.route('/fine_tuning/jobs', methods=['POST'])
def create_job():
    data = request.json
    print(data)
    try:
        result = jsonify(create_fine_tuning_job(**data))
        print(result)
        return result, 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/fine_tuning/jobs/<string:job_id>', methods=['GET'])
def retrieve_job(job_id: str):
    try:
        result = retrieve_fine_tuning_job(job_id)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

@app.route('/fine_tuning/jobs', methods=['GET'])
def list_jobs():
    after = request.args.get('after')
    limit = request.args.get('limit', default=20, type=int)
    result = list_fine_tuning_jobs(after=after, limit=limit)
    return jsonify(result)

@app.route('/fine_tuning/jobs/<string:job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str):
    try:
        result = cancel_fine_tuning_job(job_id)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/fine_tuning/jobs/<string:job_id>/events', methods=['GET'])
def list_job_events(job_id: str):
    after = request.args.get('after')
    limit = request.args.get('limit', default=20, type=int)
    result = list_fine_tuning_job_events(job_id, after=after, limit=limit)
    return jsonify(result)



@app.route('/files', methods=['POST'])
def create_file_route():
    if 'file' not in request.files:
        return jsonify({"error": {"message": "No file part"}}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": {"message": "No selected file"}}), 400
    purpose = request.form.get('purpose')
    if not purpose:
        return jsonify({"error": {"message": "Purpose is required"}}), 400
    
    try:
        result = create_file(file, purpose)
        return jsonify(pydantic_to_dict(result)), 201
    except ValueError as e:
        return jsonify({"error": {"message": str(e)}}), 400

@app.route('/files/<string:file_id>', methods=['GET'])
def retrieve_file_route(file_id: str):
    try:
        result = retrieve_file(file_id)
        return jsonify(pydantic_to_dict(result))
    except FileNotFoundError as e:
        return jsonify({"error": {"message": str(e),'type': 'invalid_request_error', 'param': 'id', 'code': None}}), 404

@app.route('/files', methods=['GET'])
def list_files_route():
    purpose = request.args.get('purpose')
    result = list_files(purpose=purpose)
    return jsonify(pydantic_to_dict(result))

@app.route('/files/<string:file_id>', methods=['DELETE'])
def delete_file_route(file_id: str):
    try:
        result = delete_file(file_id)
        return jsonify(pydantic_to_dict(result))
    except FileNotFoundError as e:
        return jsonify({"error": {"message": str(e),'type': 'invalid_request_error', 'param': 'id', 'code': None}}), 404

@app.route('/files/<string:file_id>/content', methods=['GET'])
def get_file_content_route(file_id: str):
    try:
        content = get_file_content(file_id)
        return send_file(
            BytesIO(content),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=file_id
        )
    except FileNotFoundError as e:
        return jsonify({"error": {"message": str(e),'type': 'invalid_request_error', 'param': 'id', 'code': None}}), 404



app = Flask(__name__)

def run_oai_train() -> None:
    oai_train_dir = os.path.expanduser("~/.OAI_TRAIN")
    
    # Remove existing directory if it exists
    if os.path.exists(oai_train_dir):
        shutil.rmtree(oai_train_dir)
    
    # Create necessary directories
    os.makedirs(oai_train_dir, exist_ok=True)
    os.makedirs(os.path.join(oai_train_dir, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(oai_train_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(oai_train_dir, "output_dir"), exist_ok=True)
    
    # Prepare configuration data
    data = {
        "model_name_or_path": "meta-llama/Llama-3.2-1B",
        "num_train_epochs": 3.0,
        "learning_rate": "5.0e-6",
        "output_dir": os.path.join(oai_train_dir, "output_dir")
    }
    
    # Write configuration to file
    config_path = os.path.join(oai_train_dir, "configs", "master.json")
    with open(config_path, "w") as f:
        json.dump(data, f)
    
    # Set up and run the API
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    print(f"API is running on http://{api_host}:{api_port}")
    app.run(host=api_host, port=api_port, debug=False)