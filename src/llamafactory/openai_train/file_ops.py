# file_ops/file_operations.py

import os
import uuid
import json
from openai.pagination import SyncPage
from openai.types import FileObject
from typing import List, Dict, Any
from werkzeug.datastructures import FileStorage


ROOT = os.path.expanduser(os.environ.get('ROOT_OAI_TRAIN', '~/.OAI_TRAIN'))
UPLOAD_FOLDER = os.path.expanduser(os.environ.get('UPLOAD_FOLDER', '~/.OAI_TRAIN/uploads'))
MASTER_CONFIG_FILE = os.path.expanduser(os.environ.get('MASTER_CONFIG', '~/.OAI_TRAIN/configs/master_config.json'))

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def create_file(file: FileStorage, purpose: str="fine-tune") -> Dict[str, Any]:
    if purpose in ["fine-tune", "batch"]:
        if not file.filename.lower().endswith('.jsonl'):
            raise ValueError(f"For {purpose} purpose, only .jsonl files are supported.")
    
    if purpose == "batch":
        file_size = file.content_length
        if file_size > 100 * 1024 * 1024:  # 100 MB
            raise ValueError("For batch purpose, file size must not exceed 100 MB.")

    name = str(uuid.uuid4()) # Generate a unique filename
    filename= name+".jsonl"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    file_info = {
        "id": name,
        "purpose": purpose,
        "filename": file.filename,
        "bytes": os.path.getsize(file_path),
        "created_at": int(os.path.getctime(file_path)),
        "status": "uploaded",
        'object': 'file'  # Add this line
    }
    
    # Save metadata
    with open(f"{os.path.join(UPLOAD_FOLDER, name)}.json", 'w') as f:
        json.dump(file_info, f)
    
    return file_info
    
    
def list_files(purpose: str = None) -> SyncPage[FileObject]:
    files: List[FileObject] = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(UPLOAD_FOLDER, filename), 'r') as f:
                    file_info = json.load(f)
                    if purpose is None or file_info['purpose'] == purpose:
                        print(file_info)
                        file_object = FileObject(**file_info)
                        files.append(file_object)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

    return SyncPage(data=files, object="list")


def retrieve_file(file_id: str) -> Dict[str, Any]:
    metadata_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"'No such File object: {file_id}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)
    
def delete_file(file_id: str) -> Dict[str, Any]:
    file_path = os.path.join(UPLOAD_FOLDER, file_id)
    metadata_path = f"{file_path}.json"
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"'No such File object: {file_id}")
    
    with open(metadata_path, 'r') as f:
        file_info = json.load(f)
    
    os.remove(file_path)
    os.remove(metadata_path)
    tmp = {}
    tmp['id'] = file_id
    tmp['deleted']=True
    tmp['object']='file'
    return tmp

def get_file_content(file_id: str) -> bytes:
    file_path = os.path.join(UPLOAD_FOLDER, file_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'No such File object: {file_id}")
    
    with open(file_path, 'rb') as f:
        return f.read()