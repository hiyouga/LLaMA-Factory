import os
import json
import uuid
import time
import yaml
import threading
from typing import Dict, Any, List, Optional

# In-memory storage
job_queue = []
job_info_storage = {}
job_events_storage = {}
queue_lock = threading.Lock()
import os
from .utils import convert_jsonl_to_json
ROOT = os.path.expanduser(os.environ.get('ROOT_OAI_TRAIN', '~/.OAI_TRAIN'))
UPLOAD_FOLDER = os.path.expanduser(os.environ.get('UPLOAD_FOLDER', '~/.OAI_TRAIN/uploads'))


MASTER_CONFIG_FILE = os.path.expanduser(os.environ.get('MASTER_CONFIG', '~/.OAI_TRAIN/configs/master_config.json'))

from ..train.tuner import export_model, run_exp


def create_fine_tuning_job(
    model: str,
    training_file: str,
    hyperparameters: Dict[str, Any] = {},
    validation_file: Optional[str] = None,
    suffix: Optional[str] = None
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job_info = {
        "id": job_id,
        "model": model,
        "training_file": training_file,
        "validation_file": validation_file,
        "hyperparameters": hyperparameters,
        "suffix": suffix,
        "status": "queued",
        "created_at": int(time.time()),
    }
    
    # Store job info in memory
    job_info_storage[job_id] = job_info
    
    # Add job to queue and start processing
    with queue_lock:
        job_queue.append(job_id)
    
    # Start a new thread to process the job
    threading.Thread(target=process_job, args=(job_id,)).start()
    
    return job_info

def retrieve_fine_tuning_job(job_id: str) -> Dict[str, Any]:
    job_info = job_info_storage.get(job_id)
    if job_info is None:
        raise ValueError(f"Job with id {job_id} not found")
    return job_info

def list_fine_tuning_jobs(after: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    all_jobs = list(job_info_storage.values())
    jobs = []
    for job_info in all_jobs:
        if after is None or job_info['created_at'] > int(after):
            jobs.append(job_info)
    
    # Sort jobs by created_at in descending order
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit the number of jobs
    jobs = jobs[:limit]
    
    # Prepare the response
    response = {
        "object": "list",
        "data": jobs,
        "has_more": len(all_jobs) > len(jobs)
    }
    
    # Add pagination info if there are more jobs
    if response["has_more"] and jobs:
        response["after"] = jobs[-1]["id"]
    
    return response

def list_fine_tuning_job_events(job_id: str, after: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    all_events = job_events_storage.get(job_id, [])
    events = []
    
    for event_data in all_events:
        if after is None or event_data['id'] > after:
            events.append(event_data)
    
    # Sort events by created_at in descending order
    events.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit the number of events
    events = events[:limit]
    
    # Prepare the response
    response = {
        "object": "list",
        "data": events,
        "has_more": len(all_events) > len(events)
    }
    
    return response

def cancel_fine_tuning_job(job_id: str) -> Dict[str, Any]:
    job_info = retrieve_fine_tuning_job(job_id)
    if job_info['status'] in ['succeeded', 'failed', 'cancelled']:
        raise ValueError(f"Cannot cancel job with status {job_info['status']}")
    
    job_info['status'] = 'cancelled'
    job_info_storage[job_id] = job_info
    
    # Remove job from queue if it's still there
    with queue_lock:
        if job_id in job_queue:
            job_queue.remove(job_id)
    
    return job_info

# Helper function to add an event to a job
def add_job_event(job_id: str, message: str, level: str = "info"):
    event = {
        "object": "fine_tuning.job.event",
        "id": str(uuid.uuid4()),
        "created_at": int(time.time()),
        "level": level,
        "message": message,
    }
    if job_id not in job_events_storage:
        job_events_storage[job_id] = []
    job_events_storage[job_id].append(event)
    
    # Update job status if necessary
    if level == "error":
        job_info = retrieve_fine_tuning_job(job_id)
        job_info['status'] = 'failed'
        job_info_storage[job_id] = job_info





def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_job(job_id: str) -> None:
    job = retrieve_fine_tuning_job(job_id)
    print(f"\nProcessing job: {job_id}")
    print("=" * 50)
    
    # Print job information
    print(f"Model: {job.get('model', 'N/A')}")
    print(f"Status: {job.get('status', 'N/A')}")
    print(f"Created at: {job.get('created_at', 'N/A')}")
    print(f"Training file: {job.get('training_file', 'N/A')}")
 
    
    print("\nHyperparameters:")
    hyperparams = job.get('hyperparameters', {})
    for param, value in hyperparams.items():
        print(f"  {param}: {value}")
    
    print(f"\nOrganization ID: {job.get('organization_id', 'N/A')}")
    print(f"Seed: {job.get('seed', 'N/A')}")
    print(f"User provided suffix: {job.get('user_provided_suffix', 'N/A')}")
    
    print("=" * 50)
    print("\nStarting model training...")

    # Update job status to 'running'
    job['status'] = 'running'
    job_info_storage[job_id] = job

    
    config =json.load(open(os.path.expanduser("~/.OAI_TRAIN/configs/master.json"), "r"))

    config['model_name_or_path'] = job.get('model', 'N/A')

    config['output_dir'] = os.path.join(ROOT, "output_dir", str(job_id))
    config['num_train_epochs'] = float(hyperparams.get('n_epochs', config['num_train_epochs']))
    config['learning_rate'] = hyperparams.get('learning_rate_multiplier', config['learning_rate'])
    
    
    training_file =os.path.join(ROOT, "uploads", f"data_{job_id}.json")
    jsonl_file= os.path.join(UPLOAD_FOLDER, f"{job['training_file']}.jsonl")
    convert_jsonl_to_json(jsonl_file ,training_file )
    config['dataset'] = ['oai_finetune',training_file]
    config['template']='phi'
    print(config)
    run_exp(config)
    


    print(f"\nJob {job_id} completed successfully.")
 

    # except Exception as e:
    #     # Update job status to 'failed' and add error information
    #     job['status'] = 'failed'
    #     job['error'] = str(e)
    #     job_info_storage[job_id] = job

    #     print(f"\nJob {job_id} failed.")
    #     print(f"Error: {str(e)}")

    print("=" * 50)

    # Remove the job from the queue after processing
    with queue_lock:
        if job_id in job_queue:
            job_queue.remove(job_id)