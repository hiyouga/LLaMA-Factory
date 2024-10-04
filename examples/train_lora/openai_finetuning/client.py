import os
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:8000',
                api_key='sk-6TWftpgBjwbF3nCnHIeYT3klbkFJhG1jHR0LFZ4RYLzPRBz1'
                )



file = client.files.create(
  file=open("training_data.jsonl", "rb"),
  purpose="fine-tune"
)

res =client.fine_tuning.jobs.create(
  training_file=file.id, 
  model="microsoft/Phi-3.5-mini-instruct",
)

progress = client.fine_tuning.jobs.list_events(fine_tuning_job_id=res.id, limit=10)
print(progress)

