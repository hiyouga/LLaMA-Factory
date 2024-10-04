import json


def load_api_keys():
    #dummy api keys
    return { "sk-6TWftpgBjwbF3nCnHIeYT3klbkFJhG1jHR0LFZ4RYLzPRBz1": "user1","key2": "user2"}



def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    with open(jsonl_file_path, 'r') as jsonl_file:
        json_lines = [line.strip() for line in jsonl_file if line.strip()]

    json_objects = [json.loads(line) for line in json_lines]
    print(json_file_path)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_objects, json_file, indent=4)