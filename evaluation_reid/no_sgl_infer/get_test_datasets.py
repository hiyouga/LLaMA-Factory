import json
import os
from collections import defaultdict

# Query ⊂ Gallery（但评估时排除同图）

def load_pstp_test(caption_json):
    with open(caption_json, "r") as f:
        data = json.load(f)

    test_data = [x for x in data if x["split"] == "test"]

    print("=" * 60)
    print(f"[Dataset] Total samples : {len(data)}")
    print(f"[Dataset] Test samples  : {len(test_data)}")
    print("=" * 60)

    return test_data


