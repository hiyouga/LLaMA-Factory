from datasets import load_dataset
import re, collections, functools
import numpy as np

# preprocess datasets to be loaded by peft-factory

def preprocess_wsc():
    _id2label = {0: "False", 1: "True", -1: ""}


    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r"^((?:\S+\s){N})(W)"
        pattern = re.sub("N", str(span_idx), pattern_tmpl)
        pattern = re.sub("W", span_str, pattern)
        return re.sub(pattern, r"\1{0} \2 {0}".format(mark), text)
    
    
    def preprocessor(example):
        # converts text as done in T5.
        text = example["text"]
        text = _mark_span(text, example["span1_text"], example["span1_index"], "*")
        # Compensate for 2 added "words" added in previous step.
        span2_index = example["span2_index"] + 2 * int(
            example["span1_index"] < example["span2_index"]
        )
        input_text = _mark_span(text, example["span2_text"], span2_index, "#")
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}

    wsc = load_dataset("super_glue", "wsc.fixed")

    return wsc.map(preprocessor)

def preprocess_wic():
    _id2label = {0: "False", 1: "True", -1: ""}

    def preprocessor(example):
        input_text = f"{example['sentence1']}\n\n{example['sentence2']}\n\n{example['word']}"
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}
    
    wic = load_dataset("super_glue", "wic")

    return wic.map(preprocessor)


def preprocess_multirc():
    _id2label = {0: "False", 1: "True", -1: ""}

    def preprocessor(example):
        input_text = f"{example['paragraph']}\n\n{example['question']}\n\n{example['answer']}"
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}
    
    multirc = load_dataset("super_glue", "multirc")

    return multirc.map(preprocessor)


def preprocess_copa():
    _id2label = {0: "choice1", 1: "choice2", -1: ""}

    def preprocessor(example):
        input_text = f"{example['premise']}\n\n{example['choice1']}\n\n{example['choice2']}"
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}
    
    copa = load_dataset("super_glue", "copa")

    return copa.map(preprocessor)  

def preprocess_record():
    def preprocessor(batch):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # print(ex["entities"])
            # updates the passage.
            passage = ex["passage"]
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"{ex['query']}\n\n{', '.join(ex['entities'])}\n\n{passage}"

            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["inputs"].extend([inputs] * num_duplicates)
            new_batch["targets"].extend(ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["idx"].extend([ex["idx"]] * num_duplicates)
            new_batch["answers"].extend([ex["answers"] if num_answers > 0 else ["<unk>"]]*num_duplicates)

        # print(new_batch)
        return new_batch
    
    record = load_dataset("super_glue", "record")
    return record.map(preprocessor, batched=True, remove_columns=record["train"].column_names)


# wsc = preprocess_wsc()
# wsc.push_to_hub("rbelanec/wsc")

# wic = preprocess_wic()
# wic.push_to_hub("rbelanec/wic")

# multirc = preprocess_multirc()
# multirc.push_to_hub("rbelanec/multirc")

# copa = preprocess_copa()
# copa.push_to_hub("rbelanec/copa")

record = preprocess_record()
record.push_to_hub("rbelanec/record")