import json
from datasets import load_dataset
import datetime

def serialize(line):
    ts = line.get("timestamp")
    if isinstance(ts, (datetime.datetime, datetime.date)):
        line["timestamp"] = ts.isoformat()
    return line


# extract num_samples rows of users' data (each user is a row)
def extract_samples(file_path, num_samples_we_want_each_time, prev_ptr, ds_start_ptr):
    samples = []
    if ds_start_ptr is None:
        ds = load_dataset(file_path, split="train", streaming=True)
    else:
        ds = ds_start_ptr

    iterator = iter(ds) if prev_ptr is None else prev_ptr
    i = 0
    for _, line in enumerate(iterator):
        line = serialize(line)
        if i >= num_samples_we_want_each_time:
            break
        if line:
            i = i + 1
            try:
                samples.append(line)
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {i}")
    return samples, iterator, ds



def run(num_samples_we_want_each_time, prev_ptr, ds_start_ptr):
    transformed_comment_samples, curr_ptr, ds_start_ptr = extract_samples("BoqiaoZ/reddit_20M_JSON", num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
    return transformed_comment_samples, curr_ptr, ds_start_ptr

