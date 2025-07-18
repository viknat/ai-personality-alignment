import json
from collections import defaultdict

from datasets import load_dataset
import datetime


CHUNK_SIZE = 5  # max messages per user
PLATFORM = "reddit"

def serialize(line):
    ts = line.get("timestamp")
    if isinstance(ts, (datetime.datetime, datetime.date)):
        line["timestamp"] = ts.isoformat()
    return line


# extract num_samples rows of users' data (each user is a row)
def extract_samples(file_path, num_samples_we_want_each_time, prev_ptr, ds_start_ptr):
    user_message_buffer = defaultdict(list)
    user_metadata = {}
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
            # Append new comment data
            user_id = line.get("user_id")
            text = line.get("content")
            user_message_buffer[user_id].append({
                "speaker": "SELF",
                "text": text
            })
            # Update latest metadata
            user_metadata[user_id] = {
                "chunk_id": line.get("chunk_id"),
                "timestamp": line.get("timestamp")
            }

            # add full chunks
            if len(user_message_buffer[user_id]) >= CHUNK_SIZE:
                sample = {
                    "user_id": user_id,
                    "chunk_id": line.get("chunk_id"),
                    "platform": PLATFORM,
                    "content": user_message_buffer[user_id].copy(),
                    "timestamp": line.get("timestamp")
                }
                samples.append(sample)

                del user_message_buffer[user_id]
                del user_metadata[user_id]

    # handle partial chunks
    for user_id, messages in user_message_buffer.items():
        meta = user_metadata.get(user_id)
        if not meta or not messages:
            continue
        sample = {
            "user_id": user_id,
            "chunk_id": meta["chunk_id"],
            "platform": PLATFORM,
            "content": messages.copy(),
            "timestamp": meta["timestamp"]
        }
        samples.append(sample)
    return samples, iterator, ds



def run(num_samples_we_want_each_time, prev_ptr, ds_start_ptr):
    transformed_comment_samples, curr_ptr, ds_start_ptr = extract_samples("BoqiaoZ/reddit_2M_JSON", num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
    return transformed_comment_samples, curr_ptr, ds_start_ptr

