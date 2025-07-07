from datasets import load_dataset
import random
import json

PLATFORM = "twitter"
DATASET_NAME = "enryu43/twitter100m_tweets"

def extract_data(num_samples_each_time, prev_ptr, ds_start_ptr):
    if ds_start_ptr is None:
        streamed_dataset = load_dataset(DATASET_NAME, split=f"train", streaming=True)
    else:
        streamed_dataset = ds_start_ptr
    samples = []
    conversation = []
    chunk_count = 0
    conv_len = random.randint(2, 5)  # Random conversation length between 2-5 utterances

    iterator = iter(streamed_dataset) if prev_ptr is None else prev_ptr

    for _, item in enumerate(iterator):
        # Extract fields with safe fallback
        user = item.get("user", "").strip()
        text = item.get("tweet", "").strip()
        timestamp = item.get("date", "").strip()

        # Skip incomplete or short entries
        if not user or not text or len(text) < 5 or not timestamp:
            continue

        # Assign random speaker role
        speaker = "SELF" if random.random() > 0.5 else "OTHER"
        conversation.append({"speaker": speaker, "text": text})

        # Once we reach the desired conversation length, commit the sample
        if len(conversation) >= conv_len:
            sample = {
                "user_id": user,
                "chunk_id": f"conv_{random.randint(1000, 9999)}#{chunk_count:02}",
                "platform": PLATFORM,
                "content": conversation.copy(),
                "timestamp": timestamp
            }

            samples.append(sample)
            conversation.clear()
            chunk_count += 1
            conv_len = random.randint(2, 5)  # Refresh next conversation length

        if len(samples) >= num_samples_each_time:
            break

    if not samples:
        raise ValueError("No samples were generated. Check filtering and tweet fields.")

    return samples, iterator, streamed_dataset


def run(prev_ptr, ds_start_ptr, num_samples_we_want_each_time):
    transformed_twitter_lines, curr_ptr, ds_start_ptr = extract_data(num_samples_we_want_each_time, prev_ptr, ds_start_ptr)
    return transformed_twitter_lines, curr_ptr, ds_start_ptr
