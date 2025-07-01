from datasets import load_dataset, Dataset
import random


NUM_SAMPLES = 50_000                         # Number of transformed samples to generate
PLATFORM = "twitter"                         # Platform tag for the schema
DATASET_NAME = "enryu43/twitter100m_tweets"  # Hugging Face dataset to stream from
HUB_REPO_ID = "<HF_REPO_ID>/deep-alignment-twitter"  # Your HF repo ID (must exist or be creatable)


streamed_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)


samples = []
conversation = []
chunk_count = 0
conv_len = random.randint(2, 5)  # Random conversation length between 2-5 utterances

for item in streamed_dataset:
    # Extract fields with safe fallback
    user = item.get("user", "").strip()
    text = item.get("tweet", "").strip()
    timestamp = item.get("date", "").strip()

    # Skip incomplete or short entries
    if not user or not text or len(text) < 5 or not timestamp:
        continue

    # Assign random speaker role
    speaker = "SELF" if random.random() > 0.5 else "OTHER"
    conversation.append({ "speaker": speaker, "text": text })

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

    
    if len(samples) >= NUM_SAMPLES:
        break


if not samples:
    raise ValueError("No samples were generated. Check filtering and tweet fields.")


print(f" Successfully generated {len(samples)} samples. Converting to Hugging Face Dataset...")
hf_dataset = Dataset.from_list(samples)

print(f" Uploading to Hugging Face Hub at: https://huggingface.co/datasets/{HUB_REPO_ID}")
hf_dataset.push_to_hub(HUB_REPO_ID, private=True)
print("Upload complete! Dataset is now live on the Hub.")
