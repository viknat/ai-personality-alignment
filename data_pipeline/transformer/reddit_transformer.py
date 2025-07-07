import zstandard as zstd
import json
import io

# Config
# Currently using local file path, but can be easily changed to use data in google drive with "from google.colab import drive"
comment_file = r"C:\Users\Administrator\Desktop\data\RC_2019-04.zst" # input
save_comment_samples = r"C:\Users\Administrator\Desktop\data\comment_samples.json" # temp output
save_transformed_samples = r"C:\Users\Administrator\Desktop\data\transformed_comment_samples.json" # output

## helper functions
# extract num_samples rows of users' data (each user is a row)
def extract_samples(file_path, num_samples=20000):
    samples = []
    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_streamer = io.TextIOWrapper(stream_reader, encoding='utf-8')

        for i, line in enumerate(text_streamer):
            if i >= num_samples:
                break
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i}")

    return samples

# Safely get value of key from each user's data
def safe_get(data, key, default=None):
    value = data.get_data(key, default)
    return value if value is not None else default

# transform to the unified schema
def transform_comment(comment):
    return {
        "user_id": "usr_" + safe_get(comment, "author_fullname", "unknown_user"),
        "chunk_id": safe_get(comment, "permalink", "unknown_chunk_id"),
        "platform": "reddit",
        "content": [
            {
                "speaker": "SELF",
                "text": safe_get(comment, "body", "empty_body"),
            }
        ],
        "timestamp": safe_get(comment, "created_utc", "unknown_timestamp")
    }


def run():
    comment_samples = extract_samples(comment_file, 20000)
    # with open(save_comment_samples, 'w') as f:
    #     json.dump(comment_samples, f, indent=2)
    # with open(save_transformed_samples, 'r') as f:
    #     comment_samples = json.load(f)
    transformed_comment_samples = [transform_comment(comment) for comment in comment_samples]
    with open(save_transformed_samples, 'w') as f:
        json.dump(transformed_comment_samples, f, indent=2)

    print(f"Transformed {len(transformed_comment_samples)} comments successfully")


# def run():
#     print(f"test")