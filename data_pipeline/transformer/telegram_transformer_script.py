#@title Set up

"""
ETL pipeline to process the Pushshift Telegram dataset in Google Colab.

This script reads data from the three .ndjson files (Messages, Accounts, Channels),
transforms a sample of the messages into the ChatChunk format, and saves the
result as a single JSON file back to Google Drive.
"""
from datasets import load_dataset
import zstandard as zstd
import json
from datetime import datetime

# --- CONFIGURATION ---
# Adjust these variables to match your setup.

# 1. The number of messages to group into a single ChatChunk.
CHUNK_SIZE = 5

# 2. Total number of messages in the dataset (from the Zenodo record).
#    Used to calculate how many messages to process based on the percentage.
TOTAL_MESSAGES = 317000000

# --- END OF CONFIGURATION ---


#@title Utils

# ---------------------------
# function: read the zst file (file_path example: "BoqiaoZ/deep_alignment")
# ---------------------------
def read_zst_file_lines(
        file_path,
        num_samples_we_want_each_time,
        prev_ptr,
        ds_start_ptr):
    if ds_start_ptr is None:
        ds = load_dataset(file_path, split="train", streaming=True)
    else:
        ds = ds_start_ptr

    i = 0
    samples = []
    iterator = iter(ds) if prev_ptr is None else prev_ptr
    for _, sample in enumerate(iterator):
        if i >= num_samples_we_want_each_time:
            break
        if sample:
            i = i + 1
            samples.append(sample)
    return samples, iterator, ds


def read_whole_file_lines(file_path):
    ds = load_dataset(file_path, split="train", streaming=True)
    for line in ds:
        yield line


# -----------------------
# function: Custom JSON encoder
# -----------------------

def custom_serializer_post_process(obj):
    # Check if the object is of type datetime
    if isinstance(obj, datetime):
        # Convert datetime objects to ISO format string
        return obj.isoformat()
    # Add more custom handling for other non-serializable types here if needed
    raise TypeError(f"Type {type(obj)} not serializable")


def custom_serializer(obj):
    if  hasattr(datetime, 'datetime') and isinstance(obj, datetime.datetime):
        return obj.isoformat()  # or obj.timestamp() if you prefer a number
    return str(obj)  # fallback for other unknown types


def recursive_serialize(line):
    if isinstance(line, dict):
        # If data is a dictionary, recursively apply custom_serializer to each key-value pair
        return {key: recursive_serialize(value) for key, value in line.items()}
    elif isinstance(line, list):
        # If data is a list, recursively apply custom_serializer to each item
        return [recursive_serialize(item) for item in line]
    else:
        # Otherwise, apply custom_serializer to the current value
        return custom_serializer(line)

# -----------------------
# function: Preprocess each line to have consistent json schema
# -----------------------
def preprocess_line_for_json(line):
    """
    Preprocess the raw JSON line to ensure `restriction_reason` is consistent (always a string).
    """
    try:
        # Normalize `restriction_reason` field if present
        if isinstance(line, dict) and 'chats' in line:
            for chat in line['chats']:
                if 'restriction_reason' in chat:
                    # If it's an array, convert it to a string
                    if not isinstance(chat['restriction_reason'], str):
                        chat['restriction_reason'] = "restriction_reason is not a string"
        return recursive_serialize(line)
    except json.JSONDecodeError as e:
        print(f"Error decoding line: {line}. Error: {e}")
        return None


# --------------------------
# function: load lookup data
# --------------------------
def load_lookup_data(
        file_path: str,
        key_field: str
) -> dict:
    """
    Loads a .ndjson.zst file into a dictionary for fast lookups.
    This function is now robust enough to handle the complex 'ChatFull'
    object found in the channels file.
    """
    if not zstd: return {}
    lookup_dict = {}
    print(f"Loading lookup data from {file_path}...")
    try:
        for i, line in enumerate(read_whole_file_lines(file_path)):
            if (i + 1) % 500000 == 0:
                print(f"  ...processed {i+1:,} lines")
            try:
                data = preprocess_line_for_json(line)
                if data is None:
                    continue

                # Check if this is a complex ChatFull object from channels.ndjson
                if data.get('_') == 'ChatFull' and 'chats' in data:
                    for chat in data['chats']:
                        if key_field in chat and chat[key_field] is not None:
                            lookup_dict[chat[key_field]] = chat
                # Handle simple objects (like from accounts.ndjson)
                elif key_field in data and data[key_field] is not None:
                    lookup_dict[data[key_field]] = data
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"Finished loading {len(lookup_dict):,} items.\n")
        return lookup_dict
    except FileNotFoundError:
        print(f"ERROR: Lookup file not found at {file_path}. Please check the path.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return {}

# ------------------------------------------
# function: process and stream telegram data
# ------------------------------------------
def process_and_stream_telegram_data(
    messages_file: str,
    accounts_lookup: dict,
    channels_lookup: dict,
    num_samples_we_want_each_time,
    prev_ptr,
    ds_start_ptr,
    num_of_messages: int,
    chunk_size: int
):
    """
    Processes the messages file and streams the transformed ChatChunk objects
    directly to a file. To keep memory footprint low

    TRADE-OFF: Because we are not sorting, messages within a chunk are not
    guaranteed to be in chronological order. They will appear in the order
    they exist in the source file.
    """

    if not zstd: return 0
    num_messages_to_process = num_of_messages

    # --- Low-Memory Streaming Logic ---
    # Holds incomplete chunks for channels currently being processed.
    # This keeps memory usage low and constant.
    incomplete_chunks = {}
    # Keeps track of the chunk number for each channel.
    chunk_counters = {}

    total_chunks_written = 0
    processed_count = 0

    is_first_chunk = True

    chat_chunks = []

    lines, curr_ptr, ds_start_ptr = read_zst_file_lines(
        messages_file,
        num_samples_we_want_each_time,
        prev_ptr,
        ds_start_ptr)

    for i, line in enumerate(lines):
        if processed_count >= num_messages_to_process:
            print(f"Reached processing limit of {num_messages_to_process:,} messages.")
            break

        if (i + 1) % 2000000 == 0:
            print(f"  ...scanned {i+1:,} lines, processed {processed_count:,} valid messages.")

        try:
            msg = line

            if not msg.get('message'): continue

            channel_id = msg.get('to_id', {}).get('channel_id') or msg.get('peer_id', {}).get('channel_id')

            from_id_val = msg.get('from_id')
            author_id = from_id_val if isinstance(from_id_val, int) else (from_id_val.get('user_id') if isinstance(from_id_val, dict) else None)

            if not channel_id or not author_id: continue

            processed_count += 1

            # Initialize tracking for a new channel
            if channel_id not in incomplete_chunks:
                incomplete_chunks[channel_id] = []
                chunk_counters[channel_id] = 0

            # Add the current message to its channel's incomplete chunk
            incomplete_chunks[channel_id].append(msg)

            # If a chunk is full, process and write it
            if len(incomplete_chunks[channel_id]) == chunk_size:
                message_chunk = incomplete_chunks[channel_id]
                chunk_counters[channel_id] += 1

                # --- Format the chunk ---
                channel_info = channels_lookup.get(channel_id, {})
                content = []
                for m in message_chunk:
                    auth_id = m.get('from_id') if isinstance(m.get('from_id'), int) else m.get('from_id', {}).get('user_id')
                    author_info = accounts_lookup.get(auth_id, {})
                    first_name, last_name = author_info.get('first_name', ''), author_info.get('last_name', '')
                    speaker_name = f"{first_name} {last_name}".strip() or f"user_{auth_id}"
                    content.append({"speaker": speaker_name, "text": m.get('message', '')})

                chat_chunk = {
                    "user_id": f"channel_{channel_id}",
                    "chunk_id": f"conv_{channel_id}#{chunk_counters[channel_id]:04d}",
                    "platform": "telegram", "content": content,
                    "timestamp": message_chunk[0].get('date', datetime.now().isoformat()),
                    "metadata": {"channel_name": channel_info.get('title', f"channel_{channel_id}")}
                }

                chat_chunks.append(chat_chunk)

                # Clear the chunk now that it's written
                incomplete_chunks[channel_id] = []

        except (json.JSONDecodeError, KeyError):
            continue

    # --- After the loop, write any remaining incomplete chunks ---
    print("\nFinished reading messages. Writing remaining incomplete chunks...")
    for channel_id, message_chunk in incomplete_chunks.items():
        if not message_chunk: continue

        chunk_counters[channel_id] += 1
        channel_info = channels_lookup.get(channel_id, {})
        content = []
        for m in message_chunk:
            auth_id = m.get('from_id') if isinstance(m.get('from_id'), int) else m.get('from_id', {}).get('user_id')
            author_info = accounts_lookup.get(auth_id, {})
            first_name, last_name = author_info.get('first_name', ''), author_info.get('last_name', '')
            speaker_name = f"{first_name} {last_name}".strip() or f"user_{auth_id}"
            content.append({"speaker": speaker_name, "text": m.get('message', '')})

        chat_chunk = {
            "user_id": f"channel_{channel_id}",
            "chunk_id": f"conv_{channel_id}#{chunk_counters[channel_id]:04d}",
            "platform": "telegram", "content": content,
            "timestamp": message_chunk[0].get('date', datetime.now().isoformat()),
            "metadata": {"channel_name": channel_info.get('title', f"channel_{channel_id}")}
        }
        chat_chunks.append(chat_chunk)

    print(f"Successfully created chat_chunks: {len(chat_chunks)} lines")
    return chat_chunks, curr_ptr, ds_start_ptr


#@title Main

# -----------------------
# function: main function
# -----------------------
def run(num_samples_we_want_each_time,
        prev_ptr,
        ds_start_ptr):
    """Main function to run the ETL pipeline."""

    if not zstd:
        print("Please install the 'zstandard' library by running '!pip install zstandard' and try again.")
        return

    print("--- Starting Telegram ETL Pipeline ---")

    channels_lookup = load_lookup_data(
        "BoqiaoZ/channels",
        'id')
    accounts_lookup = load_lookup_data(
        "BoqiaoZ/accounts",
        'id')

    if not accounts_lookup or not channels_lookup:
        print("\nCould not load lookup files. Aborting pipeline.")
        return

    # The processing and writing happens inside this function now
    chat_chunks, curr_ptr, ds_start_ptr = process_and_stream_telegram_data(
        "BoqiaoZ/cleaned_messages_20M",
        accounts_lookup,
        channels_lookup,
        num_samples_we_want_each_time,
        prev_ptr,
        ds_start_ptr,
        num_of_messages=20,
        chunk_size=CHUNK_SIZE
    )

    print("--- Pipeline Finished ---")
    return chat_chunks, curr_ptr, ds_start_ptr
