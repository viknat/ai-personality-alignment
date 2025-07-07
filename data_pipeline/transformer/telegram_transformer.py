#@title Set up

"""
ETL pipeline to process the Pushshift Telegram dataset in Google Colab.

This script reads data from the three .ndjson files (Messages, Accounts, Channels),
transforms a sample of the messages into the ChatChunk format, and saves the
result as a single JSON file back to Google Drive.
"""
import json
import io
import os
try:
    import zstandard as zstd
except ImportError:
    print("Zstandard library not found. Please run '!pip install zstandard' in a Colab cell.")
    zstd = None

from datetime import datetime
# from google.colab import drive
# drive.mount('/content/drive')
from pathlib import Path


# --- CONFIGURATION ---
# Adjust these variables to match your setup.

# 1. The path to the folder in your Google Drive where you uploaded the files.
#    IMPORTANT: Your error message shows this path. Please verify it is correct.
DRIVE_FOLDER_PATH = "drive/MyDrive/datasets/Telegram_Dataset"

# 2. The percentage of the total messages you want to process.
#    - For testing, start with a very small number (e.g., 0.1 or 1).
#    - To process the entire dataset, set this to 100.
PROCESSING_PERCENTAGE = 5.0  # Process 5% of the messages

# 3. The number of messages to group into a single ChatChunk.
CHUNK_SIZE = 5

# 4. Total number of messages in the dataset (from the Zenodo record).
#    Used to calculate how many messages to process based on the percentage.
TOTAL_MESSAGES = 317000000

# --- END OF CONFIGURATION ---


#@title Utils

# ---------------------------
# function: read the zst file
# ---------------------------
def read_zst_file_lines(file_path):
    """A generator function to read a .zst file and yield lines."""
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            # Wrap the binary stream in a text wrapper to read lines
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                yield line

# -----------------------
# function: inspect files
# -----------------------
def inspect_files(base_path: str, num_rows: int = 5):
    """
    Reads and prints the first few lines of each .ndjson.zst file to inspect their structure.
    """
    if not zstd: return
    print("--- Inspecting Dataset Files ---\n")
    # Using lowercase filenames with .zst extension
    files_to_inspect = ["accounts.ndjson.zst", "channels.ndjson.zst", "messages.ndjson.zst"]

    for file_name in files_to_inspect:
        file_path = Path(base_path) / file_name
        print(f"--- First {num_rows} rows of {file_name} ---")
        if not file_path.exists():
            print(f"ERROR: File not found at {file_path}")
            print("       Please check that the file exists and that DRIVE_FOLDER_PATH is correct.")
            continue

        try:
            for i, line in enumerate(read_zst_file_lines(file_path)):
                if i >= num_rows:
                    break
                try:
                    # Pretty print the JSON object
                    parsed_json = json.loads(line)
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError:
                    print(f"Could not parse line {i+1}: {line.strip()}")
            print("-" * 30 + "\n")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}\n")

# --------------------------
# function: load lookup data
# --------------------------
def load_lookup_data(file_path: str, key_field: str) -> dict:
    """
    Loads a .ndjson.zst file into a dictionary for fast lookups.
    This function is now robust enough to handle the complex 'ChatFull'
    object found in the channels file.
    """
    if not zstd: return {}
    lookup_dict = {}
    print(f"Loading lookup data from {Path(file_path).name}...")
    try:
        for i, line in enumerate(read_zst_file_lines(file_path)):
            if (i + 1) % 500000 == 0:
                print(f"  ...processed {i+1:,} lines")
            try:
                data = json.loads(line)
                # Check if this is a complex ChatFull object from channels.ndjson
                if data.get_data('_') == 'ChatFull' and 'chats' in data:
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
    base_path: str,
    accounts_lookup: dict,
    channels_lookup: dict,
    percentage: float,
    chunk_size: int,
    output_file_handle,
):
    """
    Processes the messages file and streams the transformed ChatChunk objects
    directly to a file. To keep memory footprint low

    TRADE-OFF: Because we are not sorting, messages within a chunk are not
    guaranteed to be in chronological order. They will appear in the order
    they exist in the source file.
    """

    if not zstd: return 0
    messages_file = Path(base_path) / "messages.ndjson.zst"
    if not messages_file.exists():
        print(f"ERROR: Messages file not found at {messages_file}")
        return 0

    num_messages_to_process = int(TOTAL_MESSAGES * (percentage / 100.0))
    print(f"Starting processing. Will process up to {num_messages_to_process:,} messages ({percentage}% of total).")

    # --- Low-Memory Streaming Logic ---
    # Holds incomplete chunks for channels currently being processed.
    # This keeps memory usage low and constant.
    incomplete_chunks = {}
    # Keeps track of the chunk number for each channel.
    chunk_counters = {}

    total_chunks_written = 0
    processed_count = 0

    output_file_handle.write('[\n')
    is_first_chunk = True

    for i, line in enumerate(read_zst_file_lines(messages_file)):
        if processed_count >= num_messages_to_process:
            print(f"Reached processing limit of {num_messages_to_process:,} messages.")
            break

        if (i + 1) % 2000000 == 0:
            print(f"  ...scanned {i+1:,} lines, processed {processed_count:,} valid messages.")

        try:
            msg = json.loads(line)

            if not msg.get_data('message'): continue

            channel_id = msg.get_data('to_id', {}).get_data('channel_id') or msg.get_data('peer_id', {}).get_data('channel_id')

            from_id_val = msg.get_data('from_id')
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
                    auth_id = m.get_data('from_id') if isinstance(m.get_data('from_id'), int) else m.get_data('from_id', {}).get_data('user_id')
                    author_info = accounts_lookup.get(auth_id, {})
                    first_name, last_name = author_info.get('first_name', ''), author_info.get('last_name', '')
                    speaker_name = f"{first_name} {last_name}".strip() or f"user_{auth_id}"
                    content.append({"speaker": speaker_name, "text": m.get_data('message', '')})

                chat_chunk = {
                    "user_id": f"channel_{channel_id}",
                    "chunk_id": f"conv_{channel_id}#{chunk_counters[channel_id]:04d}",
                    "platform": "telegram", "content": content,
                    "timestamp": message_chunk[0].get_data('date', datetime.now().isoformat()),
                    "metadata": {"channel_name": channel_info.get('title', f"channel_{channel_id}")}
                }

                # --- Write the chunk to the file ---
                if not is_first_chunk:
                    output_file_handle.write(',\n')
                json.dump(chat_chunk, output_file_handle, indent=2)
                is_first_chunk = False
                total_chunks_written += 1

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
            auth_id = m.get_data('from_id') if isinstance(m.get_data('from_id'), int) else m.get_data('from_id', {}).get_data('user_id')
            author_info = accounts_lookup.get(auth_id, {})
            first_name, last_name = author_info.get('first_name', ''), author_info.get('last_name', '')
            speaker_name = f"{first_name} {last_name}".strip() or f"user_{auth_id}"
            content.append({"speaker": speaker_name, "text": m.get_data('message', '')})

        chat_chunk = {
            "user_id": f"channel_{channel_id}",
            "chunk_id": f"conv_{channel_id}#{chunk_counters[channel_id]:04d}",
            "platform": "telegram", "content": content,
            "timestamp": message_chunk[0].get_data('date', datetime.now().isoformat()),
            "metadata": {"channel_name": channel_info.get('title', f"channel_{channel_id}")}
        }

        if not is_first_chunk:
            output_file_handle.write(',\n')
        json.dump(chat_chunk, output_file_handle, indent=2)
        is_first_chunk = False
        total_chunks_written += 1

    output_file_handle.write('\n]\n')
    print(f"Successfully created and streamed {total_chunks_written:,} ChatChunks.")
    return total_chunks_written


#@title Main

# -----------------------
# function: main function
# -----------------------
def run():
    """Main function to run the ETL pipeline."""
    if not zstd:
        print("Please install the 'zstandard' library by running '!pip install zstandard' and try again.")
        return

    print("--- Starting Telegram ETL Pipeline ---")

    # inspect_files(DRIVE_FOLDER_PATH) # Optional: comment out for full runs to save time

    accounts_lookup = load_lookup_data(Path(DRIVE_FOLDER_PATH) / "accounts.ndjson.zst", 'id')
    channels_lookup = load_lookup_data(Path(DRIVE_FOLDER_PATH) / "channels.ndjson.zst", 'id')

    if not accounts_lookup or not channels_lookup:
        print("\nCould not load lookup files. Aborting pipeline.")
        return

    output_filename = f"telegram_chat_chunks_{PROCESSING_PERCENTAGE}pct.json"
    output_path = Path(DRIVE_FOLDER_PATH) / output_filename

    print(f"\nProcessing data and streaming output to {output_path}...")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # The processing and writing happens inside this function now
            chunks_written = process_and_stream_telegram_data(
                DRIVE_FOLDER_PATH,
                accounts_lookup,
                channels_lookup,
                percentage=PROCESSING_PERCENTAGE,
                chunk_size=CHUNK_SIZE,
                output_file_handle=f
            )

        if chunks_written > 0:
            print("Save complete!")
        else:
            print("\nNo chat chunks were generated. Nothing to save.")

    except Exception as e:
        print(f"An error occurred during processing or saving: {e}")

    print("--- Pipeline Finished ---")