import zstandard as zstd
import json
import io


def clean(obj):
    # Extract 'channel_id' from either 'to_id' or 'peer_id'
    to_id_channel_id = obj.get_data('to_id', {}).get_data('channel_id', None)
    peer_id_channel_id = obj.get_data('peer_id', {}).get_data('channel_id', None)

    # Prepare the cleaned object, maintaining nested structure
    cleaned_obj = {
        "to_id": {
            "channel_id": to_id_channel_id if to_id_channel_id else "unknown"
        },
        "peer_id": {
            "channel_id": peer_id_channel_id if peer_id_channel_id else "unknown"
        },
        "from_id": obj.get_data('from_id', None),
        "message": obj.get_data("message", "unknown"),
        "date": str(obj.get_data("date", "unknown"))
    }

    return cleaned_obj

def fix():
    # Define input and output paths
    input_path = r'Z:\data\telegram\messages.ndjson.zst'
    output_path = r'Z:\data\telegram\cleaned_messages_20M.ndjson.zst'

    line_cnt = 0

    # Initialize the zstd decompressor and compressor
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor()

    with open(input_path, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f_in)
        text_streamer = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='replace')

        with open(output_path, 'wb') as f_out:
            cctx = zstd.ZstdCompressor()
            with cctx.stream_writer(f_out) as writer:
                # Read the file line by line
                for line in text_streamer:
                    # Convert to dict (JSON object), clean it, then convert back to JSON string
                    if line.strip():  # Skip empty lines
                        try:
                            json_obj = json.loads(line.strip())  # Parse the JSON
                            cleaned_obj = clean(json_obj)  # Clean the JSON

                            # Check if either of the 'channel_id' fields are valid (non-"unknown")
                            if cleaned_obj["to_id"]["channel_id"] == "unknown" and cleaned_obj["peer_id"][
                                "channel_id"] == "unknown":
                                continue

                            # Convert the cleaned JSON object to a string and encode it to bytes
                            cleaned_json_str = json.dumps(cleaned_obj) + '\n'
                            writer.write(cleaned_json_str.encode('utf-8'))  # Write as bytes

                            line_cnt += 1
                            if line_cnt >= 20000000:
                                break
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
    print("Finished the cleanup process.")