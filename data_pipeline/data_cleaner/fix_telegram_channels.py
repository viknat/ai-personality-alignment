import zstandard as zstd
import json
import io


def remove_restriction_reason(data):
    if isinstance(data, dict):
        # If the dictionary has a 'restriction_reason' key, remove it
        if 'restriction_reason' in data:
            del data['restriction_reason']

        # Recursively call the function for all dictionary values
        for key, value in data.items():
            data[key] = remove_restriction_reason(value)

    elif isinstance(data, list):
        # If the data is a list, recursively apply the function to each item
        return [remove_restriction_reason(item) for item in data]

    return data



def fix(
        input_path = r"C:\Users\Administrator\Desktop\data\telegram\channels.ndjson.zst",
        output_path = r"C:\Users\Administrator\Desktop\data\telegram\cleaned_channels.ndjson.zst"):
    """ Read a .zst file line by line, clean the data, and store the cleaned data. """
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
                            cleaned_obj = remove_restriction_reason(json_obj)  # Clean the JSON
                            # Convert the cleaned JSON object to a string and encode it to bytes
                            cleaned_json_str = json.dumps(cleaned_obj) + '\n'
                            writer.write(cleaned_json_str.encode('utf-8'))  # Write as bytes
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")

def test(input_path = r"C:\Users\Administrator\Desktop\data\telegram\cleaned_channels.ndjson.zst"):
    with open(input_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            # Wrap the binary stream in a text wrapper to read lines
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                print(line)
                break