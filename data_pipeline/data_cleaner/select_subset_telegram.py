import zstandard as zstd
import io
import json

def select_subset_telegram\
                (lines_of_samples,
                 input_path = r"Z:\data\telegram\messages.ndjson.zst",
                 output_path = r"Z:\data\telegram\subset_messages.ndjson.zst"):
    with open(input_path, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f_in)

        # Wrap the stream with TextIOWrapper to read as text (lines)
        text_reader = io.TextIOWrapper(reader, encoding='utf-8')

        with open(output_path, 'wb') as f_out:
            cctx = zstd.ZstdCompressor()
            with cctx.stream_writer(f_out) as writer:
                line_count = 0
                for line in text_reader:
                    if line_count >= lines_of_samples:
                        break
                    if line.strip():  # Make sure line is not empty
                        writer.write(line.encode('utf-8'))  # Write as bytes back to output
                        line_count += 1


def check_line_count(expected_count, output_path):
    with open(output_path, 'rb') as f_in:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f_in)

        # Wrap the stream with TextIOWrapper to read as text (lines)
        text_reader = io.TextIOWrapper(reader, encoding='utf-8')

        line_count = 0
        for line in text_reader:
            line_count += 1
        print(f"Number of lines in the file: {line_count}")
        return line_count == expected_count


def check_decompression(output_path):
    try:
        with open(output_path, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(f_in)
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')

            # Read and print the first few lines to verify correctness
            for i, line in enumerate(text_reader):
                if i < 5:  # Check the first 5 lines, adjust as needed
                    print(f"Line {i + 1}: {line.strip()}")
                else:
                    break
        return True
    except Exception as e:
        print(f"Error during decompression: {e}")
        return False


def check_json_format(output_path):
    try:
        with open(output_path, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(f_in)
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')

            # Attempt to parse the first few lines as JSON
            for i, line in enumerate(text_reader):
                if i < 5:  # Check the first 5 lines
                    try:
                        json_data = json.loads(line)
                        print(f"Line {i + 1} is valid JSON: {json_data}")
                    except json.JSONDecodeError:
                        print(f"Line {i + 1} is NOT valid JSON")
                else:
                    break
        return True
    except Exception as e:
        print(f"Error during JSON parsing: {e}")
        return False



def test_subset_telegram (expected_count, output_path = r"Z:\data\telegram\subset_messages.ndjson.zst"):
    return check_line_count(expected_count, output_path), check_decompression(output_path), check_json_format(output_path)