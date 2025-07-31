import zstandard as zstd
import json
import io
import os
import glob
from datasets import Dataset
from huggingface_hub import HfApi
import asyncio
from torrentp import TorrentDownloader


def safe_get(data, keys, default=None):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    return default


# Clean data and transform to the format expected
def transform_comment(comment):
    user_id_keyword = ["author", "author_fullname", "author_name"]
    chunk_id_keyword = ["permalink", "link_id", "id"]
    content_keyword = ["body", "text"]
    timestamp = ["created_utc", "timestamp"]

    return {
        "user_id": "usr_" + safe_get(comment, user_id_keyword, "unknown_user"),
        "chunk_id": safe_get(comment, chunk_id_keyword, "unknown_chunk_id"),
        "platform": "reddit",
        "content": [{
            "speaker": "SELF",
            "text": safe_get(comment, content_keyword, "empty_body"),
        }],
        "timestamp": safe_get(comment, timestamp, "unknown_timestamp")
    }


# Write zst data to a local json file
# If want all data in a file, num_samples should be inf or larger than the rows exist
def process_chunks(num_samples, input_file, output_file):
    with open(input_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write('[\n')  # Start of JSON array
            first_entry = True
            processed_count = 0

            try:
                for line in text_stream:
                    if processed_count >= num_samples:
                        break
                    if not line.strip():
                        continue

                    try:
                        comment = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines

                    transformed = transform_comment(comment)

                    # Add comma separator between entries
                    if not first_entry:
                        outfile.write(',\n')
                    else:
                        first_entry = False

                    json.dump(transformed, outfile, indent=2)
                    processed_count += 1
            finally:
                outfile.write('\n]')  # End of JSON array

    print(f"Transformed {processed_count} comments successfully")


# input_folder: a folder containing all .zst files downloaded from academic torrent
# output_folder: a folder that will be used to store cleaned files to be pushed to HF
# delete_after_pushing: set True if we want to delete cleaned files locally after pushing to HF
# e.g. input_folder = r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/reddit/comments/"
# e.g. output_folder = r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/reddit/clean_comments/"
def push_to_huggingface(input_folder, output_folder, delete_after_pushing=False):
    num_samples = 100000000000  # get all data

    # Get all .zst files in the folder recursively
    input_files = glob.glob(os.path.join(input_folder, "**", "*.zst"), recursive=True)

    # Loop over each .zst file
    for zst_file in input_files:
        filename = os.path.basename(zst_file).replace(".zst", "")
        output_file = os.path.join(output_folder, f"cleaned_{filename}.json")

        print(f"Processing {zst_file} -> cleaned_{filename}.json")
        try:
            process_chunks(num_samples, zst_file, output_file)
        except Exception as e:
            print(f"Failed to process {zst_file}: {e}")

        # upload to HF
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=output_file,
                path_in_repo=f"clean_comments/cleaned_{filename}.json",  # change it if necessary
                repo_id="BoqiaoZ/reddit_comments_all",  # change it if necessary
                repo_type="dataset",
            )
            print(f"cleaned_{filename}.json pushed to HF.")

            if delete_after_pushing:
                os.remove(output_file)
        except Exception as e:
            print(f"Failed to process {output_file}: {e}")


# input_torrent_path: path to the local .torrent file
# output_folder: the folder that will be used to store .zst files downloaded from HF
# e.g. input_torrent_path = r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/test.torrent"
# e.g. output_path = r"/Users/permanj/Desktop/Cambridge Research/Deep_Alignment/"
def download_from_huggingface(input_torrent_path, output_folder):
    torrent_file = TorrentDownloader(input_torrent_path, output_folder)
    # Start the download process
    # DO ensure sufficient memory for downloading
    asyncio.run(torrent_file.start_download())  # start_download() is a asynchronous method