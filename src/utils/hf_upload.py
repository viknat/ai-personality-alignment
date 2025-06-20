#!/usr/bin/env python3
"""
Utility functions for uploading processed data to the Hugging Face Hub.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from huggingface_hub import HfApi, create_repo, upload_file


def upload_to_huggingface(
    file_path: str,
    repo_id: str,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    private: bool = False,
    commit_message: Optional[str] = None
) -> str:
    """
    Upload a JSON file to the Hugging Face Hub.
    
    Args:
        file_path: Path to the JSON file to upload
        repo_id: Repository ID on Hugging Face (format: username/repo_name)
        token: Hugging Face API token (if None, will look for HF_TOKEN env var)
        repo_type: Type of repository ('dataset' or 'model')
        private: Whether the repository should be private
        commit_message: Custom commit message (if None, a default message is used)
        
    Returns:
        URL of the uploaded file on Hugging Face Hub
        
    Raises:
        ValueError: If the file doesn't exist or token is not provided
    """
    # Validate file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Hugging Face token not provided. Either pass it as an argument "
                "or set the HF_TOKEN environment variable."
            )
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, token=token, repo_type=repo_type, private=private, exist_ok=True)
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Set default commit message if not provided
    if commit_message is None:
        file_name = os.path.basename(file_path)
        commit_message = f"Upload {file_name}"
    
    # Upload the file
    print(f"Uploading {file_path} to {repo_id}...")
    url = api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message
    )
    
    print(f"File uploaded successfully to: {url}")
    return url


def upload_dataset_with_metadata(
    data_file_path: str,
    repo_id: str,
    metadata: Dict[str, Any],
    token: Optional[str] = None,
    private: bool = False
) -> Dict[str, str]:
    """
    Upload a dataset file along with its metadata to the Hugging Face Hub.
    
    Args:
        data_file_path: Path to the data file to upload
        repo_id: Repository ID on Hugging Face (format: username/repo_name)
        metadata: Dictionary containing dataset metadata
        token: Hugging Face API token (if None, will look for HF_TOKEN env var)
        private: Whether the repository should be private
        
    Returns:
        Dictionary with URLs of the uploaded files
    """
    # Upload the data file
    data_url = upload_to_huggingface(
        file_path=data_file_path,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=private,
        commit_message=f"Upload dataset: {os.path.basename(data_file_path)}"
    )
    
    # Create and upload a README.md file
    readme_content = f"""# {metadata.get('title', 'WhatsApp Chat Dataset')}

{metadata.get('description', 'Processed WhatsApp chat data in JSON format.')}

## Dataset Information

- **Source**: {metadata.get('source', 'WhatsApp chat data')}
- **Format**: {metadata.get('format', 'JSON')}
- **Size**: {metadata.get('size', 'N/A')}
- **Created**: {metadata.get('created', 'N/A')}

## Schema

```json
{
  "user_id": "string",
  "chunk_id": "string",
  "platform": "string",
  "content": [
    {
      "speaker": "SELF | OTHER",
      "text": "string"
    }
  ],
  "timestamp": "ISO datetime string"
}
```

## License

{metadata.get('license', 'Please refer to the original data source for license information.')}
"""
    
    readme_path = os.path.join(os.path.dirname(data_file_path), "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Upload the README
    readme_url = upload_to_huggingface(
        file_path=readme_path,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=private,
        commit_message="Add README"
    )
    
    # Create and upload dataset-metadata.json
    metadata_content = {
        "title": metadata.get('title', 'WhatsApp Chat Dataset'),
        "description": metadata.get('description', 'Processed WhatsApp chat data in JSON format.'),
        "license": metadata.get('license', ''),
        "language": metadata.get('language', 'en'),
        "tags": metadata.get('tags', ['chat', 'whatsapp', 'conversation']),
        "creator": metadata.get('creator', ''),
    }
    
    metadata_path = os.path.join(os.path.dirname(data_file_path), "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_content, f, indent=2)
    
    # Upload the metadata file
    metadata_url = upload_to_huggingface(
        file_path=metadata_path,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=private,
        commit_message="Add dataset metadata"
    )
    
    # Clean up temporary files
    os.remove(readme_path)
    os.remove(metadata_path)
    
    return {
        "data": data_url,
        "readme": readme_url,
        "metadata": metadata_url
    }


def count_chat_chunks(file_path: str) -> Dict[str, Any]:
    """
    Count the number of chat chunks and messages in a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with statistics
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    total_chunks = len(data)
    total_messages = sum(len(chunk['content']) for chunk in data)
    platforms = {}
    users = {}
    
    for chunk in data:
        platform = chunk.get('platform', 'unknown')
        user_id = chunk.get('user_id', 'unknown')
        
        platforms[platform] = platforms.get(platform, 0) + 1
        users[user_id] = users.get(user_id, 0) + 1
    
    return {
        "total_chunks": total_chunks,
        "total_messages": total_messages,
        "platforms": platforms,
        "users": users,
        "file_size_bytes": os.path.getsize(file_path)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload a dataset to Hugging Face Hub')
    parser.add_argument('--file', type=str, required=True, help='Path to the JSON file to upload')
    parser.add_argument('--repo-id', type=str, required=True, help='Repository ID on Hugging Face (username/repo_name)')
    parser.add_argument('--token', type=str, help='Hugging Face API token (if not provided, will look for HF_TOKEN env var)')
    parser.add_argument('--private', action='store_true', help='Make the repository private')
    parser.add_argument('--title', type=str, default='WhatsApp Chat Dataset', help='Dataset title')
    parser.add_argument('--description', type=str, default='Processed WhatsApp chat data in JSON format.', help='Dataset description')
    args = parser.parse_args()
    
    # Get statistics about the dataset
    stats = count_chat_chunks(args.file)
    
    # Prepare metadata
    metadata = {
        'title': args.title,
        'description': args.description,
        'size': f"{stats['total_chunks']} chunks, {stats['total_messages']} messages",
        'created': Path(args.file).stat().st_mtime,
        'tags': ['chat', 'whatsapp', 'conversation'],
    }
    
    # Upload to Hugging Face Hub
    upload_dataset_with_metadata(
        data_file_path=args.file,
        repo_id=args.repo_id,
        metadata=metadata,
        token=args.token,
        private=args.private
    )
