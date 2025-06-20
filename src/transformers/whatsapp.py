#!/usr/bin/env python3
"""
Transform WhatsApp chat data from FoLiA XML format to the defined JSON schema.

This script reads WhatsApp chat data from FoLiA XML files and converts them
into the ChatChunk format defined in schemas/universal_schema.py file.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any
import json
import argparse
from pathlib import Path

from schemas.universal_schema import validate_chat_chunk, serialize_chat_chunks
from utils.hf_upload import upload_dataset_with_metadata, count_chat_chunks



# Define XML namespaces used in FoLiA files
NAMESPACES = {
    'folia': 'http://ilk.uvt.nl/folia',
    'xlink': 'http://www.w3.org/1999/xlink'
}


def extract_messages_from_folia(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract messages from a FoLiA XML file.
    
    Args:
        file_path: Path to the FoLiA XML file
        
    Returns:
        List of dictionaries containing message data
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract all event elements (messages)
        events = root.findall('.//folia:event', NAMESPACES)
        
        messages = []
        for event in events:
            # Check if this is a message event
            if event.get('class') == 'message':
                actor = event.get('actor', '').replace('[', '').replace(']', '')
                timestamp_str = event.get('begindatetime', '')
                
                # Get the message text
                text_element = event.find('.//folia:t', NAMESPACES)
                text = text_element.text if text_element is not None and text_element.text else ""
                
                # Add to messages list
                messages.append({
                    'actor': actor,
                    'timestamp': timestamp_str,
                    'text': text
                })
                
        return messages
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def group_messages_into_chunks(messages: List[Dict[str, Any]], 
                              chunk_size: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Group messages into chunks of specified size.
    
    Args:
        messages: List of message dictionaries
        chunk_size: Maximum number of messages per chunk
        
    Returns:
        List of message chunks
    """
    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def convert_to_chat_chunks(message_chunks: List[List[Dict[str, Any]]], 
                           file_name: str) -> List[Dict[str, Any]]:
    """
    Convert message chunks to ChatChunk dictionaries.
    
    Args:
        message_chunks: List of message chunks
        file_name: Name of the source file (used for user_id)
        
    Returns:
        List of ChatChunk dictionaries
    """
    chat_chunks = []
    
    # Extract a user ID from the file name
    user_id = f"usr_{file_name.split('_')[-1].split('.')[0]}"
    
    for i, chunk in enumerate(message_chunks):
        # Create a unique chunk ID
        chunk_id = f"conv_{file_name.split('_')[-1].split('.')[0]}#{i:02d}"
        
        # Get timestamp from the first message in the chunk
        timestamp_str = chunk[0].get('timestamp', datetime.now().isoformat())
        
        # Convert to proper ISO format if needed
        try:
            # Parse the timestamp and convert to ISO format
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp = dt.isoformat()
        except ValueError:
            # If parsing fails, use current time
            timestamp = datetime.now().isoformat()
        
        # Convert messages to MessageContent format
        content = []
        for msg in chunk:
            # Determine if the message is from SELF or OTHER
            # For simplicity, we'll consider the first user in the conversation as SELF
            # and all others as OTHER
            speaker = "SELF" if msg['actor'] == "user1" else "OTHER"
            
            content.append({
                "speaker": speaker,
                "text": msg['text']
            })
        
        # Create the ChatChunk dictionary
        chat_chunk = {
            "user_id": user_id,
            "chunk_id": chunk_id,
            "platform": "whatsapp",
            "content": content,
            "timestamp": timestamp
        }
        
        # Validate the chat chunk
        validated_chunk = validate_chat_chunk(chat_chunk)
        chat_chunks.append(validated_chunk)
    
    return chat_chunks


def process_folia_file(file_path: str, chunk_size: int = 3) -> List[Dict[str, Any]]:
    """
    Process a single FoLiA XML file and convert it to ChatChunk dictionaries.
    
    Args:
        file_path: Path to the FoLiA XML file
        chunk_size: Maximum number of messages per chunk
        
    Returns:
        List of ChatChunk dictionaries
    """
    # Extract the file name from the path
    file_name = os.path.basename(file_path)
    
    # Extract messages from the FoLiA file
    messages = extract_messages_from_folia(file_path)
    
    # Group messages into chunks
    message_chunks = group_messages_into_chunks(messages, chunk_size)
    
    # Convert to ChatChunk objects
    chat_chunks = convert_to_chat_chunks(message_chunks, file_name)
    
    return chat_chunks


def upload_to_huggingface_hub(
    output_file: str, 
    repo_id: str, 
    token: str = None, 
    private: bool = False, 
    title: str = "WhatsApp Chat Dataset",
    description: str = "Processed WhatsApp chat data in JSON format."
) -> None:
    """
    Upload the processed data to the Hugging Face Hub.
    
    Args:
        output_file: Path to the JSON output file
        repo_id: Repository ID on Hugging Face (format: username/repo_name)
        token: Hugging Face API token (if None, will look for HF_TOKEN env var)
        private: Whether the repository should be private
        title: Dataset title
        description: Dataset description
    """

    
    # Get statistics about the dataset
    stats = count_chat_chunks(output_file)
    
    # Format the creation date
    creation_date = datetime.fromtimestamp(Path(output_file).stat().st_mtime).strftime('%Y-%m-%d')
    
    # Prepare metadata
    metadata = {
        'title': title,
        'description': description,
        'source': "WhatsApp chat data in FoLiA XML format",
        'format': "JSON",
        'size': f"{stats['total_chunks']} chunks, {stats['total_messages']} messages",
        'created': creation_date,
        'tags': ['chat', 'whatsapp', 'conversation'],
        'language': 'en',
    }
    
    # Upload to Hugging Face Hub
    try:
        upload_dataset_with_metadata(
            data_file_path=output_file,
            repo_id=repo_id,
            metadata=metadata,
            token=token,
            private=private
        )
        print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")


def main() -> None:
    """Main function to process WhatsApp FoLiA XML files."""
    parser = argparse.ArgumentParser(description='Transform WhatsApp FoLiA XML files to JSON format')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing FoLiA XML files')
    parser.add_argument('--output-file', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--chunk-size', type=int, default=3, help='Number of messages per chunk')
    
    hf_group = parser.add_argument_group('Hugging Face Upload Options')
    hf_group.add_argument('--upload-to-hub', action='store_true', help='Upload the processed data to Hugging Face Hub')
    hf_group.add_argument('--repo-id', type=str, help='Repository ID on Hugging Face (username/repo_name)')
    hf_group.add_argument('--token', type=str, help='Hugging Face API token (if not provided, will look for HF_TOKEN env var)')
    hf_group.add_argument('--private', action='store_true', help='Make the repository private')
    hf_group.add_argument('--title', type=str, default='WhatsApp Chat Dataset', help='Dataset title')
    hf_group.add_argument('--description', type=str, default='Processed WhatsApp chat data in JSON format.', help='Dataset description')
    
    args = parser.parse_args()
    
    # Get all FoLiA XML files in the input directory
    input_dir = Path(args.input_dir)
    folia_files = list(input_dir.glob('*.folia.xml'))
    
    if not folia_files:
        print(f"No FoLiA XML files found in {input_dir}")
        return
    
    # Process each file
    all_chat_chunks = []
    for file_path in folia_files:
        print(f"Processing {file_path}")
        chat_chunks = process_folia_file(str(file_path), args.chunk_size)
        all_chat_chunks.extend(chat_chunks)
    
    # Write the results to the output file
    with open(args.output_file, 'w') as f:
        json_str = serialize_chat_chunks(all_chat_chunks)
        f.write(json_str)
    
    print(f"Processed {len(folia_files)} files, generated {len(all_chat_chunks)} chat chunks")
    print(f"Results written to {args.output_file}")
    
    # Upload to Hugging Face Hub if requested
    if hasattr(args, 'upload_to_hub') and args.upload_to_hub:
        if not args.repo_id:
            print("Error: --repo-id is required when using --upload-to-hub")
            return
        
        upload_to_huggingface_hub(
            output_file=args.output_file,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            title=args.title,
            description=args.description
        )


if __name__ == "__main__":
    main()
