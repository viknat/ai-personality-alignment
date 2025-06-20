"""
Schema definitions for WhatsApp chat data transformation.
Uses simple Python dictionaries and JSON for data validation and serialization.
"""

from typing import List, Dict, Any, Literal, TypedDict
from datetime import datetime
import json


class MessageContent(TypedDict):
    """Represents a single message in a conversation."""
    speaker: Literal["SELF", "OTHER"]  # Indicates whether the message is from the user (SELF) or another person (OTHER)
    text: str  # The actual message text content


class ChatChunk(TypedDict):
    """Represents a chunk of conversation from a specific platform."""
    user_id: str  # Unique identifier for the user
    chunk_id: str  # Unique identifier for this conversation chunk
    platform: str  # The platform where the conversation took place (e.g., 'whatsapp', 'reddit')
    content: List[MessageContent]  # List of messages in this conversation chunk
    timestamp: str  # Timestamp when this conversation chunk was created or processed (ISO format)


def validate_message_content(message: Dict[str, Any]) -> MessageContent:
    """
    Validate a message content dictionary against the schema.
    
    Args:
        message: Dictionary containing message data
        
    Returns:
        Validated MessageContent dictionary
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(message, dict):
        raise ValueError("Message must be a dictionary")
        
    if "speaker" not in message:
        raise ValueError("Message must have a 'speaker' field")
    
    if message["speaker"] not in ["SELF", "OTHER"]:
        raise ValueError("Speaker must be either 'SELF' or 'OTHER'")
        
    if "text" not in message:
        raise ValueError("Message must have a 'text' field")
        
    if not isinstance(message["text"], str):
        raise ValueError("Message text must be a string")
        
    return {
        "speaker": message["speaker"],
        "text": message["text"]
    }


def validate_chat_chunk(chunk: Dict[str, Any]) -> ChatChunk:
    """
    Validate a chat chunk dictionary against the schema.
    
    Args:
        chunk: Dictionary containing chat chunk data
        
    Returns:
        Validated ChatChunk dictionary
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(chunk, dict):
        raise ValueError("Chat chunk must be a dictionary")
        
    required_fields = ["user_id", "chunk_id", "platform", "content", "timestamp"]
    for field in required_fields:
        if field not in chunk:
            raise ValueError(f"Chat chunk must have a '{field}' field")
            
    if not isinstance(chunk["user_id"], str):
        raise ValueError("user_id must be a string")
        
    if not isinstance(chunk["chunk_id"], str):
        raise ValueError("chunk_id must be a string")
        
    if not isinstance(chunk["platform"], str):
        raise ValueError("platform must be a string")
        
    if not isinstance(chunk["content"], list):
        raise ValueError("content must be a list")
        
    # Validate each message in the content list
    validated_content = [validate_message_content(msg) for msg in chunk["content"]]
    
    # Validate timestamp format
    if not isinstance(chunk["timestamp"], str):
        raise ValueError("timestamp must be a string in ISO format")
    
    # Try to parse the timestamp to ensure it's valid
    try:
        datetime.fromisoformat(chunk["timestamp"].replace('Z', '+00:00'))
    except ValueError:
        raise ValueError("timestamp must be a valid ISO format datetime string")
        
    return {
        "user_id": chunk["user_id"],
        "chunk_id": chunk["chunk_id"],
        "platform": chunk["platform"],
        "content": validated_content,
        "timestamp": chunk["timestamp"]
    }


def serialize_chat_chunks(chat_chunks: List[ChatChunk]) -> str:
    """
    Serialize a list of chat chunks to JSON.
    
    Args:
        chat_chunks: List of validated ChatChunk dictionaries
        
    Returns:
        JSON string representation
    """
    return json.dumps(chat_chunks, indent=2)


def deserialize_chat_chunks(json_str: str) -> List[ChatChunk]:
    """
    Deserialize a JSON string to a list of chat chunks.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        List of validated ChatChunk dictionaries
    """
    data = json.loads(json_str)
    return [validate_chat_chunk(chunk) for chunk in data]


# Example of how to use the schema
if __name__ == "__main__":
    # Example data
    example_data = [
        {
            "user_id": "usr_042",
            "chunk_id": "conv_8423#05",
            "platform": "reddit",
            "content": [
                {"speaker": "SELF", "text": "Because most people just want things to work out of the box."}
            ],
            "timestamp": "2025-06-17T18:34:20Z"
        },
        {
            "user_id": "usr_023",
            "chunk_id": "conv_8423#07",
            "platform": "whatsapp",
            "content": [
                {"speaker": "SELF", "text": "Just a small project."},
                {"speaker": "OTHER", "text": "Cool, what's it about?"},
                {"speaker": "SELF", "text": "It's a small project."}
            ],
            "timestamp": "2025-06-17T18:34:20Z"
        }
    ]
    
    # Validate the data
    validated_chunks = [validate_chat_chunk(chunk) for chunk in example_data]
    
    # Serialize to JSON
    json_str = serialize_chat_chunks(validated_chunks)
    print(json_str)
    
    # Deserialize from JSON
    deserialized_chunks = deserialize_chat_chunks(json_str)
    
    # You can now work with the validated data
    for chunk in deserialized_chunks:
        print(f"User {chunk['user_id']} on {chunk['platform']}:")
        for message in chunk["content"]:
            print(f"  [{message['speaker']}]: {message['text']}")