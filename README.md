# Universal User Embeddings

A comprehensive implementation of Universal User Embeddings based on the research paper "Universal User Embeddings". This project implements a contrastive learning approach to learn universal representations of users from their text history that can be used for preference inference, personality analysis, and user similarity computation.

## Overview

Universal User Embeddings (UUEs) are representations learned from a user's full text history (e.g., messages, posts, reviews) that enable preference inference across natural language preferences. The approach bridges the gap between user modeling and natural language understanding, providing a foundation for scalable, cross-domain user modeling in alignment-sensitive systems.

### Key Features

- **Dual-Encoder Architecture**: Shared transformer encoder for both user chunks and behavioral statements
- **Attention Pooling**: Hierarchical attention mechanism for aggregating user chunk representations
- **Contrastive Learning**: Symmetric InfoNCE loss for training user-statement alignments
- **Zero-shot Inference**: Open-vocabulary preference inference without retraining
- **Comprehensive Evaluation**: Multiple evaluation protocols for generalization and performance assessment

## Architecture

The model consists of two main components:

1. **User Encoder**: Processes user history chunks with attention pooling
2. **Statement Encoder**: Processes behavioral statements with optional context

Both encoders share weights and produce L2-normalized embeddings in a shared representational space.

### Model Components

- **Base Model**: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) from Hugging Face
- **Attention Pooling**: Multi-head attention mechanism for chunk aggregation
- **Contrastive Loss**: Symmetric InfoNCE loss with learnable temperature
- **Negative Sampling**: Multiple strategies (random, hard, semi-hard)

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.1+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-personality-alignment
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Quick Start

### Running the Demo

The easiest way to get started is to run the comprehensive demo:

```bash
python -m src.train.demo
```

This will:
1. Train a Universal User Embeddings model on sample data
2. Evaluate the model performance
3. Demonstrate inference capabilities
4. Show statement similarity analysis
5. Generate a comprehensive report

### Training Your Own Model

1. **Prepare your data** in the required format (see Data Format section)

2. **Create a configuration**:
```python
from src.schemas.universal_schema import ModelConfig

config = ModelConfig(
    base_model_name="Qwen/Qwen3-Embedding-0.6B",
    embedding_dim=1024,
    batch_size=32,
    learning_rate=2e-5,
    temperature=0.1,
    num_epochs=10
)
```

3. **Train the model**:
```python
from src.train.trainer import UniversalUserEmbeddingTrainer

trainer = UniversalUserEmbeddingTrainer(
    config=config,
    train_data=your_train_data,
    eval_data=your_eval_data,
    statement_pool=your_statement_pool
)

trainer.train()
```

### Using the Trained Model

```python
from src.train.evaluation import UniversalUserEmbeddingEvaluator

# Load the trained model
evaluator = UniversalUserEmbeddingEvaluator("path/to/model.pt")

# Get user embedding
user_chunks = ["I love programming", "Working on AI projects"]
user_embedding = evaluator.get_user_embedding(user_chunks)

# Get statement embedding
statement_embedding = evaluator.get_statement_embedding("interested in technology")

# Compute similarity
similarity = evaluator.compute_similarity(user_chunks, "interested in technology")
print(f"Similarity: {similarity:.4f}")
```

## Data Format

### User Data Structure

```python
from src.schemas.universal_schema import UserData, UserChunk

user_data = UserData(
    user_id="user_001",
    chunks=[
        UserChunk(
            text="I love discussing machine learning and AI.",
            platform="reddit",
            timestamp=datetime.now()
        ),
        UserChunk(
            text="Working on a new project involving user embeddings.",
            platform="whatsapp",
            timestamp=datetime.now()
        )
    ],
    statements=[
        "interested in machine learning",
        "enjoys reading research papers",
        "works on AI projects"
    ]
)
```

### Training Data

The training data should consist of:
- **User Data**: List of `UserData` objects with chunks and positive statements
- **Statement Pool**: List of behavioral statements for negative sampling
- **Evaluation Data**: Separate set of users for evaluation

## Configuration

The model can be configured using the `ModelConfig` class:

```python
@dataclass
class ModelConfig:
    # Model architecture
    base_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    max_chunk_length: int = 250
    max_statement_length: int = 128
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    temperature: float = 0.1
    num_epochs: int = 10
    
    # Data processing
    num_chunks_per_user: int = 100
    num_negative_samples: int = 15
    
    # Architecture specific
    use_attention_pooling: bool = True
    attention_heads: int = 8
    dropout: float = 0.1
```

## Evaluation

The model provides comprehensive evaluation capabilities:

### 1. Preference Statement Retrieval
Evaluates how well the model can retrieve relevant behavioral statements for users.

### 2. Zero-shot Preference Inference
Tests the model's ability to infer user preferences for unseen statements.

### 3. Personality Trait Regression
Analyzes how well user embeddings correlate with personality traits.

### 4. User Similarity Analysis
Examines the structure of user embeddings in the latent space.

### Running Evaluation

```python
from src.train.evaluation import UniversalUserEmbeddingEvaluator, create_evaluation_report

evaluator = UniversalUserEmbeddingEvaluator("path/to/model.pt")
report = create_evaluation_report(evaluator, test_users, candidate_statements)
```

## Command Line Interface

### Training

```bash
# Train with default configuration
python -m src.train.trainer

# Train with custom configuration
python -m src.train.trainer --config config.json --output_dir ./outputs

# Resume training from checkpoint
python -m src.train.trainer --resume checkpoint.pt
```

### Evaluation

```bash
# Evaluate trained model
python -m src.train.evaluation --model_path model.pt --output_dir ./evaluation
```

## Project Structure

```
src/
├── schemas/
│   ├── __init__.py
│   └── universal_schema.py          # Data schemas and configuration
├── train/
│   ├── __init__.py
│   ├── model.py                     # Model architecture
│   ├── loss.py                      # Contrastive loss functions
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── trainer.py                   # Training pipeline
│   ├── evaluation.py                # Evaluation framework
│   └── demo.py                      # Comprehensive demo script
└── utils/
    ├── __init__.py
    └── hf_upload.py                 # Hugging Face utilities
```

## Model Architecture Details

### Dual-Encoder Framework

The model uses a shared transformer encoder for both user chunks and behavioral statements:

1. **User Encoder**:
   - Processes user history chunks independently
   - Uses attention pooling to aggregate chunk representations
   - Produces a single user embedding vector

2. **Statement Encoder**:
   - Processes behavioral statements with optional context
   - Produces statement embedding vectors

### Attention Pooling

The attention pooling mechanism aggregates user chunk representations:

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        # Multi-head attention for chunk aggregation
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
```

### Contrastive Learning

The model uses symmetric InfoNCE loss:

```
L = (1/B) * sum_i [-log(exp(sim(u_i, s_i^+) / τ) / sum_j exp(sim(u_i, s_j^+) / τ)) 
                   -log(exp(sim(s_i^+, u_i) / τ) / sum_j exp(sim(s_i^+, u_j) / τ))]
```

## Performance

The model achieves strong performance on various evaluation metrics:

- **Preference Retrieval**: High precision and recall for retrieving relevant behavioral statements
- **Zero-shot Inference**: Effective generalization to unseen statements
- **User Similarity**: Meaningful clustering of users in the embedding space

## Applications

Universal User Embeddings can be used for:

1. **Personalized Recommendations**: User preference inference for recommendation systems
2. **Content Personalization**: Tailoring content based on user embeddings
3. **User Clustering**: Identifying similar users for community building
4. **Personality Analysis**: Understanding user traits from text behavior
5. **Alignment Systems**: Ensuring AI systems align with user preferences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
WIP
```

## Acknowledgments
- Uses the Qwen3-Embedding-0.6B model from Alibaba Cloud
- Built with PyTorch and Hugging Face Transformers

## Support

For questions and support, please open an issue on the GitHub repository.