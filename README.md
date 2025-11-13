# LeelaZero Training Pipeline

A training system for LeelaZero-style chess neural networks built with TensorFlow and Keras. This implementation provides a complete pipeline for training deep learning models on chess game data, with support for advanced training techniques and efficient data processing.

## Overview

This project implements the training infrastructure for chess neural networks similar to LeelaZero. The system handles everything from data loading and preprocessing to model training, checkpointing, and weight serialization. It's designed to work with large-scale distributed training data while maintaining flexibility for experimentation.

The architecture is based on ResNet with squeeze-excitation blocks, producing policy and value predictions for chess positions. The training pipeline supports multiple input formats, various network configurations, and includes several advanced techniques to improve training stability and model performance.

## Features

The training system includes several key components:

**Neural Network Architecture**
- ResNet-based architecture with configurable depth and width
- Squeeze-excitation blocks for channel attention
- Multiple output heads: policy (convolutional or classical), value (WDL or classical), and optional moves-left prediction
- Support for different input encodings (classical, canonical variants)

**Training Infrastructure**
- Stochastic Weight Averaging (SWA) for improved generalization
- Batch renormalization for stable training with small batches
- Mixed precision training (FP16) for faster computation
- Gradient clipping to prevent explosion
- Learning rate scheduling with warmup
- Legal move masking during policy training
- Weighted multi-component loss functions

**Data Pipeline**
- Efficient binary format parsing supporting V3, V4, and V5 data formats
- Multiprocessing for parallel data loading
- Shuffle buffers for data randomization
- Support for large-scale distributed training datasets

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are TensorFlow 2.0.3 or later, NumPy, PyYAML, and Protobuf. If you need to regenerate protobuf files from the source definitions:

```bash
protoc --python_out=. libs/lczero-common/proto/*.proto
```

## Quick Start

Generate synthetic test data to verify the setup:

```bash
python generate_test_data.py
```

This creates test chunks in `test_data/train/` and `test_data/test/` directories. You should see output like:

```
Generating training data...
Generated 100 records in test_data/train/training.1.gz
...
Generating test data...
Generated 100 records in test_data/test/training.1.gz
Test data generation complete!
```

Run training with the test configuration:

```bash
python train.py --cfg configs/test_config.yaml
```

This will train a small network (64 filters, 6 residual blocks) for 20 steps as a quick verification. The test configuration is optimized for fast validation with minimal resource usage.

For production training, use a full configuration:

```bash
python train.py --cfg configs/example.yaml --output my_model
```

To resume training from a checkpoint:

```bash
python train.py --cfg configs/example.yaml --resume
```

## Configuration

Training behavior is controlled through YAML configuration files. The configuration is divided into three main sections: model architecture, training parameters, and dataset settings.

**Model Configuration**

Controls the neural network architecture:

```yaml
model:
  filters: 128              # Number of filters in residual blocks
  residual_blocks: 10       # Number of residual blocks
  se_ratio: 4              # Squeeze-excitation ratio
  value_channels: 32       # Value head channels
  moves_left: 'v1'         # Moves left head type (or 'none')
  input_type: 'classic'    # Input format
  policy: 'convolution'    # Policy head type
  value: 'wdl'            # Value head type
```

**Training Configuration**

Sets training hyperparameters and behavior:

```yaml
training:
  batch_size: 4096
  total_steps: 1000000
  lr_values: [0.0002, 0.0002]
  lr_boundaries: [100]
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  swa: true
  swa_steps: 25
```

**Dataset Configuration**

Specifies data sources and loading parameters:

```yaml
dataset:
  input_train: '/path/to/train/chunks/*.gz'
  input_test: '/path/to/test/chunks/*.gz'
  train_workers: 16
  test_workers: 8
  num_chunks: 100000
```

See `configs/example.yaml` for a complete configuration example.

## Architecture

The network processes chess positions through the following structure:

```
Input (112 planes × 8×8)
  ↓
Initial Convolution (3×3, filters)
  ↓
Residual Blocks × N
  ├─ Conv 3×3 → BN → ReLU
  ├─ Conv 3×3 → BN → SE
  └─ Residual Connection
  ↓
Policy Head                    Value Head
  ├─ Conv 3×3 → BN → ReLU      ├─ Conv 1×1 → BN → ReLU
  ├─ Conv 3×3 (80 channels)    ├─ Dense 128 → ReLU
  └─ Policy Map                 └─ Dense 3 (WDL)
  ↓                              ↓
Policy Output (1858 moves)    Value Output
```

The input consists of 112 feature planes encoding the board state, piece positions, castling rights, and other game information. The network processes this through a series of residual blocks with squeeze-excitation, then branches into separate heads for policy (move probabilities) and value (position evaluation).

## Key Components

**TFProcess** (`tfprocess.py`)
Main training class that handles model construction, loss computation, and the training loop. Manages checkpointing, SWA updates, and logging.

**Net** (`net.py`)
Handles network weight management and conversion between TensorFlow formats and LeelaZero protobuf format. Supports loading and saving model weights.

**ChunkParser** (`chunkparser.py`)
Efficient binary data parser that reads training chunks in V3/V4/V5 formats. Uses multiprocessing for parallel data loading and includes shuffle buffers for randomization.

**train.py**
Main entry point for training. Handles configuration loading, dataset creation, and orchestrates the training process.

## File Structure

```
.
├── train.py                 # Main training script
├── tfprocess.py            # Training process and model definition
├── net.py                  # Network weight management
├── chunkparser.py          # Data parsing pipeline
├── keras_net.py            # Model inference wrapper
├── generate_test_data.py   # Synthetic data generator
├── configs/                # Configuration files
│   ├── example.yaml
│   └── test_config.yaml
├── scripts/                # Utility scripts
└── proto/                  # Protobuf definitions
```

## Advanced Usage

**Custom Network Architecture**

Modify `construct_net_v2()` in `tfprocess.py` to change the network structure:

```python
def construct_net_v2(self, inputs):
    # Modify architecture here
    ...
```

**Custom Loss Functions**

Loss functions are defined in `TFProcess.init_net_v2()`. You can modify the policy, value, or moves-left loss functions to experiment with different objectives.

**Data Format**

Training data uses a binary format with three versions:
- V3: Legacy format
- V4: Extended format with root/best Q values
- V5: Full format with moves left and input format tags

Each training record contains 112 input planes (8×8 board representation), 1858 policy probabilities, a 3-value WDL target (win/draw/loss), and various metadata including Q values and move counts.

## Monitoring Training

Training metrics are logged to TensorBoard. View them with:

```bash
tensorboard --logdir leelalogs/
```

The logs include policy and value losses, accuracies, MSE loss, learning rate, gradient norms, and weight update ratios. This helps track training progress and diagnose issues.

## Example Training Output

Here's a sample of what training output looks like:

```
2025-11-12 23:15:32,124 - __main__ - INFO - Loading configuration from configs/test_config.yaml
2025-11-12 23:15:32,128 - __main__ - INFO - Found 9 chunk files
2025-11-12 23:15:32,128 - __main__ - INFO - Creating training dataset...
Using 2 worker processes.
2025-11-12 23:15:33,456 - __main__ - INFO - Starting training:
2025-11-12 23:15:33,456 - __main__ - INFO -   Batch size: 64
2025-11-12 23:15:33,456 - __main__ - INFO -   Total steps: 20

step 0, policy=2.345 value=0.892 mse=0.234 policy accuracy=12.5% value accuracy=45.2%
step 5, lr=0.01 policy=2.123 value=0.856 mse=0.198 reg=0.012 total=2.991 (1250.3 pos/s)
step 10, policy=1.987 value=0.823 mse=0.167 policy accuracy=18.3% value accuracy=52.1%
step 15, lr=0.01 policy=1.856 value=0.789 mse=0.145 reg=0.011 total=2.656 (1280.7 pos/s)
step 20, policy=1.734 value=0.756 mse=0.128 policy accuracy=22.7% value accuracy=58.3%
Model saved in file: test_model/test-64x6-20
Weights saved as 'test_model/test-64x6-20.pb.gz' 2.1M
```

The output shows training progress with policy and value losses decreasing over time, along with improving accuracy metrics. The training speed (positions per second) indicates data pipeline efficiency. Checkpoints are automatically saved at specified intervals.

## Troubleshooting

**Out of Memory**

If you encounter memory issues, try reducing the batch size, increasing the number of batch splits, or reducing the shuffle buffer size in your configuration.

**Slow Training**

Training speed can be improved by increasing the number of data loading workers, enabling mixed precision training (FP16), and ensuring data is stored on fast storage like SSDs or RAM disks.

**No Training Data Found**

Verify that the paths specified in `input_train` and `input_test` are correct, that chunk files exist at those locations, and that file permissions allow reading.

## License

This project is licensed under the GNU General Public License v3.0.

## References

- [LeelaZero](https://github.com/LeelaChessZero/lc0) - The original LeelaZero project
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - The research paper that inspired this work
- [TensorFlow Documentation](https://www.tensorflow.org/) - Framework documentation

## Acknowledgments

This codebase is based on the LeelaZero training infrastructure, with improvements focused on maintainability, code quality, and testing.
