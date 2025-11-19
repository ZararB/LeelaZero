# LeelaZero Training Pipeline

Training infrastructure for LeelaZero-style chess neural networks. This codebase handles the full training pipeline from data loading to model checkpointing, built on TensorFlow and Keras.

## About This Fork

This repository is a fork of the LeelaZero training codebase. The original code was developed by Gian-Carlo Pascutto, Folkert Huizinga, and the LCZero Authors as part of the LeelaZero/Leela Chess project.

**Original Repository:** The training code originates from the LeelaZero/Leela Chess training infrastructure, which is part of the broader [LeelaZero project](https://github.com/LeelaChessZero/lc0).

### Changes Made in This Fork

This fork includes the following modifications and improvements:

- **Weight Export Functionality**: Added functionality to save model weights in text format (both compressed and uncompressed) for easier inspection and debugging
- **Model Evaluation Enhancements**: Modified the `evaluate` function in the `keras_net` class to output probabilities directly
- **Version 4 Weights Compatibility**: Updated code to be compatible with LeelaZero version 4 weight formats
- **Additional Tools**: Added `fool.py` utility script
- **Submodule Integration**: Integrated `lczero_tools` as a submodule for weight decoding and manipulation
- **Training Pipeline Refactoring**: Refactored and improved the training pipeline for better maintainability and performance
- **Documentation Improvements**: Cleaned up and rewrote documentation for better clarity
- **Bug Fixes**: Fixed various bugs related to model loading, weight decoding (UTF-8 support), and other minor issues
- **Configuration Management**: Added and improved YAML configuration files for easier training setup
- **Dependency Management**: Froze pip requirements for reproducible builds

## What This Does

This is a complete training system for chess neural networks. It takes game data in binary format, trains a ResNet-based network with squeeze-excitation blocks, and produces models compatible with LeelaZero. The network predicts both move probabilities (policy) and position evaluations (value).

The training pipeline supports various network configurations, multiple data formats, and includes techniques like stochastic weight averaging and batch renormalization. It's designed to handle large datasets efficiently through multiprocessing and optimized data loading.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

You'll need TensorFlow 2.0.3 or later, NumPy, PyYAML, and Protobuf. If you need to regenerate the protobuf files:

```bash
protoc --python_out=. libs/lczero-common/proto/*.proto
```

## Getting Started

First, generate some test data to make sure everything works:

```bash
python generate_test_data.py
```

This creates synthetic training chunks in `test_data/train/` and `test_data/test/`. You should see output confirming the files were created.

Run a quick training test:

```bash
python train.py --cfg configs/test_config.yaml
```

The test config uses a small network (64 filters, 6 blocks) and runs for just 20 steps. It's meant to verify the setup works without using many resources.

For actual training, use the example configuration:

```bash
python train.py --cfg configs/example.yaml --output my_model
```

To resume from a checkpoint:

```bash
python train.py --cfg configs/example.yaml --resume
```

## Configuration

Everything is configured through YAML files. There are three main sections:

**Model settings** control the network architecture - number of filters, residual blocks, squeeze-excitation ratio, and which heads to use (policy, value, moves left).

**Training settings** set batch size, learning rates, loss weights, and training features like SWA or batch renormalization.

**Dataset settings** specify where to find training data, how many workers to use for data loading, and how many chunks to process.

See `configs/example.yaml` for a complete example with comments explaining each option.

## Network Architecture

The network takes 112 input planes (8×8 board representations) encoding piece positions, castling rights, and game state. These go through an initial convolution, then a stack of residual blocks with squeeze-excitation. The output splits into two heads: one for policy (1858 possible moves) and one for value (win/draw/loss probabilities).

The residual blocks use 3×3 convolutions with batch normalization and ReLU activations. Squeeze-excitation adds channel attention to help the network focus on important features. The policy head uses a convolutional approach that maps to the move space, while the value head uses fully connected layers.

## Code Structure

The main components are:

- `train.py` - Entry point that loads config, sets up datasets, and runs training
- `tfprocess.py` - Handles model construction, loss functions, and the training loop
- `net.py` - Manages weight conversion between TensorFlow and LeelaZero protobuf formats
- `chunkparser.py` - Parses binary training data with multiprocessing support
- `generate_test_data.py` - Creates synthetic data for testing

The training loop handles checkpointing, learning rate scheduling, gradient clipping, and logging. Metrics go to TensorBoard in the `leelalogs/` directory.

## Training Output

When training runs, you'll see output like this:

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

Losses should decrease over time, and accuracy should improve. The positions per second metric shows how fast the data pipeline is running. Checkpoints are saved automatically at the intervals you configure.

View detailed metrics in TensorBoard:

```bash
tensorboard --logdir leelalogs/
```

## Data Format

Training data comes in binary chunks. The code supports V3, V4, and V5 formats. V5 is the most complete, including moves left predictions and input format tags. Each record contains 112 input planes, 1858 policy probabilities, a 3-value WDL target, and metadata like Q values.

## Customization

To modify the network architecture, edit `construct_net_v2()` in `tfprocess.py`. Loss functions are defined in `TFProcess.init_net_v2()` - you can adjust the policy, value, or moves-left losses there.

## Common Issues

If you run out of memory, reduce the batch size or increase batch splits. For slow training, add more data loading workers or enable mixed precision (FP16). Make sure your data paths in the config are correct and the chunk files are readable.

## License

GNU General Public License v3.0

## References

- [LeelaZero](https://github.com/LeelaChessZero/lc0) - Original LeelaZero engine project
- [LeelaZero Training](https://github.com/LeelaChessZero/lczero-training) - Original training repository (if available)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Research background

## Credits

Original code copyright:
- Copyright (C) 2017-2018 Gian-Carlo Pascutto
- Copyright (C) 2018 Folkert Huizinga  
- Copyright (C) 2019 The LCZero Authors

This fork maintains the same GNU General Public License v3.0 as the original project.
