#!/usr/bin/env python3
#
# Main training script for LeelaZero neural network
#
# This script loads configuration, sets up data pipelines,
# and runs the training loop.

import argparse
import glob
import logging
import os
import sys
import tensorflow as tf
import yaml

import chunkparser
import tfprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_chunks(input_paths, num_chunks=None, allow_less=False):
    """
    Find training chunk files from input paths.
    
    Args:
        input_paths: List of glob patterns or directories
        num_chunks: Maximum number of chunks to use (None = all)
        allow_less: If True, allow fewer chunks than requested
        
    Returns:
        List of chunk file paths
    """
    chunks = []
    for path in input_paths:
        if os.path.isdir(path):
            # Find all .gz files in directory
            pattern = os.path.join(path, "*.gz")
            chunks.extend(glob.glob(pattern))
        else:
            # Treat as glob pattern
            chunks.extend(glob.glob(path))
    
    # Sort and limit
    chunks = sorted(chunks)
    if num_chunks is not None:
        if len(chunks) < num_chunks:
            if not allow_less:
                logger.warning(
                    f"Found {len(chunks)} chunks, requested {num_chunks}. "
                    "Set allow_less_chunks: true to proceed."
                )
                sys.exit(1)
            else:
                logger.warning(
                    f"Found {len(chunks)} chunks, requested {num_chunks}. "
                    "Proceeding with available chunks."
                )
        chunks = chunks[:num_chunks]
    
    logger.info(f"Found {len(chunks)} chunk files")
    return chunks


def create_dataset(chunks, cfg, is_test=False):
    """
    Create a TensorFlow dataset from chunk files.
    
    Args:
        chunks: List of chunk file paths
        cfg: Configuration dictionary
        is_test: If True, create test dataset (no shuffling)
        
    Returns:
        TensorFlow dataset
    """
    dataset_cfg = cfg.get('dataset', {})
    workers = dataset_cfg.get('test_workers' if is_test else 'train_workers', 4)
    batch_size = cfg['training']['batch_size']
    shuffle_size = cfg['training'].get('shuffle_size', 10000)
    
    # Determine input format from model config
    model_cfg = cfg.get('model', {})
    input_mode = model_cfg.get('input_type', 'classic')
    
    # Map input mode to expected format
    input_format_map = {
        'classic': 1,
        'frc_castling': 2,
        'canonical': 3,
        'canonical_100': 4,
        'canonical_armageddon': 132,
        'canonical_v2': 5,
        'canonical_v2_armageddon': 133
    }
    expected_input_format = input_format_map.get(input_mode, 1)
    
    # Create parser
    parser = chunkparser.ChunkParser(
        chunks,
        expected_input_format=expected_input_format,
        shuffle_size=shuffle_size if not is_test else 1,
        batch_size=batch_size,
        workers=workers
    )
    
    # Create dataset from generator
    def gen():
        for batch in parser.parse():
            yield batch
    
    output_types = (tf.string, tf.string, tf.string, tf.string, tf.string)
    output_shapes = ((), (), (), (), ())
    
    dataset = tf.data.Dataset.from_generator(gen, output_types, output_shapes)
    
    # Parse batches
    def parse_batch(planes, probs, winner, q, plies_left):
        return chunkparser.ChunkParser.parse_function(
            planes, probs, winner, q, plies_left
        )
    
    dataset = dataset.map(parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, parser


def main():
    parser = argparse.ArgumentParser(description='Train LeelaZero neural network')
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output model name (optional)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.cfg}")
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override output name if provided
    if args.output:
        cfg['name'] = args.output
    
    # Setup GPU if specified
    use_gpu = 'gpu' in cfg and cfg['gpu'] is not None
    if use_gpu:
        logger.info(f"Using GPU {cfg['gpu']}")
    
    # Find training chunks
    dataset_cfg = cfg.get('dataset', {})
    if 'input_train' in dataset_cfg and 'input_test' in dataset_cfg:
        train_chunks = find_chunks(
            [dataset_cfg['input_train']],
            num_chunks=dataset_cfg.get('num_chunks'),
            allow_less=dataset_cfg.get('allow_less_chunks', False)
        )
        test_chunks = find_chunks([dataset_cfg['input_test']])
    elif 'input' in dataset_cfg:
        all_chunks = find_chunks(
            [dataset_cfg['input']],
            num_chunks=dataset_cfg.get('num_chunks'),
            allow_less=dataset_cfg.get('allow_less_chunks', False)
        )
        train_ratio = dataset_cfg.get('train_ratio', 0.9)
        split_idx = int(len(all_chunks) * train_ratio)
        train_chunks = all_chunks[:split_idx]
        test_chunks = all_chunks[split_idx:]
    else:
        logger.error("No input data specified in configuration")
        sys.exit(1)
    
    if not train_chunks:
        logger.error("No training chunks found!")
        sys.exit(1)
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset, train_parser = create_dataset(train_chunks, cfg, is_test=False)
    
    logger.info("Creating test dataset...")
    test_dataset, test_parser = create_dataset(test_chunks, cfg, is_test=True)
    
    # Optional validation dataset
    validation_dataset = None
    if 'input_validation' in dataset_cfg:
        val_chunks = find_chunks([dataset_cfg['input_validation']])
        if val_chunks:
            validation_dataset, _ = create_dataset(val_chunks, cfg, is_test=True)
    
    # Create training process
    logger.info("Initializing training process...")
    tfp = tfprocess.TFProcess(cfg, gpu=use_gpu)
    tfp.init_v2(train_dataset, test_dataset, validation_dataset)
    
    # Restore checkpoint if resuming
    if args.resume:
        logger.info("Resuming from checkpoint...")
        tfp.restore_v2()
    
    # Get training parameters
    batch_size = cfg['training']['batch_size']
    test_batches = cfg['training'].get('num_test_positions', 10000) // batch_size
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    
    logger.info(f"Starting training:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Batch splits: {batch_splits}")
    logger.info(f"  Test batches: {test_batches}")
    logger.info(f"  Total steps: {cfg['training']['total_steps']}")
    
    # Run training
    try:
        tfp.process_loop_v2(batch_size, test_batches, batch_splits)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        train_parser.shutdown()
        test_parser.shutdown()
        logger.info("Training completed")


if __name__ == '__main__':
    main()

