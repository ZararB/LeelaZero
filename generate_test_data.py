#!/usr/bin/env python3
#
# Generate synthetic test data for training verification
#

import gzip
import numpy as np
import os
import struct

# Version constants matching chunkparser
V5_VERSION = struct.pack('i', 5)
CLASSICAL_INPUT = 1  # Integer value for struct format 'i'
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'

def generate_test_chunk(filename, num_records=100):
    """
    Generate a synthetic test chunk file.
    
    Args:
        filename: Output filename
        num_records: Number of training records to generate
    """
    v5_struct = struct.Struct(V5_STRUCT_STRING)
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with gzip.open(filename, 'wb') as f:
        for i in range(num_records):
            # Generate random data
            # 1858 float32 probabilities (policy)
            probs = np.random.rand(1858).astype(np.float32)
            probs = probs / probs.sum()  # Normalize
            probs_bytes = probs.tobytes()
            
            # 104 packed bit planes (832 bytes = 104 * 8)
            planes = np.random.randint(0, 2, size=(104, 64), dtype=np.uint8)
            planes_packed = np.packbits(planes).tobytes()
            
            # Castling rights (4 bytes)
            us_ooo = np.random.randint(0, 2, dtype=np.uint8)
            us_oo = np.random.randint(0, 2, dtype=np.uint8)
            them_ooo = np.random.randint(0, 2, dtype=np.uint8)
            them_oo = np.random.randint(0, 2, dtype=np.uint8)
            
            # Side to move (1 byte)
            stm = np.random.randint(0, 2, dtype=np.uint8)
            
            # Rule 50 count (1 byte)
            rule50_count = np.random.randint(0, 100, dtype=np.uint8)
            
            # Deprecated ply count (1 byte)
            dep_ply_count = 0
            
            # Winner: -1 (loss), 0 (draw), 1 (win)
            winner = np.int8(np.random.choice([-1, 0, 1]))
            
            # Q values
            root_q = np.float32(np.random.uniform(-1.0, 1.0))
            best_q = np.float32(np.random.uniform(-1.0, 1.0))
            root_d = np.float32(np.random.uniform(0.0, 0.5))
            best_d = np.float32(np.random.uniform(0.0, 0.5))
            root_m = np.float32(np.random.uniform(0.0, 200.0))
            best_m = np.float32(np.random.uniform(0.0, 200.0))
            plies_left = np.float32(np.random.uniform(0.0, 200.0))
            
            # Pack record (convert numpy types to Python native types for struct)
            record = v5_struct.pack(
                V5_VERSION,
                CLASSICAL_INPUT,  # int32 input format
                probs_bytes,
                planes_packed,
                int(us_ooo), int(us_oo), int(them_ooo), int(them_oo),
                int(stm), int(rule50_count), int(dep_ply_count),
                int(winner),
                float(root_q), float(best_q), float(root_d), float(best_d),
                float(root_m), float(best_m), float(plies_left)
            )
            
            f.write(record)
    
    print(f"Generated {num_records} records in {filename}")


def main():
    """Generate test data directories and files."""
    train_dir = 'test_data/train'
    test_dir = 'test_data/test'
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate training chunks
    print("Generating training data...")
    for i in range(9):  # 9 chunks for training
        filename = os.path.join(train_dir, f'training.{i+1}.gz')
        generate_test_chunk(filename, num_records=100)
    
    # Generate test chunks
    print("Generating test data...")
    for i in range(1):  # 1 chunk for testing
        filename = os.path.join(test_dir, f'training.{i+1}.gz')
        generate_test_chunk(filename, num_records=100)
    
    print("Test data generation complete!")
    print(f"Training chunks: {train_dir}")
    print(f"Test chunks: {test_dir}")


if __name__ == '__main__':
    main()

