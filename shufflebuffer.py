#!/usr/bin/env python3
#
# Shuffle buffer implementation for data pipeline
# Provides efficient shuffling of binary records

import random


class ShuffleBuffer:
    """
    A buffer that maintains a random sample of items.
    When full, returns a random item for each new item inserted.
    """
    
    def __init__(self, record_size, buffer_size):
        """
        Args:
            record_size: Size of each record in bytes
            buffer_size: Maximum number of records to hold
        """
        self.record_size = record_size
        self.buffer_size = buffer_size
        self.buffer = []
        self.count = 0

    def insert_or_replace(self, item):
        """
        Insert an item into the buffer. If buffer is full,
        return a random item to be replaced.
        
        Args:
            item: The record to insert (bytes)
            
        Returns:
            A random item from the buffer if full, None otherwise
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
            self.count += 1
            return None
        else:
            # Buffer is full, replace a random item
            idx = random.randint(0, self.buffer_size - 1)
            old_item = self.buffer[idx]
            self.buffer[idx] = item
            self.count += 1
            return old_item

    def extract(self):
        """
        Extract a random item from the buffer.
        
        Returns:
            A random item from the buffer, or None if empty
        """
        if not self.buffer:
            return None
        idx = random.randint(0, len(self.buffer) - 1)
        return self.buffer.pop(idx)

