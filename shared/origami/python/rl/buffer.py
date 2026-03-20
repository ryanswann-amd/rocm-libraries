#!/usr/bin/env python3
"""Thread-safe replay buffer with persistent storage.

Stores (shape, kernel, features, measured_latency) tuples.
Supports saving to / loading from parquet for resumable training.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class ReplayBuffer:
    """Thread-safe replay buffer for online training.

    Each entry is a dict:
        {shape_id, M, N, K, kernel_sig, features, log_correction, gflops, us}

    Features are stored as numpy arrays. Supports:
    - Thread-safe append (from benchmark workers)
    - Random sampling for training
    - Persistence to/from parquet
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._shape_count = 0  # unique shapes added

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def n_shapes(self) -> int:
        with self._lock:
            return self._shape_count

    def add_shape(self, entries: list[dict]):
        """Add all kernel entries for a single shape.

        Each entry must have:
            {shape_id, M, N, K, kernel_sig, features, log_correction, gflops, us}
        where features is a 1D numpy array.
        """
        with self._lock:
            for entry in entries:
                self._buffer.append(entry)
            self._shape_count += 1

    def sample(self, batch_size: int) -> dict:
        """Sample a random mini-batch. Returns stacked numpy arrays.

        Returns:
            {features: (B, D), log_correction: (B,), shape_ids: (B,)}
        """
        with self._lock:
            n = len(self._buffer)
            if n == 0:
                return None
            idx = np.random.randint(0, n, size=min(batch_size, n))
            entries = [self._buffer[i] for i in idx]

        return {
            'features': np.stack([e['features'] for e in entries]),
            'log_correction': np.array([e['log_correction'] for e in entries], dtype=np.float32),
            'shape_ids': np.array([e['shape_id'] for e in entries], dtype=np.int64),
        }

    def get_all(self) -> dict:
        """Return all entries as stacked arrays. For evaluation."""
        with self._lock:
            if not self._buffer:
                return None
            entries = list(self._buffer)

        return {
            'features': np.stack([e['features'] for e in entries]),
            'log_correction': np.array([e['log_correction'] for e in entries], dtype=np.float32),
            'shape_ids': np.array([e['shape_id'] for e in entries], dtype=np.int64),
            'gflops': np.array([e['gflops'] for e in entries], dtype=np.float32),
            'us': np.array([e['us'] for e in entries], dtype=np.float32),
            'M': np.array([e['M'] for e in entries], dtype=np.int64),
            'N': np.array([e['N'] for e in entries], dtype=np.int64),
            'K': np.array([e['K'] for e in entries], dtype=np.int64),
        }

    def get_shapes_data(self) -> dict[int, list[dict]]:
        """Return entries grouped by shape_id. For per-shape evaluation."""
        with self._lock:
            entries = list(self._buffer)

        shapes = {}
        for e in entries:
            sid = e['shape_id']
            if sid not in shapes:
                shapes[sid] = []
            shapes[sid].append(e)
        return shapes

    def save_parquet(self, path: Path):
        """Persist buffer contents to a parquet file."""
        import pandas as pd

        with self._lock:
            if not self._buffer:
                log.warning("Empty buffer, nothing to save")
                return
            entries = list(self._buffer)

        # Flatten features into columns
        n_features = len(entries[0]['features'])
        rows = []
        for e in entries:
            row = {
                'shape_id': e['shape_id'],
                'M': e['M'], 'N': e['N'], 'K': e['K'],
                'kernel_sig': e['kernel_sig'],
                'log_correction': e['log_correction'],
                'gflops': e['gflops'],
                'us': e['us'],
            }
            for i in range(n_features):
                row[f'f_{i}'] = e['features'][i]
            rows.append(row)

        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Saved %d entries (%d shapes) to %s", len(entries), self._shape_count, path)

    def load_parquet(self, path: Path) -> int:
        """Load buffer contents from a parquet file. Returns number of entries loaded."""
        import pandas as pd

        if not path.exists():
            log.info("No existing dataset at %s", path)
            return 0

        df = pd.read_parquet(path)
        feature_cols = sorted([c for c in df.columns if c.startswith('f_')],
                              key=lambda x: int(x[2:]))
        n_loaded = 0

        with self._lock:
            existing_shapes = set()
            for e in self._buffer:
                existing_shapes.add(e['shape_id'])

            for _, row in df.iterrows():
                features = np.array([row[c] for c in feature_cols], dtype=np.float32)
                entry = {
                    'shape_id': int(row['shape_id']),
                    'M': int(row['M']),
                    'N': int(row['N']),
                    'K': int(row['K']),
                    'kernel_sig': str(row['kernel_sig']),
                    'features': features,
                    'log_correction': float(row['log_correction']),
                    'gflops': float(row['gflops']),
                    'us': float(row['us']),
                }
                self._buffer.append(entry)
                existing_shapes.add(entry['shape_id'])
                n_loaded += 1

            self._shape_count = len(existing_shapes)

        log.info("Loaded %d entries (%d shapes) from %s", n_loaded, self._shape_count, path)
        return n_loaded
