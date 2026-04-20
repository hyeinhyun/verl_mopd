"""Stratified sampler for multi-teacher distillation with hierarchical proportions.

Samples each batch so that data sources appear in specified ratios.
Supports nested grouping: define teacher-level splits, then per-dataset splits within.

Example config (via CLI overrides):
    data.sampler.class_path=verl.experimental.dataset.stratified_sampler.StratifiedSampler
    +data.sampler.init_args.proportions.openai/gsm8k=0.25
    +data.sampler.init_args.proportions.math500=0.25
    +data.sampler.init_args.proportions.ifeval=0.30
    +data.sampler.init_args.proportions.nemotroncc=0.20

Proportions must sum to 1.0. If a source has fewer samples than its quota in a batch,
remaining slots are filled from other sources proportionally (best-effort).
"""

import logging
import os
from collections import defaultdict
from collections.abc import Iterator, Sized

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class StratifiedSampler(AbstractSampler):
    """Sampler that produces batches with fixed proportions per data_source.

    Args:
        data_source: The dataset (must have a `data_source` column or attribute).
        data_config: verl data config (contains train_batch_size, seed, etc.).
        proportions: dict mapping data_source values to their target fraction of each batch.
            E.g. {"openai/gsm8k": 0.25, "math500": 0.25, "ifeval": 0.30, "nemotroncc": 0.20}
            Must sum to 1.0 (tolerance 1e-3).
        data_source_column: Column name to read routing values from (default: "data_source").
    """

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
        proportions: dict[str, float] = None,
        data_source_column: str = "data_source",
    ):
        self.dataset = data_source
        self.data_config = data_config
        self.batch_size = data_config.train_batch_size
        self.seed = data_config.get("seed", 42)
        self.data_source_column = data_source_column

        if proportions is None:
            raise ValueError(
                "StratifiedSampler requires `proportions` dict mapping data_source values to fractions. "
                "E.g. proportions={'gsm8k': 0.5, 'ifeval': 0.5}"
            )
        self.proportions = dict(proportions)
        total = sum(self.proportions.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"proportions must sum to 1.0, got {total:.4f}: {self.proportions}")

        self._build_index()

    def _build_index(self):
        """Group dataset indices by their data_source value."""
        self.source_indices: dict[str, list[int]] = defaultdict(list)

        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if isinstance(item, dict):
                source = item.get(self.data_source_column)
            elif hasattr(item, self.data_source_column):
                source = getattr(item, self.data_source_column)
            elif hasattr(item, "non_tensor_batch") and self.data_source_column in item.non_tensor_batch:
                source = item.non_tensor_batch[self.data_source_column]
            else:
                raise ValueError(
                    f"Cannot find '{self.data_source_column}' in dataset item at index {i}. "
                    f"Ensure your dataset returns items with a '{self.data_source_column}' field."
                )
            if source not in self.proportions:
                raise ValueError(
                    f"Dataset contains data_source={source!r} at index {i}, but it's not in "
                    f"proportions={list(self.proportions.keys())}. Either add it to proportions "
                    f"or filter it from the dataset."
                )
            self.source_indices[source].append(i)

        for source, indices in self.source_indices.items():
            logger.info(f"StratifiedSampler: {source} has {len(indices)} samples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)

        # Shuffle indices within each source
        shuffled: dict[str, np.ndarray] = {}
        for source, indices in self.source_indices.items():
            arr = np.array(indices)
            rng.shuffle(arr)
            shuffled[source] = arr

        # Track position within each source's shuffled array
        positions = {source: 0 for source in shuffled}
        total_samples = len(self.dataset)
        yielded = 0

        while yielded < total_samples:
            batch_indices = []
            remaining_batch = min(self.batch_size, total_samples - yielded)

            for source, fraction in self.proportions.items():
                count = int(round(remaining_batch * fraction))
                arr = shuffled[source]
                pos = positions[source]
                available = len(arr) - pos

                # Take min(count, available) from this source
                take = min(count, available)
                if take > 0:
                    batch_indices.extend(arr[pos : pos + take].tolist())
                    positions[source] = pos + take

            # If some sources exhausted, fill remainder from any source that has leftovers
            if len(batch_indices) < remaining_batch:
                for source in shuffled:
                    if len(batch_indices) >= remaining_batch:
                        break
                    arr = shuffled[source]
                    pos = positions[source]
                    available = len(arr) - pos
                    take = min(remaining_batch - len(batch_indices), available)
                    if take > 0:
                        batch_indices.extend(arr[pos : pos + take].tolist())
                        positions[source] = pos + take

            # Shuffle within the batch to avoid source clustering
            rng.shuffle(batch_indices)
            for idx in batch_indices:
                yield idx
                yielded += 1

        # Bump seed for next epoch
        self.seed += 1
