"""
Lazy combined-weight store.

Instead of eagerly computing ~1300 combined-weight matrices (each N×N),
this class stores only the *recipes* – which single-weight DataKeys to
combine via element-wise maximum – and evaluates them on access.

Typical savings: 2–5 GB peak RAM for a 1 000-zone model.
CPU overhead: negligible – sparse_maximum is O(nnz), far cheaper than the
downstream matrix–vector products that dominate runtime.
"""

import logging
import sys
import threading

import numpy as np

from ikob.datasource import DataKey, DataSource, DataType
from ikob.utils import sparse_maximum

logger = logging.getLogger(__name__)


class LazyCombinedDataSource:
    """Drop-in replacement for ``DataSource`` used by combined weights."""

    def __init__(self, config, single_weights: DataSource):
        self._config = config
        self._single = single_weights
        self._recipes: dict[DataKey, list[DataKey]] = {}
        self._cache: dict[DataKey, np.ndarray] = {}

    # ── public API (compatible with DataSource) ──────────────────────

    def register(self, key: DataKey, component_keys: list[DataKey]):
        """Store a recipe:  result = sparse_maximum(component_0, component_1, …)."""
        self._recipes[key] = component_keys

    def get(self, key: DataKey):
        if key in self._cache:
            return self._cache[key]
        if key in self._recipes:
            parts = [self._single.get(k) for k in self._recipes[key]]
            result = parts[0]
            for p in parts[1:]:
                result = sparse_maximum(result, p)
            # intentionally NOT cached – keeps memory flat
            return result
        raise KeyError(f"Combined weight not registered: {key}")

    def set(self, key: DataKey, data):
        self._cache[key] = data

    @property
    def cache(self):
        return self._cache

    def clear_cache(self):
        n = len(self._cache)
        self._cache.clear()
        logger.debug("Cleared %d lazy-combined cache entries.", n)

    def cache_size_mb(self) -> float:
        total = 0
        for v in self._cache.values():
            try:
                total += v.nbytes
            except AttributeError:
                try:
                    from scipy import sparse
                    if sparse.issparse(v):
                        total += v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
                except (ImportError, AttributeError):
                    total += sys.getsizeof(v)
        return total / (1024 * 1024)

    def recipe_count(self) -> int:
        return len(self._recipes)

    def store(self):
        """Materialise every recipe one-by-one and write to disk.

        Only needed when ``write_weights=True`` (test mode).
        Each matrix is written and immediately discarded so peak memory
        stays bounded to one N×N matrix at a time.
        """
        logger.info("Materialising %d combined-weight recipes to disk…", len(self._recipes))
        writer = DataSource(self._config, DataType.WEIGHTS)
        for key in self._recipes:
            data = self.get(key)          # compute on the fly
            writer.write_csv(data, key)   # write immediately
            # data goes out of scope → freed
        # also write anything that was .set() directly
        for key, data in self._cache.items():
            if key not in self._recipes:
                writer.write_csv(data, key)
        logger.info("Combined weights written.")