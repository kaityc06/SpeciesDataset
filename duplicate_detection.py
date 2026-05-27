import numpy as np
from PIL import Image

# HOG parameters — 128×128 → 8×8 cells → 7×7 blocks → 1764-dim feature vector.
# Memory: ~7 KB per image. Suitable for up to ~100 K images in RAM.
# Dependencies: numpy and PIL only (no scikit-image required).
_RESIZE = (128, 128)
_PIXELS_PER_CELL = 16    # cell size in pixels (square)
_CELLS_PER_BLOCK = 2     # block size in cells (square)
_ORIENTATIONS = 9        # number of gradient orientation bins


def compute_hog(pil_image: Image.Image) -> np.ndarray:
    """Return a unit-normalised HOG feature vector for *pil_image*.

    Implemented with numpy only — no scikit-image required.
    """
    img = np.array(pil_image.convert("L").resize(_RESIZE, Image.BILINEAR), dtype=np.float32) / 255.0

    # Gradients via simple central differences
    gx = np.empty_like(img)
    gy = np.empty_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gx[:, 0]    = img[:, 1] - img[:, 0]
    gx[:, -1]   = img[:, -1] - img[:, -2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gy[0, :]    = img[1, :] - img[0, :]
    gy[-1, :]   = img[-1, :] - img[-2, :]

    magnitude = np.hypot(gx, gy)
    # unsigned orientation in [0, π)
    orientation = (np.arctan2(np.abs(gy), np.abs(gx))) % np.pi

    H, W = img.shape
    n_cells_y = H // _PIXELS_PER_CELL
    n_cells_x = W // _PIXELS_PER_CELL

    # Build per-cell gradient histograms
    cell_hist = np.zeros((n_cells_y, n_cells_x, _ORIENTATIONS), dtype=np.float32)
    bin_width = np.pi / _ORIENTATIONS
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            r0, r1 = cy * _PIXELS_PER_CELL, (cy + 1) * _PIXELS_PER_CELL
            c0, c1 = cx * _PIXELS_PER_CELL, (cx + 1) * _PIXELS_PER_CELL
            mag_patch = magnitude[r0:r1, c0:c1].ravel()
            ori_patch = orientation[r0:r1, c0:c1].ravel()
            bins = (ori_patch / bin_width).astype(int).clip(0, _ORIENTATIONS - 1)
            np.add.at(cell_hist[cy, cx], bins, mag_patch)

    # Block normalisation (L2-Hys: clamp to 0.2, renormalise)
    n_blocks_y = n_cells_y - _CELLS_PER_BLOCK + 1
    n_blocks_x = n_cells_x - _CELLS_PER_BLOCK + 1
    block_features = np.empty(
        (n_blocks_y, n_blocks_x, _CELLS_PER_BLOCK, _CELLS_PER_BLOCK, _ORIENTATIONS),
        dtype=np.float32,
    )
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hist[by:by + _CELLS_PER_BLOCK, bx:bx + _CELLS_PER_BLOCK, :]
            norm = np.sqrt(np.sum(block ** 2) + 1e-6)
            block = block / norm
            block = np.clip(block, 0.0, 0.2)
            norm2 = np.sqrt(np.sum(block ** 2) + 1e-6)
            block_features[by, bx] = block / norm2

    features = block_features.ravel()
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features


class DuplicateDetector:
    """Track seen images via HOG cosine similarity to detect near-duplicates.

    Typical usage inside a processing loop::

        detector = DuplicateDetector(threshold=0.97)
        for image in stream:
            if detector.is_duplicate_and_add(image):
                continue          # skip duplicate
            process(image)        # only unique images reach here

    threshold:  cosine-similarity cutoff.  Values ≥ threshold are considered
                duplicates.  0.98 catches near-identical images; 0.95 catches
                moderate re-encodes / small crops.  Default 0.97 is a
                conservative middle ground for wildlife imagery.
    """

    def __init__(self, threshold: float = 0.97):
        self.threshold = threshold
        self._features: list[np.ndarray] = []

    @property
    def num_seen(self) -> int:
        return len(self._features)

    def is_duplicate(self, pil_image: Image.Image) -> bool:
        """Return True if *pil_image* is near-identical to any previously seen image."""
        if not self._features:
            return False
        query = compute_hog(pil_image)
        seen = np.stack(self._features, axis=0)   # (N, D)
        sims = seen @ query                        # cosine similarity, features are unit vectors
        return float(sims.max()) >= self.threshold

    def add(self, pil_image: Image.Image) -> None:
        """Register *pil_image* as seen (call only after is_duplicate returns False)."""
        self._features.append(compute_hog(pil_image))

    def is_duplicate_and_add(self, pil_image: Image.Image) -> bool:
        """Check for a duplicate; if not one, register and return False.

        Computes the HOG vector once.  Returns True if the image is a duplicate
        (image is NOT added); False if it is new (image IS added).
        """
        is_dup, _ = self.check_and_add(pil_image)
        return is_dup

    def check_and_add(self, pil_image: Image.Image) -> tuple:
        """Like is_duplicate_and_add but also returns the max cosine similarity.

        Returns (is_duplicate, max_similarity).  max_similarity is 0.0 when no
        images have been seen yet.  If is_duplicate is False the image is added.
        """
        query = compute_hog(pil_image)
        max_sim = 0.0
        if self._features:
            seen = np.stack(self._features, axis=0)
            max_sim = float((seen @ query).max())
            if max_sim >= self.threshold:
                return True, max_sim
        self._features.append(query)
        return False, max_sim
