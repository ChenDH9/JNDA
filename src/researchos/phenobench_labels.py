"""Original PhenoBench three-class label mapping."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from PIL import Image
RAW_TO_TASK_LABEL = {0: 0, 1: 1, 2: 2, 3: 1, 4: 2}
TASK_PALETTE = ((0, 0, 0), (34, 139, 34), (255, 0, 255))

def task_label_array(
    image: Image.Image | np.ndarray,
    *,
    raw_to_task: Mapping[int, int] = RAW_TO_TASK_LABEL,
    palette: Sequence[Sequence[int]] = TASK_PALETTE,
) -> np.ndarray:
    """Decode either indexed PhenoBench semantics or the frozen RGB task palette."""

    array = np.asarray(image)
    if array.ndim == 2:
        observed = {int(value) for value in np.unique(array)}
        if not observed <= set(raw_to_task):
            raise ValueError(f"unexpected raw semantic IDs: {sorted(observed)}")
        lookup_size = max(raw_to_task) + 1
        lookup = np.full(lookup_size, 255, dtype=np.uint8)
        for raw, task in raw_to_task.items():
            lookup[int(raw)] = int(task)
        result = lookup[array.astype(np.int64, copy=False)]
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[:, :, :3].astype(np.uint8, copy=False)
        result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for task, color in enumerate(palette):
            result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)] = task
    else:
        raise ValueError(f"unsupported semantic target shape: {array.shape}")
    invalid = np.unique(result[result > 2])
    if invalid.size:
        raise ValueError("semantic target contains pixels outside the frozen three-class mapping")
    return result
