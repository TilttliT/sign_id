import pytest
import cv2
import numpy as np
from pathlib import Path

from signature_model.inference import SignatureVerifier

MODEL_PATH = Path("checkpoints/signature_model.pth")


def split_image_into_grid(image: np.ndarray, rows: int = 2, cols: int = 6) -> list[list[np.ndarray]]:
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            cell = image[y1:y2, x1:x2]
            row_cells.append(cell)
        grid.append(row_cells)
    return grid


@pytest.fixture(scope="module")
def verifier():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model checkpoint not found at {MODEL_PATH}")
    return SignatureVerifier(str(MODEL_PATH))


def test_writing_grid_similarity(verifier, tmp_path):
    image_path = Path("tests/input/test_signature_model/test.jpg")
    if not image_path.exists():
        pytest.skip(f"Test image not found at {image_path}")

    img = cv2.imread(str(image_path))
    assert img is not None, "Failed to load test image"

    grid = split_image_into_grid(img, rows=2, cols=6)

    cell_files = []
    for r in range(2):
        for c in range(6):
            cell_path = tmp_path / f"cell_{r}_{c}.jpg"
            cv2.imwrite(str(cell_path), grid[r][c])
            cell_files.append(cell_path)

    correct_same = 0
    false_rejects = 0
    false_accepts = 0

    for c1 in range(6):
        idx1 = c1
        for c2 in range(6):
            idx2 = 6 + c2
            result, sim = verifier.verify(str(cell_files[idx1]), str(cell_files[idx2]))
            is_same_word = c1 == c2

            if is_same_word:
                if result:
                    correct_same += 1
                else:
                    false_rejects += 1
            else:
                if result:
                    false_accepts += 1

    tpr = correct_same / 6
    print(f"\nTPR (correct same-word pairs): {tpr:.2%} ({correct_same}/6)")
    print(f"False rejects: {false_rejects}")
    print(f"False accepts (different words): {false_accepts}")

    assert false_rejects / 6 <= 0.2  # false_rejects <= 1
    assert false_accepts / 30 <= 0.05  # false_accepts <= 1

    for c in range(6):
        _, same_sim = verifier.verify(str(cell_files[c]), str(cell_files[6 + c]))
        other_sims = []
        for other in range(6):
            if other == c:
                continue
            _, sim = verifier.verify(str(cell_files[c]), str(cell_files[6 + other]))
            other_sims.append(sim)
        if other_sims:
            max_other = max(other_sims)
            assert same_sim > max_other, (
                f"Word {c}: similarity to itself ({same_sim:.3f}) not greater than max other ({max_other:.3f})"
            )
