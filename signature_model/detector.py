from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any, Tuple


def detect_signature(
    image: np.ndarray, config: Dict[str, Any], debug: bool = False
) -> Tuple[int, int, int, int] | None:
    h, w = image.shape[:2]
    total_area = h * w
    min_dim = min(h, w)

    block_size = int(round(config["threshold_block_size_ratio"] * min_dim))
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    kernel_size = int(round(config["morph_kernel_ratio"] * min_dim))
    kernel_size = max(1, kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    morph_iterations = config.get("morph_iterations", 1)

    padding = int(round(config.get("padding_ratio", 0.0) * min_dim))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, config["threshold_c"]
    )

    if debug:
        cv2.imshow("1. Binary", thresh)

    dilated = cv2.dilate(thresh, kernel, iterations=morph_iterations)

    if debug:
        cv2.imshow("2. Dilated", dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    if debug:
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imshow("3. Contours", contour_img)

    min_area = config["min_area_ratio"] * total_area
    max_area = config["max_area_ratio"] * total_area

    valid_contours = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if min_area <= area <= max_area:
            valid_contours.append(cnt)

    if not valid_contours:
        return None

    combined_mask = np.zeros_like(thresh)
    cv2.drawContours(combined_mask, valid_contours, -1, 255, -1)
    x, y, bw, bh = cv2.boundingRect(combined_mask)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)
    final_box = (x1, y1, x2 - x1, y2 - y1)

    if debug:
        result_img = image.copy()
        cv2.rectangle(result_img, (x1, y1), (x1 + final_box[2], y1 + final_box[3]), (0, 0, 255), 2)
        cv2.imshow("4. Final Box", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_box
