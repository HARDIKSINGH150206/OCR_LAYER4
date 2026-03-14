from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessOutput:
    original: np.ndarray
    grayscale: np.ndarray
    clahe: np.ndarray
    thresholded: np.ndarray
    denoised: np.ndarray


class OCRPreprocessor:
    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )

    def preprocess(self, image: np.ndarray) -> PreprocessOutput:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = self._clahe.apply(grayscale)
        thresholded = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        denoised = cv2.fastNlMeansDenoising(thresholded, None, 10, 7, 21)
        return PreprocessOutput(
            original=image,
            grayscale=grayscale,
            clahe=clahe,
            thresholded=thresholded,
            denoised=denoised,
        )

    @staticmethod
    def crop_regions(
        image: np.ndarray,
        boxes: Sequence[BBox],
        min_size: int = 8,
        pad_ratio: float = 0.03,
    ) -> List[Tuple[BBox, np.ndarray]]:
        height, width = image.shape[:2]
        crops: List[Tuple[BBox, np.ndarray]] = []

        for x1, y1, x2, y2 in boxes:
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < min_size or box_h < min_size:
                continue

            pad_x = int(box_w * pad_ratio)
            pad_y = int(box_h * pad_ratio)
            nx1 = max(0, x1 - pad_x)
            ny1 = max(0, y1 - pad_y)
            nx2 = min(width, x2 + pad_x)
            ny2 = min(height, y2 + pad_y)

            crop = image[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue
            crops.append(((nx1, ny1, nx2, ny2), crop))

        return crops
