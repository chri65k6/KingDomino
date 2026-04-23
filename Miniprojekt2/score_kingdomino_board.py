import argparse
import json
import pickle
import sys
import types
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS_PKL = BASE_DIR / "kingdomino_params.pkl"
DEFAULT_MODEL_PKL = BASE_DIR / "king_domino_final_model.pkl"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "Templates"
DEFAULT_IMAGE = BASE_DIR / "King Domino dataset" / "34.jpg"

TERRAIN_CLASSES = {"field", "forest", "grass", "mine", "swamp", "water"}
CLUSTER_COLORS = [
    (64, 180, 255),
    (86, 204, 113),
    (255, 171, 64),
    (168, 102, 255),
    (255, 91, 91),
    (80, 220, 220),
    (190, 190, 80),
    (220, 120, 190),
    (120, 180, 120),
    (100, 150, 255),
]


class MultiFeatureWeighter(BaseEstimator, TransformerMixin):
    """Compatibility transformer needed when loading king_domino_final_model.pkl."""

    def __init__(
        self,
        hsv_cols=None,
        sift_cols=None,
        hog_cols=None,
        hsv_weight=1.0,
        sift_weight=1.0,
        hog_weight=1.0,
    ):
        self.hsv_cols = hsv_cols
        self.sift_cols = sift_cols
        self.hog_cols = hog_cols
        self.hsv_weight = hsv_weight
        self.sift_weight = sift_weight
        self.hog_weight = hog_weight

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.hsv_cols_ = self.hsv_cols or ["hue", "saturation", "value"]
        self.sift_cols_ = self.sift_cols or [
            column for column in X.columns if column == "num_sift_keypoints" or column.startswith("sift_")
        ]
        self.hog_cols_ = self.hog_cols or [column for column in X.columns if column.startswith("hog_")]

        self.hsv_scaler_ = StandardScaler().fit(X[self.hsv_cols_])
        self.sift_scaler_ = StandardScaler().fit(X[self.sift_cols_])
        self.hog_scaler_ = StandardScaler().fit(X[self.hog_cols_])
        return self

    def transform(self, X):
        X = self._as_frame(X)
        hsv = self.hsv_scaler_.transform(X[self.hsv_cols_]) * self.hsv_weight
        sift = self.sift_scaler_.transform(X[self.sift_cols_]) * self.sift_weight
        hog = self.hog_scaler_.transform(X[self.hog_cols_]) * self.hog_weight
        return np.hstack([hsv, sift, hog])

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


feature_pipeline_module = types.ModuleType("kingdomino_feature_pipeline")
feature_pipeline_module.MultiFeatureWeighter = MultiFeatureWeighter
sys.modules.setdefault("kingdomino_feature_pipeline", feature_pipeline_module)


def load_pickle(path):
    with resolve_path(path).open("rb") as file:
        return pickle.load(file)


def resolve_path(path):
    path = Path(path)
    if path.is_file():
        return path

    project_path = BASE_DIR / path
    if project_path.is_file():
        return project_path

    return path


def parse_tile_size(tile_size):
    if isinstance(tile_size, (tuple, list)):
        return int(tile_size[0]), int(tile_size[1])
    return int(tile_size), int(tile_size)


def read_board_image(image_path, board_size, tile_size):
    image_path = resolve_path(image_path)
    image = cv.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    tile_width, tile_height = parse_tile_size(tile_size)
    target_width = board_size * tile_width
    target_height = board_size * tile_height

    if image.shape[1] != target_width or image.shape[0] != target_height:
        interpolation = cv.INTER_AREA if image.shape[1] > target_width else cv.INTER_LINEAR
        image = cv.resize(image, (target_width, target_height), interpolation=interpolation)

    return image


def get_tile(image, row, col, tile_width, tile_height, buffer=0, border_mode=cv.BORDER_REPLICATE):
    image_height, image_width = image.shape[:2]
    tile_x1 = col * tile_width
    tile_y1 = row * tile_height
    tile_x2 = (col + 1) * tile_width
    tile_y2 = (row + 1) * tile_height

    desired_x1 = tile_x1 - buffer
    desired_y1 = tile_y1 - buffer
    desired_x2 = tile_x2 + buffer
    desired_y2 = tile_y2 + buffer

    crop_x1 = max(0, desired_x1)
    crop_y1 = max(0, desired_y1)
    crop_x2 = min(image_width, desired_x2)
    crop_y2 = min(image_height, desired_y2)

    tile = image[crop_y1:crop_y2, crop_x1:crop_x2]

    pad_left = crop_x1 - desired_x1
    pad_top = crop_y1 - desired_y1
    pad_right = desired_x2 - crop_x2
    pad_bottom = desired_y2 - crop_y2

    if pad_left or pad_top or pad_right or pad_bottom:
        tile = cv.copyMakeBorder(tile, pad_top, pad_bottom, pad_left, pad_right, border_mode)

    count_area = (
        tile_x1 - desired_x1,
        tile_y1 - desired_y1,
        tile_x2 - desired_x1,
        tile_y2 - desired_y1,
    )
    return tile, count_area


def split_board(image, board_size, tile_size):
    tile_width, tile_height = parse_tile_size(tile_size)
    return [
        [get_tile(image, row, col, tile_width, tile_height)[0] for col in range(board_size)]
        for row in range(board_size)
    ]


def create_color_histogram(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv_image], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
    return hist


def compare_color_histograms(image, template_hist):
    image_hist = create_color_histogram(image)
    score = cv.compareHist(image_hist, template_hist, cv.HISTCMP_CORREL)

    if np.isnan(score):
        return 0.0

    return float(np.clip((score + 1) / 2, 0, 1))


def get_match_scales(params):
    if "MATCH_SCALES" in params:
        return np.asarray(params["MATCH_SCALES"], dtype=float)

    start = float(params.get("MATCH_SCALE_START", 0.85))
    end = float(params.get("MATCH_SCALE_END", 1.15))
    count = int(params.get("MATCH_SCALE_COUNT", 50))
    return np.linspace(start, end, count)


def load_crown_templates(params):
    template_dir = Path(params.get("TEMPLATE_DIR", DEFAULT_TEMPLATES_DIR))
    if not template_dir.is_dir():
        template_dir = DEFAULT_TEMPLATES_DIR

    templates = []
    for path in sorted(template_dir.glob("*.png")):
        template = cv.imread(str(path))
        if template is None:
            continue

        for scale in get_match_scales(params):
            resized = cv.resize(template, None, fx=float(scale), fy=float(scale))
            height, width = resized.shape[:2]

            if height == 0 or width == 0:
                continue

            templates.append(
                {
                    "filename": path.name,
                    "blur": cv.GaussianBlur(resized, (3, 3), 0),
                    "hist": create_color_histogram(resized),
                    "w": width,
                    "h": height,
                }
            )

    if not templates:
        raise FileNotFoundError(f"No crown templates found in {template_dir}")

    return templates


def is_match_inside_count_area(x, y, width, height, count_area):
    area_x1, area_y1, area_x2, area_y2 = count_area
    center_x = x + width / 2
    center_y = y + height / 2
    return area_x1 <= center_x < area_x2 and area_y1 <= center_y < area_y2


def count_crowns(tile, count_area, templates, params):
    threshold = float(params.get("TEMPLATE_MATCH_THRESHOLD", 0.78))
    color_threshold = float(params.get("COLOR_HIST_THRESHOLD", 0.60))
    template_weight = float(params.get("TEMPLATE_SCORE_WEIGHT", 0.75))
    color_weight = float(params.get("COLOR_SCORE_WEIGHT", 0.75))

    tile_blur = cv.GaussianBlur(tile, (3, 3), 0)
    matches = []

    for template in templates:
        height = template["h"]
        width = template["w"]

        if height > tile_blur.shape[0] or width > tile_blur.shape[1]:
            continue

        result = cv.matchTemplate(tile_blur, template["blur"], cv.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for point in zip(*locations[::-1]):
            x, y = point
            if not is_match_inside_count_area(x, y, width, height, count_area):
                continue

            candidate = tile[y : y + height, x : x + width]
            color_score = compare_color_histograms(candidate, template["hist"])

            if color_score < color_threshold:
                continue

            template_score = float(result[y, x])
            combined_score = (template_score * template_weight) + (color_score * color_weight)
            matches.append((x, y, width, height, combined_score))

    matches = sorted(matches, key=lambda match: match[4], reverse=True)
    filtered = []

    for match in matches:
        x, y, width, height, _ = match
        too_close = any(abs(x - fx) < width * 0.5 and abs(y - fy) < height * 0.5 for fx, fy, *_ in filtered)

        if not too_close:
            filtered.append(match)

    return len(filtered)


def count_board_crowns(image, board_size, tile_size, params):
    tile_width, tile_height = parse_tile_size(tile_size)
    buffer = int(params.get("CROWN_SEARCH_BUFFER", 10))
    border_mode = int(params.get("EDGE_PADDING_MODE", cv.BORDER_REPLICATE))
    templates = load_crown_templates(params)
    crown_grid = []

    for row in range(board_size):
        crown_row = []
        for col in range(board_size):
            tile, count_area = get_tile(image, row, col, tile_width, tile_height, buffer, border_mode)
            crown_row.append(count_crowns(tile, count_area, templates, params))
        crown_grid.append(crown_row)

    return crown_grid


def extract_hsv_features(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.mean(hsv_tile, axis=(0, 1))
    return {
        "hue": float(hue),
        "saturation": float(saturation),
        "value": float(value),
    }


def extract_sift_features(tile, kmeans, feature_params):
    max_descriptors = int(feature_params.get("sift_max_descriptors_per_image", 50))
    vocab_size = int(feature_params.get("sift_vocab_size", kmeans.n_clusters))

    if not hasattr(cv, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is not available in this Python environment.")

    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=max_descriptors)
    _keypoints, descriptors = sift.detectAndCompute(gray, None)

    histogram = np.zeros(vocab_size, dtype=float)
    num_descriptors = 0

    if descriptors is not None and len(descriptors) > 0:
        descriptors = descriptors[:max_descriptors].astype(np.float32)
        num_descriptors = len(descriptors)
        labels = kmeans.predict(descriptors)
        histogram = np.bincount(labels, minlength=vocab_size).astype(float)
        histogram /= max(num_descriptors, 1)

    features = {"num_sift_keypoints": float(num_descriptors)}
    features.update({f"sift_{index}": float(value) for index, value in enumerate(histogram)})
    return features


def extract_hog_features(tile, feature_params, expected_length):
    orientations = int(feature_params.get("hog_orientations", 8))
    pixels_per_cell = tuple(feature_params.get("hog_pixels_per_cell", (30, 30)))
    cells_per_block = tuple(feature_params.get("hog_cells_per_block", (2, 2)))

    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cell_height, cell_width = int(pixels_per_cell[0]), int(pixels_per_cell[1])
    block_height, block_width = int(cells_per_block[0]), int(cells_per_block[1])

    cells_y = gray.shape[0] // cell_height
    cells_x = gray.shape[1] // cell_width

    if cells_y < block_height or cells_x < block_width:
        return np.zeros(expected_length, dtype=float)

    gray = gray[: cells_y * cell_height, : cells_x * cell_width]
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    grad_y[1:-1, :] = gray[2:, :] - gray[:-2, :]

    magnitude = np.hypot(grad_x, grad_y)
    angle = np.rad2deg(np.arctan2(grad_y, grad_x)) % 180.0
    bin_width = 180.0 / orientations
    bin_index = np.floor(angle / bin_width).astype(int)
    bin_index = np.clip(bin_index, 0, orientations - 1)

    cell_histograms = np.zeros((cells_y, cells_x, orientations), dtype=float)

    for cell_y in range(cells_y):
        for cell_x in range(cells_x):
            y1 = cell_y * cell_height
            y2 = y1 + cell_height
            x1 = cell_x * cell_width
            x2 = x1 + cell_width
            cell_histograms[cell_y, cell_x] = np.bincount(
                bin_index[y1:y2, x1:x2].ravel(),
                weights=magnitude[y1:y2, x1:x2].ravel(),
                minlength=orientations,
            )

    features = []
    eps = 1e-5

    for block_y in range(cells_y - block_height + 1):
        for block_x in range(cells_x - block_width + 1):
            block = cell_histograms[
                block_y : block_y + block_height,
                block_x : block_x + block_width,
            ].ravel()
            block = block / np.sqrt(np.sum(block**2) + eps**2)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.sum(block**2) + eps**2)
            features.extend(block)

    features = np.asarray(features, dtype=float)

    if len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)))
    elif len(features) > expected_length:
        features = features[:expected_length]

    return features


def extract_tile_features(tile, row, col, image_path, bundle):
    feature_params = bundle.get("feature_params", {})
    hog_cols = bundle["hog_cols"]

    features = {
        "image_id": Path(image_path).stem,
        "row": row,
        "col": col,
        "tile_image_path": f"{image_path}#{row},{col}",
        "img_width": int(tile.shape[1]),
        "img_height": int(tile.shape[0]),
        "csv_file": "",
    }
    features.update(extract_hsv_features(tile))
    features.update(extract_sift_features(tile, bundle["sift_kmeans"], feature_params))

    hog_values = extract_hog_features(tile, feature_params, len(hog_cols))
    features.update({column: float(value) for column, value in zip(hog_cols, hog_values)})
    return features


def classify_terrains(image, image_path, bundle):
    board_size = int(bundle.get("board_size", 5))
    tile_size = bundle.get("tile_size", (100, 100))
    tiles = split_board(image, board_size, tile_size)

    rows = []
    for row in range(board_size):
        for col in range(board_size):
            rows.append(extract_tile_features(tiles[row][col], row, col, image_path, bundle))

    feature_frame = pd.DataFrame(rows)
    feature_columns = bundle["feature_columns"]
    encoded_predictions = bundle["model"].predict(feature_frame[feature_columns])
    terrain_predictions = bundle["label_encoder"].inverse_transform(encoded_predictions)

    terrain_grid = []
    index = 0
    for _row in range(board_size):
        terrain_row = []
        for _col in range(board_size):
            terrain_row.append(str(terrain_predictions[index]))
            index += 1
        terrain_grid.append(terrain_row)

    return terrain_grid


def score_board(terrain_grid, crown_grid):
    board_size = len(terrain_grid)
    visited = [[False for _ in range(board_size)] for _ in range(board_size)]
    components = []
    total_points = 0

    for row in range(board_size):
        for col in range(board_size):
            if visited[row][col]:
                continue

            terrain = terrain_grid[row][col]
            queue = deque([(row, col)])
            visited[row][col] = True
            cells = []
            crowns = 0

            while queue:
                current_row, current_col = queue.popleft()
                cells.append([current_row, current_col])
                crowns += int(crown_grid[current_row][current_col])

                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                        continue
                    if visited[next_row][next_col] or terrain_grid[next_row][next_col] != terrain:
                        continue

                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))

            size = len(cells)
            points = size * crowns if terrain in TERRAIN_CLASSES else 0
            total_points += points
            components.append(
                {
                    "terrain": terrain,
                    "cells": cells,
                    "size": size,
                    "crowns": crowns,
                    "points": points,
                }
            )

    return total_points, components


def apply_terrain_corrections(terrain_grid, crown_grid):
    corrected_grid = [row[:] for row in terrain_grid]
    board_size = len(terrain_grid)
    visited = [[False for _ in range(board_size)] for _ in range(board_size)]
    corrections = []

    for row in range(board_size):
        for col in range(board_size):
            if visited[row][col] or terrain_grid[row][col] != "forest":
                continue

            queue = deque([(row, col)])
            visited[row][col] = True
            cells = []
            max_crowns = 0

            while queue:
                current_row, current_col = queue.popleft()
                cells.append([current_row, current_col])
                max_crowns = max(max_crowns, int(crown_grid[current_row][current_col]))

                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                        continue
                    if visited[next_row][next_col] or terrain_grid[next_row][next_col] != "forest":
                        continue

                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))

            if max_crowns < 2:
                continue

            for cell_row, cell_col in cells:
                corrected_grid[cell_row][cell_col] = "mine"

            corrections.append(
                {
                    "from": "forest",
                    "to": "mine",
                    "reason": "forest cluster contains a tile with 2 or more crowns",
                    "cells": cells,
                }
            )

    return corrected_grid, corrections


def format_grid(grid, width=8):
    return "\n".join("  " + " ".join(str(value).ljust(width) for value in row) for row in grid)


def print_result(image_path, terrain_grid, crown_grid, components, total_points, corrections=None):
    print(f"Image: {image_path}")
    print("\nTerrain grid:")
    print(format_grid(terrain_grid))
    print("\nCrown grid:")
    print(format_grid(crown_grid, width=3))

    if corrections:
        print("\nTerrain corrections:")
        for correction in corrections:
            cells = ", ".join(f"({row},{col})" for row, col in correction["cells"])
            print(f"  {correction['from']} -> {correction['to']}: {cells}")

    print("\nScoring components:")

    for component in components:
        if component["points"] == 0:
            continue
        cells = ", ".join(f"({row},{col})" for row, col in component["cells"])
        print(
            f"  {component['terrain']}: "
            f"{component['size']} tiles * {component['crowns']} crowns = "
            f"{component['points']} points [{cells}]"
        )

    print(f"\nTotal points: {total_points}")


def result_to_dict(image_path, terrain_grid, crown_grid, components, total_points, corrections=None):
    return {
        "image_path": str(image_path),
        "terrain_grid": terrain_grid,
        "crown_grid": crown_grid,
        "components": components,
        "total_points": total_points,
        "terrain_corrections": corrections or [],
    }


def save_json(path, result):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def draw_text_with_background(image, text, origin, font_scale, color, bg_color=(0, 0, 0), thickness=1):
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = origin
    (width, height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    cv.rectangle(
        image,
        (x - 3, y - height - 4),
        (x + width + 3, y + baseline + 3),
        bg_color,
        -1,
    )
    cv.putText(image, text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)


def draw_wrapped_lines(image, lines, x, start_y, max_y, line_height=24):
    y = start_y
    for line in lines:
        if y > max_y:
            draw_text_with_background(image, "...", (x, y), 0.55, (255, 255, 255))
            break
        cv.putText(image, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv.LINE_AA)
        y += line_height


def build_component_lookup(components):
    lookup = {}
    for index, component in enumerate(components, start=1):
        for row, col in component["cells"]:
            lookup[(row, col)] = index
    return lookup


def draw_cluster_boundaries(canvas, component, color, tile_width, tile_height):
    cells = {(row, col) for row, col in component["cells"]}

    for row, col in cells:
        x1 = col * tile_width
        y1 = row * tile_height
        x2 = x1 + tile_width
        y2 = y1 + tile_height

        if (row - 1, col) not in cells:
            cv.line(canvas, (x1, y1), (x2, y1), color, 4)
        if (row + 1, col) not in cells:
            cv.line(canvas, (x1, y2), (x2, y2), color, 4)
        if (row, col - 1) not in cells:
            cv.line(canvas, (x1, y1), (x1, y2), color, 4)
        if (row, col + 1) not in cells:
            cv.line(canvas, (x2, y1), (x2, y2), color, 4)


def render_score_image(image, terrain_grid, crown_grid, components, total_points, tile_size):
    board = image.copy()
    tile_width, tile_height = parse_tile_size(tile_size)
    board_size = len(terrain_grid)
    overlay = board.copy()

    for index, component in enumerate(components, start=1):
        color = CLUSTER_COLORS[(index - 1) % len(CLUSTER_COLORS)]
        for row, col in component["cells"]:
            x1 = col * tile_width
            y1 = row * tile_height
            x2 = x1 + tile_width
            y2 = y1 + tile_height
            cv.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    board = cv.addWeighted(overlay, 0.28, board, 0.72, 0)

    for row in range(board_size + 1):
        y = row * tile_height
        cv.line(board, (0, y), (board.shape[1], y), (245, 245, 245), 1)

    for col in range(board_size + 1):
        x = col * tile_width
        cv.line(board, (x, 0), (x, board.shape[0]), (245, 245, 245), 1)

    component_lookup = build_component_lookup(components)

    for index, component in enumerate(components, start=1):
        color = CLUSTER_COLORS[(index - 1) % len(CLUSTER_COLORS)]
        draw_cluster_boundaries(board, component, color, tile_width, tile_height)

    for row in range(board_size):
        for col in range(board_size):
            x = col * tile_width
            y = row * tile_height
            terrain = terrain_grid[row][col]
            crowns = crown_grid[row][col]
            cluster_id = component_lookup[(row, col)]
            draw_text_with_background(
                board,
                f"#{cluster_id} {terrain}",
                (x + 6, y + 20),
                0.43,
                (255, 255, 255),
            )
            draw_text_with_background(
                board,
                f"C:{crowns}",
                (x + 6, y + tile_height - 10),
                0.48,
                (255, 255, 255),
            )

    for index, component in enumerate(components, start=1):
        if component["points"] == 0:
            continue

        center_row = sum(row for row, _col in component["cells"]) / len(component["cells"])
        center_col = sum(col for _row, col in component["cells"]) / len(component["cells"])
        center_x = int((center_col + 0.5) * tile_width) - 28
        center_y = int((center_row + 0.5) * tile_height) + 8
        draw_text_with_background(
            board,
            f"{component['points']}p",
            (center_x, center_y),
            0.7,
            (255, 255, 255),
            bg_color=(25, 25, 25),
            thickness=2,
        )

    panel_width = 360
    panel = np.full((board.shape[0], panel_width, 3), 245, dtype=np.uint8)
    cv.putText(panel, "Kingdomino score", (18, 34), cv.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv.LINE_AA)
    cv.putText(panel, f"Total: {total_points}", (18, 74), cv.FONT_HERSHEY_SIMPLEX, 0.8, (20, 80, 20), 2, cv.LINE_AA)
    cv.putText(panel, "Clusters:", (18, 116), cv.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv.LINE_AA)

    lines = []
    for index, component in enumerate(components, start=1):
        if component["points"] == 0:
            continue
        lines.append(
            f"#{index} {component['terrain']}: "
            f"{component['size']} x {component['crowns']} = {component['points']}"
        )

    if not lines:
        lines.append("No scoring terrain clusters.")

    draw_wrapped_lines(panel, lines, 18, 150, panel.shape[0] - 20)
    return np.hstack([board, panel])


def show_score_image(visualization):
    cv.imshow("Kingdomino clusters and score", visualization)
    print("\nImage window opened. Press any key in the image window to close it.")
    cv.waitKey(0)
    cv.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a Kingdomino board image and calculate its score."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=f"Path to a cropped 5x5 Kingdomino board image. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument("--params-pkl", default=DEFAULT_PARAMS_PKL, help="Path to kingdomino_params.pkl.")
    parser.add_argument("--model-pkl", default=DEFAULT_MODEL_PKL, help="Path to king_domino_final_model.pkl.")
    parser.add_argument("--json", dest="json_path", help="Optional path for saving the result as JSON.")
    parser.add_argument("--output-image", help="Optional path for saving the cluster/score visualization.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the image window.")
    parser.add_argument(
        "--no-terrain-corrections",
        action="store_true",
        help="Disable crown-based terrain corrections after model prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_pickle(args.params_pkl)
    bundle = load_pickle(args.model_pkl)

    board_size = int(params.get("BOARD_SIZE", bundle.get("board_size", 5)))
    tile_size = bundle.get("tile_size", params.get("TILE_SIZE", (100, 100)))
    image_path = resolve_path(args.image)
    image = read_board_image(image_path, board_size, tile_size)

    terrain_grid = classify_terrains(image, image_path, bundle)
    crown_grid = count_board_crowns(image, board_size, tile_size, params)
    corrections = []

    if not args.no_terrain_corrections:
        terrain_grid, corrections = apply_terrain_corrections(terrain_grid, crown_grid)

    total_points, components = score_board(terrain_grid, crown_grid)
    result = result_to_dict(image_path, terrain_grid, crown_grid, components, total_points, corrections)
    visualization = render_score_image(image, terrain_grid, crown_grid, components, total_points, tile_size)

    print_result(image_path, terrain_grid, crown_grid, components, total_points, corrections)

    if args.json_path:
        save_json(args.json_path, result)
        print(f"\nSaved JSON result to {args.json_path}")

    if args.output_image:
        output_image_path = Path(args.output_image)
        cv.imwrite(str(output_image_path), visualization)
        print(f"\nSaved visualization to {output_image_path}")

    if not args.no_show:
        show_score_image(visualization)


if __name__ == "__main__":
    main()
