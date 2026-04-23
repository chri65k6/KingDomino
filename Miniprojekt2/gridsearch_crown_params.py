from pathlib import Path

import cv2 as cv
import numpy as np

import kingdomino


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "King Domino dataset"
TEMPLATE_DIR = BASE_DIR / "Templates"

ERROR_CASES = [
    {"image": 14, "x": 0, "y": 0, "error": "missing", "original": 0, "target": 1},
    {"image": 31, "x": 2, "y": 1, "error": "missing", "original": 1, "target": 2},
    {"image": 38, "x": 3, "y": 2, "error": "false_positive", "original": 1, "target": 0},
    {"image": 57, "x": 1, "y": 4, "error": "false_positive", "original": 3, "target": 2},
    {"image": 61, "x": 1, "y": 4, "error": "false_positive", "original": 3, "target": 2},
    {"image": 71, "x": 0, "y": 3, "error": "missing", "original": 0, "target": 1},
]

SCALE_RANGES = [
    (0.75, 1.10),
    (0.75, 1.15),
    (0.75, 1.20),
    (0.80, 1.10),
    (0.80, 1.15),
    (0.80, 1.20),
    (0.85, 1.10),
    (0.85, 1.15),
    (0.85, 1.20),
    (0.90, 1.10),
    (0.90, 1.15),
    (0.90, 1.20),
]
TEMPLATE_THRESHOLDS = np.round(np.arange(0.68, 0.91, 0.02), 2)
COLOR_THRESHOLDS = np.round(np.arange(0.25, 0.66, 0.05), 2)
TEMPLATE_WEIGHTS = np.round(np.arange(0.55, 0.96, 0.05), 2)
CROWN_SEARCH_BUFFERS = [5, 10, 15, 20, 25]

MIN_TEMPLATE_THRESHOLD = float(min(TEMPLATE_THRESHOLDS))


def create_templates(scale_start, scale_end):
    templates = []
    seen = set()

    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        template = cv.imread(str(path))

        if template is None:
            continue

        for scale in np.linspace(scale_start, scale_end):
            resized = cv.resize(template, None, fx=scale, fy=scale)
            h, w = resized.shape[:2]
            key = (path.name, w, h)

            if h == 0 or w == 0 or key in seen:
                continue

            seen.add(key)
            templates.append({
                "filename": path.name,
                "blur": cv.GaussianBlur(resized, (3, 3), 0),
                "hist": kingdomino.create_color_histogram(resized),
                "w": w,
                "h": h,
            })

    return templates


def get_buffered_case_tile(image, x, y, buffer):
    crown_tiles = kingdomino.get_tiles(
        image,
        buffer=buffer,
        include_inner_bounds=True,
    )
    return crown_tiles[y][x]


def collect_candidates(tile, count_area, templates):
    tile_blur = cv.GaussianBlur(tile, (3, 3), 0)
    candidates = []

    for template in templates:
        h = template["h"]
        w = template["w"]

        if h > tile_blur.shape[0] or w > tile_blur.shape[1]:
            continue

        result = cv.matchTemplate(tile_blur, template["blur"], cv.TM_CCOEFF_NORMED)
        locations = np.where(result >= MIN_TEMPLATE_THRESHOLD)

        for pt in zip(*locations[::-1]):
            x, y = pt

            if not kingdomino.is_match_inside_count_area(x, y, w, h, count_area):
                continue

            candidate = tile[y:y + h, x:x + w]
            candidates.append((
                x,
                y,
                w,
                h,
                float(result[y, x]),
                kingdomino.compare_color_histograms(candidate, template["hist"]),
            ))

    return candidates


def count_from_candidates(candidates, template_threshold, color_threshold, template_weight):
    color_weight = 1 - template_weight
    matches = []

    for x, y, w, h, template_score, color_score in candidates:
        if template_score < template_threshold or color_score < color_threshold:
            continue

        combined_score = template_score * template_weight + color_score * color_weight
        matches.append((x, y, w, h, combined_score))

    matches = sorted(matches, key=lambda match: match[4], reverse=True)
    filtered = []

    for match in matches:
        x, y, w, h, score = match

        too_close = False
        for fx, fy, fw, fh, _ in filtered:
            if abs(x - fx) < w * 0.5 and abs(y - fy) < h * 0.5:
                too_close = True
                break

        if not too_close:
            filtered.append(match)

    return len(filtered)


def evaluate_case(case, params, image_cache):
    scale_start, scale_end, template_threshold, color_threshold, template_weight, buffer = params
    image = image_cache[case["image"]]
    tile, count_area = get_buffered_case_tile(image, case["x"], case["y"], buffer)
    templates = create_templates(scale_start, scale_end)
    candidates = collect_candidates(tile, count_area, templates)
    return count_from_candidates(candidates, template_threshold, color_threshold, template_weight)


def build_targets(image_cache):
    current_params = (
        float(kingdomino.MATCH_SCALES[0]),
        float(kingdomino.MATCH_SCALES[-1]),
        kingdomino.TEMPLATE_MATCH_THRESHOLD,
        kingdomino.COLOR_HIST_THRESHOLD,
        kingdomino.TEMPLATE_SCORE_WEIGHT,
        kingdomino.CROWN_SEARCH_BUFFER,
    )

    targets = []

    for case in ERROR_CASES:
        current_count = evaluate_case(case, current_params, image_cache)

        targets.append({
            **case,
            "current": current_count,
        })

    return targets


def make_candidate_cache(targets, image_cache):
    candidate_cache = {}
    total_jobs = len(SCALE_RANGES) * len(CROWN_SEARCH_BUFFERS)
    finished_jobs = 0

    for scale_start, scale_end in SCALE_RANGES:
        templates = create_templates(scale_start, scale_end)

        for buffer in CROWN_SEARCH_BUFFERS:
            key = (scale_start, scale_end, buffer)
            candidate_cache[key] = []

            for case in targets:
                image = image_cache[case["image"]]
                tile, count_area = get_buffered_case_tile(image, case["x"], case["y"], buffer)
                candidate_cache[key].append(collect_candidates(tile, count_area, templates))

            finished_jobs += 1
            print(f"Prepared {finished_jobs}/{total_jobs}: scales={scale_start:.2f}-{scale_end:.2f}, buffer={buffer}")

    return candidate_cache


def score_params(params, targets, candidate_cache):
    scale_start, scale_end, template_threshold, color_threshold, template_weight, buffer = params
    candidates_key = (scale_start, scale_end, buffer)
    predictions = [
        count_from_candidates(candidates, template_threshold, color_threshold, template_weight)
        for candidates in candidate_cache[candidates_key]
    ]
    errors = [
        abs(prediction - target["target"])
        for prediction, target in zip(predictions, targets)
    ]
    exact_hits = sum(error == 0 for error in errors)
    total_error = sum(errors)
    changed_from_current = sum(
        prediction != target["original"]
        for prediction, target in zip(predictions, targets)
    )

    return {
        "params": params,
        "predictions": predictions,
        "errors": errors,
        "exact_hits": exact_hits,
        "total_error": total_error,
        "changed_from_current": changed_from_current,
    }


def main():
    image_cache = {}

    for case in ERROR_CASES:
        if case["image"] in image_cache:
            continue

        image_path = DATASET_DIR / f"{case['image']}.jpg"
        image = cv.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        image_cache[case["image"]] = image

    targets = build_targets(image_cache)

    print("Targets from submitted error list:")
    print(f"{'Image':<8} {'Tile':<8} {'Type':<15} {'Original':>8} {'Now':>4} {'Target':>6}")
    print("-" * 57)
    for target in targets:
        print(
            f"{target['image']}.jpg"
            f"{'':<2} ({target['x']},{target['y']})"
            f"{'':<2} {target['error']:<15}"
            f"{target['original']:>8}"
            f"{target['current']:>4}"
            f"{target['target']:>6}"
        )

    candidate_cache = make_candidate_cache(targets, image_cache)
    results = []

    for scale_start, scale_end in SCALE_RANGES:
        for buffer in CROWN_SEARCH_BUFFERS:
            for template_threshold in TEMPLATE_THRESHOLDS:
                for color_threshold in COLOR_THRESHOLDS:
                    for template_weight in TEMPLATE_WEIGHTS:
                        params = (
                            scale_start,
                            scale_end,
                            float(template_threshold),
                            float(color_threshold),
                            float(template_weight),
                            buffer,
                        )
                        results.append(score_params(params, targets, candidate_cache))

    results.sort(key=lambda result: (
        result["total_error"],
        -result["exact_hits"],
        result["changed_from_current"],
        result["params"][2],
        result["params"][3],
        abs(result["params"][4] - kingdomino.TEMPLATE_SCORE_WEIGHT),
        abs(result["params"][5] - kingdomino.CROWN_SEARCH_BUFFER),
    ))

    print()
    print("Best parameter sets:")
    print(
        f"{'Rank':<5} {'Error':>5} {'Hits':>4} {'Scales':<11} "
        f"{'T_thr':>5} {'C_thr':>5} {'T_w':>5} {'C_w':>5} {'Buf':>4} Predictions"
    )
    print("-" * 88)

    for rank, result in enumerate(results[:20], start=1):
        scale_start, scale_end, template_threshold, color_threshold, template_weight, buffer = result["params"]
        color_weight = 1 - template_weight
        predictions = ", ".join(str(prediction) for prediction in result["predictions"])
        print(
            f"{rank:<5}"
            f"{result['total_error']:>5}"
            f"{result['exact_hits']:>4}"
            f" {scale_start:.2f}-{scale_end:.2f}"
            f"{template_threshold:>7.2f}"
            f"{color_threshold:>6.2f}"
            f"{template_weight:>6.2f}"
            f"{color_weight:>6.2f}"
            f"{buffer:>5}"
            f" [{predictions}]"
        )


if __name__ == "__main__":
    main()
