import cv2 as cv
import numpy as np
from pathlib import Path

#TEST: 46, 47, 50, 51, 53, 54, 56, 61, 62, 63, 65, 68, 69 og 73.

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "King Domino dataset"
TEMPLATE_DIR = BASE_DIR / "Templates"
MATCH_SCALES = np.linspace(0.85, 1.15)
TEMPLATE_MATCH_THRESHOLD = 0.78
COLOR_HIST_THRESHOLD = 0.60
TEMPLATE_SCORE_WEIGHT = 0.75
COLOR_SCORE_WEIGHT = 0.75
BOARD_SIZE = 5
TILE_SIZE = 100
CROWN_SEARCH_BUFFER = 10
EDGE_PADDING_MODE = cv.BORDER_REPLICATE

_CROWN_TEMPLATES = None


# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")

    image_path = DATASET_DIR / "73.jpg"

    if not image_path.is_file():
        print("Image not found")
        return

    image = cv.imread(str(image_path))
    debug_image = image.copy()

    tiles = get_tiles(image)
    crown_tiles = get_tiles(image, buffer=CROWN_SEARCH_BUFFER, include_inner_bounds=True)

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):

            crown_tile, count_area = crown_tiles[y][x]
            debug_offset = (
                x * TILE_SIZE - count_area[0],
                y * TILE_SIZE - count_area[1],
            )
            crowns = find_crowns(
                crown_tile,
                show_debug=False,
                count_area=count_area,
                debug_image=debug_image,
                debug_offset=debug_offset,
            )

            print(f"Crowns: {crowns}")
            print(f"Tile ({x}, {y}):")
            get_terrain(tile)
            print("=====")

    cv.imshow("Image Debug", debug_image)
    cv.waitKey()


# Break a board into tiles
def get_tiles(image, buffer=0, include_inner_bounds=False):
    tiles = []
    image_height, image_width = image.shape[:2]

    for y in range(BOARD_SIZE):
        tiles.append([])
        for x in range(BOARD_SIZE):
            tile_x1 = x * TILE_SIZE
            tile_y1 = y * TILE_SIZE
            tile_x2 = (x + 1) * TILE_SIZE
            tile_y2 = (y + 1) * TILE_SIZE

            desired_crop_x1 = tile_x1 - buffer
            desired_crop_y1 = tile_y1 - buffer
            desired_crop_x2 = tile_x2 + buffer
            desired_crop_y2 = tile_y2 + buffer

            crop_x1 = max(0, desired_crop_x1)
            crop_y1 = max(0, desired_crop_y1)
            crop_x2 = min(image_width, desired_crop_x2)
            crop_y2 = min(image_height, desired_crop_y2)

            tile = image[crop_y1:crop_y2, crop_x1:crop_x2]

            pad_left = crop_x1 - desired_crop_x1
            pad_top = crop_y1 - desired_crop_y1
            pad_right = desired_crop_x2 - crop_x2
            pad_bottom = desired_crop_y2 - crop_y2

            if pad_left or pad_top or pad_right or pad_bottom:
                tile = cv.copyMakeBorder(
                    tile,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    EDGE_PADDING_MODE,
                )

            if include_inner_bounds:
                inner_bounds = (
                    tile_x1 - desired_crop_x1,
                    tile_y1 - desired_crop_y1,
                    tile_x2 - desired_crop_x1,
                    tile_y2 - desired_crop_y1,
                )
                tiles[-1].append((tile, inner_bounds))
            else:
                tiles[-1].append(tile)

    return tiles


# Determine the type of terrain in a tile
def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.mean(hsv_tile, axis=(0, 1))
    print(f"H: {hue}, S: {saturation}, V: {value}")


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


def get_crown_templates():
    global _CROWN_TEMPLATES

    if _CROWN_TEMPLATES is not None:
        return _CROWN_TEMPLATES

    templates = []

    if not TEMPLATE_DIR.is_dir():
        print("Templates folder not found")
        _CROWN_TEMPLATES = templates
        return templates

    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        template = cv.imread(str(path))

        if template is None:
            continue

        for scale in MATCH_SCALES:
            resized = cv.resize(template, None, fx=scale, fy=scale)
            h, w = resized.shape[:2]

            if h == 0 or w == 0:
                continue

            templates.append({
                "filename": path.name,
                "blur": cv.GaussianBlur(resized, (3, 3), 0),
                "hist": create_color_histogram(resized),
                "w": w,
                "h": h,
            })

    _CROWN_TEMPLATES = templates
    return templates


def is_match_inside_count_area(x, y, w, h, count_area):
    if count_area is None:
        return True

    area_x1, area_y1, area_x2, area_y2 = count_area
    center_x = x + w / 2
    center_y = y + h / 2

    return area_x1 <= center_x < area_x2 and area_y1 <= center_y < area_y2


def find_crowns(tile, show_debug=True, count_area=None, debug_image=None, debug_offset=(0, 0)):
    tile_blur = cv.GaussianBlur(tile, (3, 3), 0)

    matches_all = []

    for template in get_crown_templates():
        h = template["h"]
        w = template["w"]

        if h > tile_blur.shape[0] or w > tile_blur.shape[1]:
            continue

        result = cv.matchTemplate(tile_blur, template["blur"], cv.TM_CCOEFF_NORMED)
        locations = np.where(result >= TEMPLATE_MATCH_THRESHOLD)

        for pt in zip(*locations[::-1]):
            template_score = float(result[pt[1], pt[0]])
            x, y = pt

            if not is_match_inside_count_area(x, y, w, h, count_area):
                continue

            candidate = tile[y:y + h, x:x + w]
            color_score = compare_color_histograms(candidate, template["hist"])

            if color_score < COLOR_HIST_THRESHOLD:
                continue

            combined_score = (
                template_score * TEMPLATE_SCORE_WEIGHT
                + color_score * COLOR_SCORE_WEIGHT
            )

            matches_all.append((
                x,
                y,
                w,
                h,
                combined_score,
                template_score,
                color_score,
                template["filename"],
            ))

    # sorter efter score
    matches_all = sorted(matches_all, key=lambda x: x[4], reverse=True)

    # overlap filtering (mindre aggressiv)
    filtered = []

    for match in matches_all:
        x, y, w, h, score, template_score, color_score, filename = match

        too_close = False
        for fx, fy, fw, fh, *_ in filtered:
            if abs(x - fx) < w * 0.5 and abs(y - fy) < h * 0.5:
                too_close = True
                break

        if not too_close:
            filtered.append(match)

    print("Final crowns:", len(filtered))

    # -------- VISUAL DEBUG --------
    debug_tile = debug_image if debug_image is not None else tile.copy()
    offset_x, offset_y = debug_offset if debug_image is not None else (0, 0)

    if count_area is not None:
        area_x1, area_y1, area_x2, area_y2 = count_area
        cv.rectangle(
            debug_tile,
            (area_x1 + offset_x, area_y1 + offset_y),
            (area_x2 - 1 + offset_x, area_y2 - 1 + offset_y),
            (255, 0, 0),
            1,
        )

    for (x, y, w, h, score, template_score, color_score, filename) in filtered:
        draw_x = x + offset_x
        draw_y = y + offset_y
        cv.rectangle(debug_tile, (draw_x, draw_y), (draw_x + w, draw_y + h), (0, 255, 0), 2)
        cv.putText(
            debug_tile,
            f"{score:.2f}",
            (draw_x, max(10, draw_y - 4)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

    if show_debug:
        cv.imshow("Tile Debug", debug_tile)
        cv.waitKey()

    return len(filtered)


if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
