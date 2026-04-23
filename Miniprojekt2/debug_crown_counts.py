import contextlib
import io
from pathlib import Path

import cv2 as cv

import kingdomino


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "King Domino dataset"


def image_sort_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def count_crowns_in_image(image):
    tiles = kingdomino.get_tiles(image)
    crown_tiles = kingdomino.get_tiles(
        image,
        buffer=kingdomino.CROWN_SEARCH_BUFFER,
        include_inner_bounds=True,
    )

    total_crowns = 0

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            crown_tile, count_area = crown_tiles[y][x]
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                total_crowns += kingdomino.find_crowns(
                    crown_tile,
                    show_debug=False,
                    count_area=count_area,
                )

    return total_crowns


def main():
    image_paths = sorted(DATASET_DIR.glob("*.jpg"), key=image_sort_key)

    if not image_paths:
        print("No jpg images found")
        return

    print(f"{'Image':<10} {'Crowns':>6}")
    print("-" * 17)

    for image_path in image_paths:
        image = cv.imread(str(image_path))

        if image is None:
            print(f"{image_path.name:<10} {'Error':>6}")
            continue

        crowns = count_crowns_in_image(image)

        print(f"{image_path.name:<10} {crowns:>6}")


if __name__ == "__main__":
    main()
