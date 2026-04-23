import pickle
from pathlib import Path

import cv2 as cv

import kingdomino


OUTPUT_PATH = Path(__file__).resolve().parent / "kingdomino_params.pkl"


def build_params():
    return {
        "MATCH_SCALES": kingdomino.MATCH_SCALES,
        "MATCH_SCALE_START": float(kingdomino.MATCH_SCALES[0]),
        "MATCH_SCALE_END": float(kingdomino.MATCH_SCALES[-1]),
        "MATCH_SCALE_COUNT": int(len(kingdomino.MATCH_SCALES)),
        "TEMPLATE_MATCH_THRESHOLD": float(kingdomino.TEMPLATE_MATCH_THRESHOLD),
        "COLOR_HIST_THRESHOLD": float(kingdomino.COLOR_HIST_THRESHOLD),
        "TEMPLATE_SCORE_WEIGHT": float(kingdomino.TEMPLATE_SCORE_WEIGHT),
        "COLOR_SCORE_WEIGHT": float(kingdomino.COLOR_SCORE_WEIGHT),
        "BOARD_SIZE": int(kingdomino.BOARD_SIZE),
        "TILE_SIZE": int(kingdomino.TILE_SIZE),
        "CROWN_SEARCH_BUFFER": int(kingdomino.CROWN_SEARCH_BUFFER),
        "EDGE_PADDING_MODE": int(kingdomino.EDGE_PADDING_MODE),
        "EDGE_PADDING_MODE_NAME": (
            "cv.BORDER_REPLICATE"
            if kingdomino.EDGE_PADDING_MODE == cv.BORDER_REPLICATE
            else str(kingdomino.EDGE_PADDING_MODE)
        ),
        "TEMPLATE_DIR": str(kingdomino.TEMPLATE_DIR),
    }


def main():
    params = build_params()

    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(params, file)

    print(f"Saved parameters to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
