"""
Download het pretrained YOLOv11-Seg CarDD model.

Dit model is finetuned op de CarDD-dataset (Wang et al., 2023) en doet
instance segmentation op 6 schadetypes: crack, dent, glass shatter,
lamp broken, scratch, tire flat.

Bron model: https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11
Bron dataset: https://huggingface.co/datasets/harpreetsahota/CarDD
Paper: Wang, Li & Wu (2023), "CarDD: A New Dataset for Vision-Based
       Car Damage Detection", IEEE Trans. Intelligent Transportation Systems.
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URL = "https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11/resolve/main/best.pt"
MODEL_FILENAME = "best.pt"


def download_baseline_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / MODEL_FILENAME

    if dest.exists():
        print(f"Model bestaat al: {dest}")
        return dest

    print(f"Model downloaden van {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, dest)
    print(f"Model opgeslagen op: {dest}")
    return dest


if __name__ == "__main__":
    download_baseline_model()
