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
MIN_SIZE_BYTES = 1_000_000  # 1 MB — een geldig YOLO-model is veel groter dan dit


def download_baseline_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / MODEL_FILENAME

    if dest.exists() and dest.stat().st_size > MIN_SIZE_BYTES:
        print(f"Model bestaat al: {dest}")
        return dest

    if dest.exists():
        print(f"Bestaand model lijkt corrupt/incompleet ({dest.stat().st_size} bytes), opnieuw downloaden...")
        dest.unlink()

    print(f"Model downloaden van {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, dest)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
    print(f"Model opgeslagen op: {dest}")
    return dest


if __name__ == "__main__":
    download_baseline_model()
   
