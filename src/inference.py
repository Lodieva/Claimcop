"""
Herbruikbare inference-functies voor het CarDD segmentatiemodel.
Wordt zowel gebruikt door de webapp (app/main.py) als door test/evaluatiescripts.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
       from ultralytics import YOLO

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best.pt"


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> "YOLO":
    from ultralytics import YOLO
    """
    Laadt het model één keer in het geheugen (cache), zodat de webapp
    niet bij elke upload opnieuw het model hoeft te laden.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model niet gevonden op {model_path}. "
            "Run eerst: python src/download_model.py"
        )
    return YOLO(str(model_path))


def mask_area_percentage(mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Berekent het percentage van de afbeelding dat door een mask bedekt wordt.
    Los van model/IO gehouden zodat dit zonder gewichten te laden getest kan worden.
    """
    if mask.size == 0:
        return 0.0
    return round(100 * float(np.sum(mask > threshold)) / mask.size, 2)


def detect_damage(image_path: str | Path, conf: float = 0.25):
    """
    Voert instance segmentation uit op een afbeelding.

    Returns:
        results: Ultralytics Results object (bevat boxes, masks, classes, confidences)
        summary: lijst van dicts met leesbare info per gevonden schade,
                 inclusief geschat oppervlaktepercentage (op basis van het mask)
    """
    model = load_model()
    results = model.predict(source=str(image_path), conf=conf, verbose=False)

    r = results[0]
    summary = []
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        confidence = float(box.conf[0])

        area_pct = None
        if r.masks is not None:
            mask = r.masks.data[i].cpu().numpy()  # (h, w) in modelresolutie
            area_pct = mask_area_percentage(mask)

        summary.append({
            "class": cls_name,
            "confidence": round(confidence, 3),
            "bbox_xyxy": [round(x, 1) for x in box.xyxy[0].tolist()],
            "area_pct": area_pct,
        })

    return r, summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Gebruik: python src/inference.py <pad_naar_afbeelding>")
        sys.exit(1)

    _, summary = detect_damage(sys.argv[1])
    print(f"Gevonden schade ({len(summary)}):")
    for item in summary:
        area = f", oppervlak: {item['area_pct']}%" if item["area_pct"] is not None else ""
        print(f"  - {item['class']} (confidence: {item['confidence']}{area})")
