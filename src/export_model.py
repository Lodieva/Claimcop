"""
Exporteer het model naar ONNX-formaat voor snellere/lichtere inference.

Waarom ONNX (pipeline-stap "Deployment" uit de opdracht):
- Kleiner en sneller dan het originele PyTorch-checkpoint bij inference
- Platform-onafhankelijk (draait zonder volledige PyTorch-installatie)
- Voorbereidende stap richting edge-deployment (bijv. op een garage-tablet
  i.p.v. altijd in de cloud te moeten inferen)

Run: python src/export_model.py [--model PAD_NAAR_MODEL]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "best.pt"


def export_to_onnx(model_path: Path = DEFAULT_MODEL, imgsz: int = 640, half: bool = False):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model niet gevonden op {model_path}. "
            "Run eerst download_model.py of train.py."
        )

    model = YOLO(str(model_path))
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,       # FP16: kleiner/sneller, alleen zinvol met GPU-inference
        simplify=True,   # verwijdert overbodige ONNX-operaties
        opset=12,
    )
    print(f"ONNX-model opgeslagen op: {onnx_path}")

    # Kort verslag van de bestandsgrootte-winst, relevant voor de "trade-offs"
    # sectie van je conclusie
    pt_size_mb = model_path.stat().st_size / (1024 * 1024)
    onnx_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"PyTorch (.pt): {pt_size_mb:.1f} MB  →  ONNX: {onnx_size_mb:.1f} MB")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Exporteer in FP16 (GPU-inference)")
    args = parser.parse_args()

    export_to_onnx(args.model, args.imgsz, args.half)
