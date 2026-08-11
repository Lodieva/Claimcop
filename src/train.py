"""
Finetune het pretrained CarDD-model verder (transfer learning), i.p.v.
training from scratch. Geschikt voor beperkte tijd/compute: we starten
vanaf gewichten die al goed werken en trainen kort door.

Vereist: python src/download_model.py  en  python src/prepare_data.py
zijn al gerund.

Run: python src/train.py
Op een laptop zonder NVIDIA-GPU (bijv. Intel Iris Xe) draait dit op CPU —
dat is significant trager dan met GPU. Gebruik daarom:
  - een kleine subset van de data om eerst te verifiëren dat alles werkt
    (zie --smoke-test), en pas daarna een langere/volledige run
  - een kleiner beeldformaat (imgsz) en batch size
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "processed" / "cardd_yolo" / "dataset.yaml"
PRETRAINED = MODELS_DIR / "best.pt"


def finetune(
    epochs: int = 15,       # CPU-vriendelijk gehouden; verhoog als je tijd hebt
    imgsz: int = 512,       # kleiner dan de standaard 640/1024 -> sneller op CPU
    batch: int = 4,         # klein houden i.v.m. beperkt CPU-geheugen
    device: str = "cpu",
):
    if not PRETRAINED.exists():
        raise FileNotFoundError("Run eerst: python src/download_model.py")
    if not DATA_YAML.exists():
        raise FileNotFoundError("Run eerst: python src/prepare_data.py")

    model = YOLO(str(PRETRAINED))  # start vanaf pretrained gewichten, geen scratch-training

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/finetune",
        name="cardd_finetune",
        patience=5,       # early stopping, belangrijk op CPU om tijd te besparen
        val=True,
        plots=True,       # genereert automatisch grafieken (loss, PR-curve, confusion matrix)
        workers=2,        # laag houden op laptops met beperkte CPU-cores
    )

    print("\nFinetuning klaar. Resultaten (incl. grafieken) staan in runs/finetune/cardd_finetune/")
    print("Het beste model staat in runs/finetune/cardd_finetune/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Zeer korte run (3 epochs) om te checken dat de hele pipeline werkt, "
             "voordat je een lange training start."
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("Smoke test: 3 epochs, klein formaat, om snel te checken dat alles werkt...")
        finetune(epochs=3, imgsz=384, batch=4, device=args.device)
    else:
        finetune(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
