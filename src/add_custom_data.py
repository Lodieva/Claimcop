"""
Voeg je eigen gefotografeerde en geannoteerde schade-afbeeldingen toe aan
de CarDD-trainingsset. Dit versterkt de "Data collection"-stap van de
pipeline met eigen, praktijkgerichte data naast de publieke dataset.

Workflow:
1. Maak zelf foto's van (nagebootste) autoschade
2. Annoteer ze met Roboflow (https://roboflow.com) of CVAT (https://cvat.ai)
   in YOLO-segmentatieformaat, met dezelfde 6 klassen als CarDD:
   crack, dent, glass shatter, lamp broken, scratch, tire flat
3. Exporteer als YOLO-segmentatiedataset (images/ + labels/)
4. Zet de geëxporteerde map in data/raw/custom/ (zelfde structuur:
   images/ en labels/ submappen)
5. Run dit script om ze te mergen in data/processed/cardd_yolo/train/

Run: python src/add_custom_data.py --source data/raw/custom
"""

import argparse
import shutil
from pathlib import Path

CARDD_YOLO_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "cardd_yolo"


def merge_custom_data(source_dir: Path, target_split: str = "train"):
    source_images = source_dir / "images"
    source_labels = source_dir / "labels"

    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(
            f"Verwacht {source_dir}/images/ en {source_dir}/labels/ — "
            "check dat je export in YOLO-formaat staat."
        )

    target_images = CARDD_YOLO_DIR / target_split / "images"
    target_labels = CARDD_YOLO_DIR / target_split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for img_file in source_images.glob("*"):
        label_file = source_labels / f"{img_file.stem}.txt"
        if not label_file.exists():
            print(f"Overslaan (geen label gevonden): {img_file.name}")
            continue

        # Prefix om naamcollisions met CarDD-bestanden te voorkomen
        new_stem = f"custom_{img_file.stem}"
        shutil.copy(img_file, target_images / f"{new_stem}{img_file.suffix}")
        shutil.copy(label_file, target_labels / f"{new_stem}.txt")
        n_copied += 1

    print(f"{n_copied} eigen afbeeldingen toegevoegd aan {target_split}-split.")
    print(
        "\nBELANGRIJK: controleer dat de class-indices in jouw labels exact "
        "overeenkomen met de volgorde in dataset.yaml van CarDD "
        "(crack, dent, glass shatter, lamp broken, scratch, tire flat), "
        "anders krijg je verkeerde labels na het mergen."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Map met eigen data in YOLO-formaat (bevat images/ en labels/)"
    )
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    merge_custom_data(args.source, args.split)
