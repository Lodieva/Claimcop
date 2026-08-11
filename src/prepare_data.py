"""
Download de CarDD-dataset van Hugging Face (via FiftyOne) en exporteer
naar YOLO-segmentatieformaat, klaar om mee te finetunen.

CarDD staat getagd als FiftyOne-dataset op Hugging Face, dus we gebruiken
de FiftyOne-integratie i.p.v. handmatige annotatie-conversie.

Bron: https://huggingface.co/datasets/harpreetsahota/CarDD
Licentie: niet-commercieel onderzoek/educatie (Flickr/Shutterstock-beelden)

Run: python src/prepare_data.py
"""

from pathlib import Path

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPORT_DIR = DATA_DIR / "processed" / "cardd_yolo"


def load_cardd():
    print("CarDD-dataset laden vanaf Hugging Face (kan even duren, ~2.2 GB)...")
    dataset = fouh.load_from_hub("harpreetsahota/CarDD", persistent=True)
    print(dataset)

    # BELANGRIJK: controleer het exacte label-veld voordat je verder gaat.
    # Op het modelcard van harpreetsahota wordt gt_field="segmentations"
    # gebruikt -- print hieronder een sample om dit te bevestigen, en pas
    # LABEL_FIELD aan als het anders heet in jouw lokale versie.
    sample = dataset.first()
    print("\nVeldnamen van een voorbeeld-sample:")
    print(sample.field_names)

    return dataset


def export_to_yolo(dataset, label_field: str = "segmentations", max_samples: int | None = None):
    """
    Splitst de dataset in train/val en exporteert naar YOLO-segmentatieformaat
    (images/ + labels/ per split, plus een data.yaml).

    max_samples: als gezet, gebruik alleen dit aantal samples (voor een
    snelle smoke test op CPU voordat je de volledige dataset verwerkt).
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if max_samples is not None:
        print(f"Let op: beperkt tot {max_samples} samples (smoke test-modus)")
        dataset = dataset.take(max_samples)

    # Eenvoudige 80/20 split als de dataset zelf geen split-tags heeft.
    # Check eerst of er al een "split" veld bestaat (tags op sample-niveau).
    if "train" in dataset.distinct("tags"):
        train_view = dataset.match_tags("train")
        val_view = dataset.match_tags(["val", "validation", "test"])
    else:
        train_view, val_view = dataset.split(splits={"train": 0.8, "val": 0.2})

    for split_name, view in [("train", train_view), ("val", val_view)]:
        print(f"Exporteren van {split_name}-split ({len(view)} afbeeldingen)...")
        view.export(
            export_dir=str(EXPORT_DIR),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=split_name,
        )

    print(f"\nKlaar. Data + data.yaml staan in: {EXPORT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Gebruik maar 200 samples i.p.v. de volledige dataset, om snel "
             "de hele pipeline te verifiëren voordat je de lange (volledige) "
             "download/verwerking start."
    )
    args = parser.parse_args()

    ds = load_cardd()
    export_to_yolo(ds, max_samples=200 if args.smoke_test else None)
