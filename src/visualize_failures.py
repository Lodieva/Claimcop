"""
Visuele inspectie van model-voorspellingen op de validatieset: waar gaat
het mis, en waarom? Dit is vaak informatiever dan alleen mAP-cijfers
(zie pipeline-punt 9 "Evaluatie": visuele inspectie van failure cases).

Genereert een grid van afbeeldingen met voorspelling vs. ground truth
naast elkaar, met nadruk op lage-confidence en gemiste detecties.

Run: python src/visualize_failures.py
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "cardd_yolo"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "failure_analysis"


def visualize_predictions(
    model_path: Path = MODELS_DIR / "best.pt",
    n_samples: int = 9,
    conf_threshold: float = 0.25,
):
    if not model_path.exists():
        raise FileNotFoundError(f"Model niet gevonden op {model_path}")

    val_images_dir = DATA_DIR / "val" / "images"
    if not val_images_dir.exists():
        raise FileNotFoundError(
            f"{val_images_dir} niet gevonden. Run eerst src/prepare_data.py"
        )

    model = YOLO(str(model_path))
    all_images = list(val_images_dir.glob("*"))
    sample_images = random.sample(all_images, min(n_samples, len(all_images)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = 3
    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    low_confidence_cases = []

    for ax, img_path in zip(axes, sample_images):
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        r = results[0]
        annotated = r.plot()  # BGR numpy array
        ax.imshow(annotated[:, :, ::-1])

        confidences = [float(c) for c in r.boxes.conf] if len(r.boxes) else []
        n_detections = len(confidences)
        min_conf = min(confidences) if confidences else None

        title = f"{img_path.name}\n{n_detections} detectie(s)"
        if min_conf is not None:
            title += f", laagste conf: {min_conf:.0%}"
            if min_conf < 0.4:
                low_confidence_cases.append((img_path.name, min_conf))
        elif n_detections == 0:
            title += " ⚠️ GEEN schade gedetecteerd"

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Lege subplots uitschakelen als n_samples niet exact op een grid past
    for ax in axes[len(sample_images):]:
        ax.axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "failure_analysis_grid.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Grid opgeslagen: {output_path}")

    if low_confidence_cases:
        print("\nAfbeeldingen met lage-confidence detecties (mogelijke failure cases):")
        for name, conf in low_confidence_cases:
            print(f"  - {name}: {conf:.0%}")
        print(
            "\nGebruik dit als startpunt voor je 'Beperkingen'-sectie: "
            "bekijk deze afbeeldingen handmatig — is het beeld onduidelijk, "
            "de schade subtiel, of iets anders (vuil, regen, hoek)?"
        )


if __name__ == "__main__":
    visualize_predictions()
