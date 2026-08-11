"""
Evalueer het gefinetunede model op de validatieset en sla de resultaten
op als markdown-tabel, klaar om in de README te plakken.

Run: python src/evaluate.py
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "processed" / "cardd_yolo" / "dataset.yaml"
FINETUNED_MODEL = Path(__file__).resolve().parent.parent / "runs" / "finetune" / "cardd_finetune" / "weights" / "best.pt"
OUTPUT_MD = Path(__file__).resolve().parent.parent / "runs" / "eigen_evaluatie.md"


def evaluate():
    if not FINETUNED_MODEL.exists():
        raise FileNotFoundError("Run eerst: python src/train.py")

    model = YOLO(str(FINETUNED_MODEL))
    metrics = model.val(data=str(DATA_YAML), split="val")

    # Bouw een leesbare markdown-tabel op, per klasse
    rows = ["| Klasse | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    names = metrics.names
    for i, cls_name in names.items():
        p = metrics.box.p[i] if i < len(metrics.box.p) else float("nan")
        r = metrics.box.r[i] if i < len(metrics.box.r) else float("nan")
        map50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
        map5095 = metrics.box.ap[i] if i < len(metrics.box.ap) else float("nan")
        rows.append(f"| {cls_name} | {p:.3f} | {r:.3f} | {map50:.3f} | {map5095:.3f} |")

    rows.append(f"| **Alle klassen** | {metrics.box.mp:.3f} | {metrics.box.mr:.3f} | {metrics.box.map50:.3f} | {metrics.box.map:.3f} |")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(rows))

    print("\n".join(rows))
    print(f"\nOpgeslagen als: {OUTPUT_MD}")
    print("Kopieer deze tabel naar de README onder 'Evaluatie' (eigen resultaten).")


if __name__ == "__main__":
    evaluate()
