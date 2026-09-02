# 🚗 Car Damage Detection

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)]()

Een Computer Vision-systeem dat autoschade automatisch detecteert en classificeert aan de hand van foto's, om de eerste beoordeling (triage) van een schadeclaim sneller en consistenter te maken.

![Demo van de detectie](app/assets/demo-screenshot.png)

*Voorbeeld: het model detecteert een dent en cracks op basis van een geüploade foto, met geschat schadeoppervlak per detectie.*

## Probleemstelling

Verzekeringsmaatschappijen ontvangen dagelijks schadeclaims waarbij medewerkers handmatig foto's moeten beoordelen. Dit kost tijd en de beoordeling kan per schade-expert verschillen.

Doelgroep: verzekeringsmaatschappijen, schade-experts, garagebedrijven, leasebedrijven.

## Computer Vision-taak

Instance Segmentation (classificatie + pixel-masker per schade-instantie). Het model herkent, lokaliseert én segmenteert schade zoals krassen, deuken, scheuren, gebroken glas/lampen en lekke banden. Dankzij de pixel-masks kan het geschatte schadeoppervlak (%) berekend worden — niet mogelijk met alleen bounding boxes.

## Model

- **Architectuur:** YOLOv11x-Seg (Ultralytics), finetuned op basis van `yolo11x-seg.pt`
- **Baseline:** pretrained model van [harpreetsahota/car-dd-segmentation-yolov11](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11)
- **Dataset:** CarDD (Wang, Li & Wu, 2023) — 4.000 hoge-resolutie afbeeldingen, 9.000+ geannoteerde schade-instanties, 6 klassen: crack, dent, glass shatter, lamp broken, scratch, tire flat
- **Waarom YOLO-Seg:** combineert detectie + classificatie + segmentatie in één forward pass, snel genoeg voor een webapp, sterke pretrained weights voor eventuele verdere finetuning

### Performance van het baseline model (mask segmentation, op CarDD testset)

| Klasse | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Alle klassen | 0.827 | 0.749 | 0.792 | 0.576 |
| Crack | 0.665 | 0.483 | 0.518 | 0.214 |
| Dent | 0.697 | 0.560 | 0.612 | 0.344 |
| Glass shatter | 0.981 | 0.989 | 0.994 | 0.784 |
| Lamp broken | 0.921 | 0.902 | 0.967 | 0.808 |
| Scratch | 0.734 | 0.613 | 0.680 | 0.368 |
| Tire flat | 0.962 | 0.949 | 0.982 | 0.941 |

## Projectstructuur

```
vehicle-damage-ai/
├── .github/workflows/
│   └── ci.yml                 # Lint, syntax-check en unit tests bij elke push
├── app/
│   └── main.py                # Streamlit webapp
├── src/
│   ├── download_model.py      # Download het pretrained baseline model
│   ├── prepare_data.py        # Download CarDD + exporteer naar YOLO-formaat
│   ├── add_custom_data.py     # Voeg eigen gefotografeerde data toe
│   ├── train.py                # Finetune het model verder (transfer learning)
│   ├── evaluate.py             # Evalueer het eigen gefinetunede model
│   ├── visualize_failures.py   # Visuele inspectie van foutieve voorspellingen
│   ├── export_model.py         # Exporteer naar ONNX (deployment)
│   └── inference.py            # Herbruikbare detectie-functies
├── tests/
│   └── test_inference.py      # Unit tests (geen model/netwerk nodig)
├── notebooks/
│   └── 01_eda.ipynb           # EDA op de CarDD-data
├── data/
│   ├── raw/custom/             # Eigen geannoteerde data (niet in git)
│   └── processed/              # YOLO-geëxporteerde data (niet in git)
├── models/                     # Pretrained baseline weights (niet in git)
├── runs/                       # Trainingsresultaten/grafieken (niet in git)
├── requirements.txt
├── requirements-dev.txt        # + pytest, ruff
├── pyproject.toml              # Ruff/pytest-configuratie
└── README.md
```

## Installatie

```bash
# 1. Clone de repo
git clone <repo-url>
cd vehicle-damage-ai

# 2. Maak een virtuele omgeving
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Installeer dependencies
pip install -r requirements.txt

# 4. Download het baseline model
python src/download_model.py
```

## Volledige pipeline (reproduceerbaar)

### Aanbevolen route: Google Colab (heeft gratis GPU)

Open `notebooks/colab_full_pipeline.ipynb` in Google Colab en run 'm van boven naar beneden. Deze ene notebook doorloopt zelfstandig de hele pipeline: dataset downloaden → EDA → finetunen → evalueren → falen-analyse → ONNX-export, en slaat alles op naar Google Drive.

Na afloop kopieer je het resultaat (model + evaluatiecijfers + EDA-plots) naar je lokale project — de laatste cel van de notebook legt precies uit welke bestanden waar naartoe gaan.

### Alternatief: losse scripts (lokaal of los in Colab)

```bash
# 1. Baseline model downloaden
python src/download_model.py

# 2. Dataset downloaden en converteren naar YOLO-formaat
python src/prepare_data.py

# 3. EDA — open en run notebooks/01_eda.ipynb

# 3b. (optioneel) Eigen gefotografeerde/geannoteerde data toevoegen
python src/add_custom_data.py --source data/raw/custom

# 4. Finetunen (transfer learning vanaf het pretrained model)
python src/train.py

# 5. Eigen model evalueren
python src/evaluate.py

# 6. Falen-analyse: visuele inspectie van moeilijke gevallen
python src/visualize_failures.py

# 7. Exporteren naar ONNX voor efficiëntere inference/deployment
python src/export_model.py
```

Let op: stap 4 vereist een GPU voor redelijke trainingstijd. Aanbevolen: draai dit op Google Colab (gratis GPU) in plaats van lokaal op CPU. Upload hiervoor de `data/processed/` map of run `prepare_data.py` direct in Colab.

## Eigen data toevoegen

Naast de publieke CarDD-dataset kun je eigen gefotografeerde schade toevoegen om het model te specialiseren op jullie eigen praktijkcases:

1. Maak foto's van (nagebootste) schade
2. Annoteer met Roboflow of CVAT in YOLO-segmentatieformaat, met dezelfde 6 klassen als CarDD
3. Zet de export in `data/raw/custom/` (met `images/` en `labels/` submappen)
4. Run `python src/add_custom_data.py --source data/raw/custom`

## Kwaliteitsborging (CI)

Elke push/pull request draait automatisch (`.github/workflows/ci.yml`):

- Linting met ruff
- Syntax-check van alle Python-bestanden
- Validiteit van de notebooks
- Unit tests (`pytest tests/`) op de pure logica (bijv. oppervlakteberekening), zonder dat daar een model of netwerktoegang voor nodig is

## App online zetten (deployment)

Om de webapp via een publieke link beschikbaar te maken, gebruiken we Streamlit Community Cloud.

**Stappen**

1. Zorg dat je laatste werkende code (incl. `requirements.txt` en het model) gepusht staat naar de `main`-branch op GitHub.
2. Maak een gratis account op [share.streamlit.io](https://share.streamlit.io) door in te loggen met GitHub.
3. Klik "Create app" → kies "Deploy from existing repo".
4. Vul in:
   - Repository: `Lodieva/Claimcop`
   - Branch: `main`
   - Main file path: `app/main.py`
5. Klik "Deploy". Streamlit Cloud installeert automatisch alles uit `requirements.txt` en downloadt het model bij de eerste request dankzij `ensure_model_available()` in `app/main.py`.
6. Na een paar minuten krijg je een publieke URL zoals: `https://claimcop-gwns2y6znmrkqrfa2py5mt.streamlit.app/`

Elke nieuwe push naar `main` update de live app automatisch, zonder dat je opnieuw hoeft te deployen.

**Live app:** https://claimcop-gwns2y6znmrkqrfa2py5mt.streamlit.app/

## Model-optimalisatie (ONNX)

Naast app-hosting bestaat er ook model-optimalisatie: het model zelf kleiner/sneller maken voor inference. Dit versnelt alleen de inference zelf en publiceert geen app. Zie `src/export_model.py`.

## Lokaal draaien (voor ontwikkeling/testen)

**Webapp starten**

```bash
streamlit run app/main.py
```

Open vervolgens `http://localhost:8501` in je browser, upload een foto en bekijk de detectie.

**Command-line inference**

```bash
python src/inference.py data/processed/voorbeeldfoto.jpg
```

## Dataset

- **Bron:** CarDD (Wang, Li & Wu, 2023)
- 4.000 hoge-resolutie afbeeldingen, 9.000+ instanties, 6 schadecategorieën
- **Annotaties:** masks + bounding boxes in COCO-formaat
- **Licentie:** niet-commercieel onderzoek/educatie; onderliggende beelden vallen onder Flickr/Shutterstock-licenties (zie dataset-pagina voor details)

## Evaluatie

### Baseline (pretrained, ongewijzigd model)

Zie performance-tabel hierboven.

### Eigen gefinetuned model

*(Vul in na het runnen van `python src/evaluate.py` — output staat in `runs/eigen_evaluatie.md`, plak de tabel hieronder)*

| Klasse | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

### EDA-bevindingen

*(Vul in na `notebooks/01_eda.ipynb` — bijv. class balance, corrupte bestanden, resolutieverdeling)*

## Beperkingen

- Lagere recall op crack (48%) en dent (56%) — fijne/subtiele schade wordt vaker gemist
- Vuile auto's, regen of slechte belichting kunnen detectie bemoeilijken
- Verborgen schade (bijv. structurele schade) wordt niet herkend, alleen zichtbare schade
- Model is getraind op een specifieke dataset — generalisatie naar andere automerken/hoeken kan variëren
- Het systeem vervangt de schade-expert niet: het is een triage-tool voor de eerste inschatting, de uiteindelijke beslissing blijft mensenwerk

## Conclusie

### Praktische toepassing

Het systeem is bedoeld als triage-tool: een eerste, geautomatiseerde inschatting van schade zodra een claim binnenkomt, zodat schade-experts sneller kunnen prioriteren welke claims urgent/duidelijk zijn en welke handmatige beoordeling nodig hebben.

### Trade-offs

| Keuze | Voordeel | Nadeel |
|---|---|---|
| YOLOv11-Seg i.p.v. Mask R-CNN | Sneller, beter geschikt voor een live webapp | Iets minder precies op zeer kleine/fijne schade |
| Transfer learning i.p.v. training from scratch | Veel sneller, werkt met beperkte data/tijd | Model blijft afhankelijk van de bias in de originele CarDD-data |
| ONNX-export | Snellere/lichtere inference, geen volledige PyTorch-stack nodig | Extra exportstap, iets minder flexibel dan het originele model |
| Cloud-inference (huidige opzet) | Simpel te bouwen/demonstreren | Vereist internetverbinding; edge-deployment zou latency verder verlagen |

### Toekomstig werk

- Meer/betere data voor ondervertegenwoordigde klassen (met name crack, met de laagste recall)
- Fraudedetectie: controleren of dezelfde schade al eerder geclaimd is
- Controle op volledigheid van foto's (ontbrekende hoeken automatisch detecteren)
- LLM-integratie voor automatisch gegenereerde, leesbare schaderapporten
- Edge-deployment (bijv. op een tablet in de garage) met het ONNX-model

## Team

| Naam | Bijdrage |
|---|---|
| Pamari Lodiëva | ... |
| Roeplal Narisha | ... |

## Licentie

Dit project is gemaakt voor educatieve doeleinden (PTC-opleiding).


