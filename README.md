# Car Damage Detection
Met dit project onderzoeken we hoe je met computer vision autoschade automatisch kunt herkennen en inschatten op basis van foto's. Het doel is simpel: het eerste deel van een schadeclaim sneller en regelmatiger laten verlopen.

Belangrijk om te noemen: de software neemt het werk van een schade-expert niet over. Zie het puur als een assistent die alvast een eerste inschatting maakt, waarna de expert zelf de knoop doorhakt.

## Probleemstelling
Het handmatig beoordelen van autoschade is tijdrovend, moeilijk schaalbaar en kan leiden tot inconsistente beoordelingen.

Onze Onderzoeksvraag:
Kan computer vision autoschade automatisch detecteren en visualiseren om schadeclaims sneller en consistenter te ondersteunen?

### Praktische toepassing

Het systeem is bedoeld als triage-tool: een eerste, geautomatiseerde inschatting van schade zodra een claim binnenkomt, zodat schade-experts sneller kunnen prioriteren welke claims urgent/duidelijk zijn en welke handmatige beoordeling nodig hebben.

## Doelgroep
Primaire Doelgroep: Verzekeringsmaatschappijen
Secundaire Toepassingen:  Leasebedrijven, Autoverhuur, Schadeherstel, Autodealers

## Wat doet de software precies?
We maken gebruik van **instance segmentation**. In plaats van alleen een rechthoek (bounding box) om de schade te tekenen, legt het model een strak masker over de exacte contouren van de beschadiging.

Hiermee kunnen we verschillende soorten schade los van elkaar herkennen:
Krassen, Deuken, Scheuren, Gebroken glaselementen / lampen, Lekke banden

Een groot voordeel van deze aanpak: doordat je de exacte omtrek van de schade hebt, kun je ook een directe schatting maken van het beschadigde oppervlak in procenten.

## Het Model & De Data

- **Architectuur:** YOLOv11x-Seg (Ultralytics), finetuned op basis van `yolo11x-seg.pt`
- **Baseline:** pretrained model van [harpreetsahota/car-dd-segmentation-yolov11](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11)
- **Dataset:** CarDD (Wang, Li & Wu, 2023) — 4.000 hoge-resolutie afbeeldingen, 9.000+ geannoteerde schade-instanties, 6 klassen: crack, dent, glass shatter, lamp broken, scratch, tire flat
- **Waarom YOLO-Seg:** combineert detectie + classificatie + segmentatie in één forward pass, snel genoeg voor een webapp, sterke pretrained weights voor eventuele verdere finetuning

**Dataset**
- **Bron:** CarDD (Wang, Li & Wu, 2023)
- 4.000 hoge-resolutie afbeeldingen, 9.000+ instanties, 6 schadecategorieën
- **Annotaties:** masks + bounding boxes in COCO-formaat
- **Licentie:** niet-commercieel onderzoek/educatie; onderliggende beelden vallen onder Flickr/Shutterstock-licenties (zie dataset-pagina voor details)


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

### Baseline (pretrained, ongewijzigd model)
Zie performance-tabel hierboven.


## Installatie

```bash
# 1. Clone de repo
git clone <repo-url>

# 2. Maak een virtuele omgeving
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Installeer dependencies
pip install -r requirements.txt

# 4. Download het baseline model
python src/download_model.py

```
## Lokaal draaien (voor ontwikkeling/testen)

**Webapp starten**

```bash
streamlit run app/main.py
```

Open vervolgens `http://localhost:8501` in je browser, upload een foto en bekijk de detectie.

## Live app
url: https://claimcop-gwns2y6znmrkqrfa2py5mt.streamlit.app/

## Hoe gebruik je de Streamlit-app?

1. **Upload een foto:** Klik op de uploadknop in de app en kies een duidelijke foto van de autoschade (JPG/PNG).
2. **Analyse:** Het YOLO-model verwerkt de foto en plaatst een gekleurd masker over de gedetecteerde schade.
3. **Bekijk de resultaten:** De app toont de categorie (bijv. *deuk* of *kras*), de confidence score en het geschatte schadeoppervlak.
4. **Verwerk de claim:** Gebruik het overzicht voor een snelle eerste triage van de schadeclaim.


## Beperkingen

- Lagere recall op crack (48%) en dent (56%) — fijne/subtiele schade wordt vaker gemist
- Vuile auto's, regen of slechte belichting kunnen detectie bemoeilijken
- Verborgen schade (bijv. structurele schade) wordt niet herkend, alleen zichtbare schade
- Model is getraind op een specifieke dataset — generalisatie naar andere automerken/hoeken kan variëren
- Het systeem vervangt de schade-expert niet: het is een triage-tool voor de eerste inschatting, de uiteindelijke beslissing blijft mensenwerk

### Toekomstig werk

- Finetunen van het pretrained model op de vlakken waar het minder sterk is en dus verder trainen met eigen date
- Voertuigonderdeel detectie
- Schade-ernst schatting
- Kostenindicatie koppeling
- Mobiele inspectie-app


## Team

| Naam | Bijdrage |
|---|---|
| Pamari Lodiëva |  Documentatie ontwikkeling , bugfixer, live brengen van de app |
| Roeplal Narisha | Onderwerp bedenker , initieel developer ClaimCop app |

## Licentie

Dit project is gemaakt voor educatieve doeleinden (PTC-opleiding) voor de module Advance Computer Vision.





