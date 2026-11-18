
# AquaFlex AI — Automated Marine Plastic Debris Detection & Predictive Drift Simulation

> **Problem Statement 5:** AI/ML-Based Detection and Mapping of Floating Marine Plastic Debris — **MSC ELSA 3 Incident Case Study**.

AquaFlex AI replicates and automates the capabilities of INCOIS's SARAT and Oil/Nurdles Spill Trajectory System. It integrates open-access, high-spatial- and spectral-resolution satellite observations (**Sentinel-2**) with oceanographic and meteorological model products (**surface currents, winds, drift patterns**) to transform raw data into actionable intelligence for **field verification** and **targeted marine debris clean-up**.

---

## 1. Case Study Context

The Liberian-flagged container vessel **MSC ELSA 3**, travelling from **Vizhinjam to Kochi Port**, developed a severe starboard list on **24 May 2025** and sank on **25 May 2025**, approximately **13 nautical miles off the coast of Kerala**. Lost cargo included **plastic resin pellets (nurdles)**, which were subsequently reported along the Kerala and southern Tamil Nadu coasts, including **Kanyakumari** and **Dhanushkodi / Rameswaram**.

AquaFlex AI ingests Sentinel-2 imagery over the Kerala–Tamil Nadu maritime zone, runs a **YOLOv8** detector to localise floating debris and nurdles, computes **confidence scores** for every detection, and projects **72-hour drift trajectories** using a physics-based kinematic engine driven by INCOIS surface currents and a **3% wind-leeway factor**. It then flags **priority coastal risk zones** and drives an interactive **GIS dashboard** with a live **72-hour forecast impact countdown**.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Open-Access Data Sources                     │
│   Sentinel-2 (B04/B03/B02/B08)      INCOIS Currents + Winds (u,v)   │
└───────────────┬───────────────────────────────┬────────────────────┘
                │                                 │
        ┌───────▼────────┐               ┌────────▼─────────┐
        │ Ingestion Svc  │               │   Drift Engine    │
        │ (tiles/vectors)│               │  (kinematic PDE)  │
        └───────┬────────┘               └────────┬─────────┘
                │                                 │
        ┌───────▼────────┐                       │
        │ Detection Svc  │  YOLOv8 inference      │
        │ + confidence   │  + GeoJSON features    │
        └───────┬────────┘                       │
                │                                 │
                └─────────────┬───────────────────┘
                              │
                   ┌──────────▼───────────┐
                   │   FastAPI Backend     │
                   │  /api/v1/*            │
                   └──────────┬───────────┘
                              │  JSON
                   ┌──────────▼───────────┐
                   │  Leaflet GIS Dashboard│
                   │  (index.html + JS)    │
                   └───────────────────────┘
```

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| Ingestion | `rasterio`, `numpy`, `requests` | Sentinel-2 tiles + INCOIS current/wind grids |
| Detection | `ultralytics` (YOLOv8) | Debris localisation + classification + confidence |
| Drift | `numpy` physics engine | 24/48/72h trajectory & dispersion |
| API | `fastapi`, `uvicorn`, `pydantic` | REST endpoints + validation |
| Dashboard | Leaflet.js, Tailwind CSS | Interactive GIS + countdown timer |

---

## 3. Directory Structure

```
aquaflex-ai/
├── .env.example
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + REST endpoints
│   ├── config.py               # Env config + case-study bounding boxes
│   ├── models/

---

## 4. Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- Optional: **Docker** & **Docker Compose** for containerised deployment
- A modern web browser (Chrome / Edge / Firefox) for the dashboard

> **Note:** By default the platform runs in **mock mode** (`USE_MOCK_SATELLITE=true`), so it works fully offline with deterministic synthetic Sentinel-2 tiles and INCOIS vector fields — no credentials or GPU required. The YOLOv8 model auto-downloads `yolov8n.pt` on first run (a lightweight fallback detector is used if the model or `ultralytics` is unavailable).

---

## 5. Local Setup

### 5.1 Clone & create a virtual environment

```bash
cd aquaflex-ai
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Configure environment

```bash
# Copy the template and adjust as needed
cp .env.example .env
```

Key variables (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_SATELLITE` | `true` | Use deterministic synthetic data (offline-safe) |
| `YOLO_MODEL` | `yolov8n.pt` | Ultralytics model path/id |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence |
| `WIND_LEEWAY_FACTOR` | `0.03` | 3% wind-leeway applied in the drift engine |
| `FORECAST_HORIZON_HOURS` | `72` | Drift forecast horizon |
| `CORS_ORIGINS` | localhost origins | Allowed dashboard origins |

### 5.4 Launch the backend

From the **project root** (`aquaflex-ai/`):

```bash
uvicorn backend.main:app --reload
```

The API is now available at `http://127.0.0.1:8000`.
Interactive API docs (Swagger UI): **http://127.0.0.1:8000/docs**

### 5.5 Serve the frontend dashboard

The dashboard is a static single-page app. Serve the `frontend/` directory with any static server, e.g.:

```bash
# Option A: Python
python -m http.server 5500 --directory frontend

# Option B: VS Code "Live Server" extension (right-click index.html → Open with Live Server)
```

Then open **http://127.0.0.1:5500** (or the Live Server URL). The dashboard fetches live data from `http://127.0.0.1:8000/api/v1/simulation/msc-elsa-3`.

> The dashboard auto-detects the API base URL. When opened via `file://`, it defaults to `http://127.0.0.1:8000`. You can override it by setting `window.AQUAFLEX_API_BASE` before the script loads.

> **Note:** The backend also serves the dashboard itself — open **http://127.0.0.1:8000** and you get the same page plus the API from a single origin (no CORS/caching issues).

### 5.6 U-Net segmentation model

AquaFlex AI includes a **trained PyTorch U-Net** for pixel-level marine-debris
segmentation (`backend/services/unet_service.py`), trained on the **MARIDA**
dataset with **11 Sentinel-2 spectral bands**, **256×256** patches, and
**BCE + Dice** loss. Artifacts:

| File | Purpose |
|------|---------|
| `models/aquaflex_unet_final.pth` | Trained U-Net weights (state_dict) |
| `models/aquaflex_model_config.json` | Band order, per-band normalization mean/std, input size (256), input channels (11), threshold (0.6) |

The architecture (`DoubleConv` blocks; encoder 11→32→64→128; bottleneck
128→256; decoder transposed-conv 256→128→64→32 with skip connections; 1×1
output conv) **exactly matches the checkpoint** and is loaded with
`load_state_dict(strict=True)`, so a mismatch fails loudly instead of silently.

**Requirements to run it:** `torch` and `Pillow` (added to `requirements.txt`),
plus `rasterio` to read GeoTIFF inputs. These are imported lazily inside the
endpoint, so the rest of the backend still starts if they are absent.

#### `POST /api/v1/detect/unet`

Accepts a **multipart** upload of an **11-band, 256×256 multispectral GeoTIFF**
(`.tif`/`.tiff`) and returns a JSON `UNetDetectResponse`:

- `model_name`, `framework`, `threshold` (from config, 0.6)
- `input_channels` (11) and `band_order`
- `width`, `height`, `total_pixels`, `debris_pixels`, `debris_percent`
- `mask_png_base64` — the binary segmentation mask as a base64 grayscale PNG
  (0 = background, 255 = debris), ready to drop into an `<img>` tag.

**Input validation & errors:** non-GeoTIFF extensions → `400`; RGB (3-band) or
wrong band count → `400`; non-256×256 patch → `400`; uploads larger than
256 MB → `413`; U-Net/model unavailable (missing deps/artifacts) → `503`.
Uploaded files are streamed to a temp file, size-checked, and deleted in a
`finally` block.

**Example (curl / PowerShell):**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/detect/unet \
  -F "file=@patch_11band_256x256.tif"
```

> ⚠️ **Important — do not feed RGB to the U-Net.** The RGB Sentinel-2 simulation
> PNG produced by the ingestion service is a **3-band** composite and is
> incompatible with this 11-band model. RGB rejection is intentional and tested.
> The U-Net requires genuine 11-band multispectral data.

#### Known limitation — live satellite integration

The ingestion service (`ingestion_service.py`) currently returns **synthetic /
mock** Sentinel-2 RGB tiles and mock current/wind fields (offline-safe demo
data). It does **not** yet fetch real 11-band Sentinel-2 products from
Copernicus, nor live INCOIS currents. The live-fetch code paths are scaffolds
that require credentials / a concrete STAC download implementation. Therefore
the U-Net endpoint expects a **user-supplied 11-band GeoTIFF**; end-to-end
automated satellite→U-Net ingestion is **not yet implemented or tested**.

#### Running the smoke test

```bash
# From the project root:
py backend\tests\run_smoke.py        # Windows
python backend/tests/run_smoke.py    # cross-platform
pytest backend/tests/run_smoke.py    # if pytest is installed
```

The test builds a synthetic 11-band GeoTIFF in a temp dir (test fixture only)
and checks: the app starts, `/openapi.json` lists the route, the checkpoint
loads, a valid 11-band tile yields a mask, RGB is rejected, and the existing
YOLO `/detect` and `/simulation` endpoints still work.

---

---

## 7. Expected Outcomes — Implementation Mapping

| # | Expected Outcome | Where Implemented |
|---|------------------|-------------------|
| 1 | **AI/ML Detection & Classification** — YOLOv8 detects & classifies floating plastic debris / nurdles | `backend/services/detection_service.py` (`DetectionService.detect_debris`) |
| 2 | **Geo-referenced Mapping & Hotspots** — spatial extent & concentration zones | `detection_service.py` (GeoJSON features) + `main.py::_build_hotspots` + `Hotspot` schema |
| 3 | **Detection Confidence Scores** — quantitative confidence per feature | `DebrisDetection.confidence_percent` in `backend/models/schemas.py` |
| 4 | **INCOIS Data Integration** — currents (u,v) + winds, physics-based trajectory | `backend/services/drift_engine.py` + `ingestion_service.py` |
| 5 | **Priority Action Map** — coastal zones (Kanyakumari, Rameswaram) for clean-up | `drift_engine.py::build_priority_risk_zones` + `PriorityRiskZone` schema |
| 6 | **Interactive GIS Dashboard** — imagery, hotspots, confidence, vectors, 72h countdown | `frontend/index.html`, `frontend/js/dashboard.js`, `frontend/css/styles.css` |

### Drift Physics
Each timestep advects a debris particle by the total surface drift velocity:

```
u_total = u_current + leeway · u_wind        (m/s)
v_total = v_current + leeway · v_wind        (m/s)
Δlat = v_total · Δt / 111320
Δlon = u_total · Δt / (111320 · cos(lat))
```

The **leeway factor (3%)** accounts for direct wind forcing on partially-submerged debris. Positions are sampled at **24h, 48h, and 72h**, and each node is assigned a risk severity based on proximity to the Kanyakumari / Rameswaram coasts (haversine distance thresholds).

---

## 8. Docker Deployment

Build and run the full stack with Docker Compose:

```bash
docker compose up --build
```

This launches the FastAPI backend on **http://localhost:8000**. Mount `./frontend` (or the container's static mount) in your browser to view the dashboard, or serve `frontend/` separately.

Stop the stack:

```bash
docker compose down
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dashboard shows "Unable to reach backend" | Ensure `uvicorn backend.main:app --reload` is running on port 8000 |
| `model_loaded: false` in `/health` | `ultralytics`/`yolov8n.pt` unavailable — the fallback detector is used automatically |
| CORS errors in the browser console | Add your dashboard origin to `CORS_ORIGINS` in `.env` |
| `rasterio` install issues on Windows | Install via conda (`conda install -c conda-forge rasterio`) or use Docker |

---

## 10. License & Acknowledgements

- Satellite data: **Copernicus Sentinel-2** (open access).
- Ocean/meteorological products: **INCOIS** (Indian National Centre for Ocean Information Services).
- Object detection: **Ultralytics YOLOv8**.
- Mapping: **Leaflet.js** with **Esri** Dark Gray base tiles.

*Built for automated marine debris response in support of the MSC ELSA 3 incident.*


## 6. REST API Reference

### `GET /api/v1/health`
System status check.

```json
{
  "status": "ok",
  "app_name": "AquaFlex AI",
  "version": "1.0.0",
  "environment": "development",
  "model_loaded": true,
  "timestamp": "2025-05-25T00:00:00"
}
```

### `POST /api/v1/detect`
Run debris detection on an uploaded satellite tile (`multipart/form-data`).

- **`file`**: image file (PNG/JPEG/GeoTIFF)
- **`bbox`** *(optional)*: `"min_lon,min_lat,max_lon,max_lat"` for geo-referencing detections

Returns a `DetectionResponse` containing every detection with its bounding box, class label, `confidence`, `confidence_percent`, and a GeoJSON `FeatureCollection`.

**Example (curl):**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/detect \
  -F "file=@tile.png" \
  -F "bbox=76.3,8.0,77.3,9.0"
```

### `GET /api/v1/simulation/msc-elsa-3`
Returns the complete pre-configured MSC ELSA 3 case-study dataset: initial AI detections with confidence metrics, aggregated hotspots, the 72-hour drift trajectory, and the priority cleanup risk zones (Kanyakumari, Rameswaram). This endpoint powers the GIS dashboard.

│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── ingestion_service.py  # Sentinel-2 + INCOIS data
│       ├── detection_service.py  # YOLOv8 detector
│       └── drift_engine.py       # Physics-based drift simulation
└── frontend/
    ├── index.html              # GIS dashboard (dark mode)
    ├── css/
    │   └── styles.css
    └── js/
        └── dashboard.js        # Leaflet map + countdown
```

# marine
AquaFlex AI: An intelligent marine debris detection and ocean plastic tracking platform developed for VJ Hackathon 2k26 to monitor and map coastal litter using advanced data processing and GIS visualization.

