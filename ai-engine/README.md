# DEFECTX AI - Industrial Anomaly Detection Engine

> **CLOCKIT 2026 Hackathon**  
> **Role:** AI / ML Subsystem (`ai-engine/`)  
> **Target Problem:** Open-Set Industrial Surface Anomaly Detection on MVTec AD (`bottle`)

---

## 1. Architecture Overview

DEFECTX AI is an unsupervised, open-set visual anomaly detection engine inspired by **PatchCore**. It trains strictly on **normal industrial images** (`train/good`), detects previously unseen defect categories (cracks, contamination, punctures, structural deformities), localizes defect regions via pixel heatmaps, estimates severity heuristics, and provides defensible confidence metadata.

```
                    ┌─────────────────────────┐
                    │ Normal Production Image │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     Pretrained ResNet-18
                   (Frozen layers 2 & 3)
                                 │
                                 ▼
                   Local Neighborhood Pooling
                  (Spatial Context Aggregation)
                                 │
                                 ▼
                     Patch Embedding Matrix
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
      [Training Pipeline]               [Inference Pipeline]
       Normal Memory Bank               Nearest-Neighbor Search
      + Coreset Subsampling               (Chunked L2 Distance)
                │                                 │
                ▼                                 ▼
    Validation Normal Holdout           Patch & Image Anomaly Scores
   (Zero-Leakage Thresholding)          + Bilinear Gaussian Heatmap
                │                                 │
                └────────────────┬────────────────┘
                                 ▼
                     DefectX Decision Engine
           (Prediction, Severity, Calibrator Interface)
                                 │
                                 ▼
               Clean JSON Output for PHP Backend
```

### Key Methodological Components:
1. **Feature Extractor:** PyTorch pretrained `ResNet-18` extracting mid-level feature representations from `layer2` (128 channels) and `layer3` (256 channels), concatenated with local average pooling to incorporate surrounding context into each patch token (384 dimensions).
2. **Normal Memory Bank:** Stores patch embeddings from normal training images. Employs greedy minimax k-center coreset subsampling to reduce memory footprint by 90% while retaining spatial boundary coverage.
3. **Thresholding (Zero-Leakage):** Thresholds are fit **strictly** on a held-out normal validation split (`val/good`). It never tunes on test defect categories.
4. **Heuristic Severity:** Categorizes anomalies into `NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` based on anomalous area percentage and peak intensity deviation. *(Explicitly documented as an engineering estimate prototype, not a certified physical damage measurement).*
5. **Defensible Confidence:** Raw nearest-neighbor distance is **not** a probability. When calibration is unvalidated, `confidence` returns `null` in the JSON response, with a dedicated evaluation-ready interface (`ConfidenceEvaluator`) reserved for Member 4's distribution shift calibration.

---

## 2. NVIDIA RTX 4050 Hardware Constraints & Optimization

The AI engine is optimized specifically for laptop GPUs such as the **NVIDIA GeForce RTX 4050 (6GB VRAM)**:
- **Conservative Image Resolution:** Uses 224x224 input tensors ($28 \times 28 = 784$ patches per image).
- **Chunked Nearest Neighbor Search:** Distance computation is chunked (`chunk_size=512`) to prevent quadratic memory allocation spikes.
- **Inference Mode & Frozen Weights:** Executes under `torch.inference_mode()` with frozen parameters (`requires_grad=False`), eliminating backward computational graphs.
- **Dynamic VRAM Monitoring & CPU Fallback:** Automatically detects GPU device status and falls back gracefully to CPU if CUDA is unavailable.

---

## 3. Directory Structure

```
ai-engine/
├── config/
│   └── default_config.yaml         # Configurable hyperparameters
├── src/
│   ├── dataset.py                  # MVTec AD loader with zero-leakage splits & GT masks
│   ├── dataset_adapter.py          # Configurable dataset adapters (MVTec, Custom, Auto-detect) & validation
│   ├── feature_extractor.py        # Pretrained ResNet-18 patch embedding hook
│   ├── memory_bank.py              # Normal feature bank, coreset subsampling, L2 distance
│   ├── anomaly_map.py              # Bilinear interpolation, Gaussian smoothing, colormap overlay
│   ├── thresholding.py             # Normal-only percentile/IQR threshold selector
│   ├── severity.py                 # Heuristic severity prototype (NONE/LOW/MEDIUM/HIGH/CRITICAL)
│   ├── confidence.py               # Defensible confidence interface for Member 4
│   ├── model.py                    # DefectXEngine orchestrator (fit, predict, checkpointing)
│   ├── evaluator.py                # AUROC, precision, recall, F1, FPR, ROC plots
│   └── utils.py                    # Device selection, seed, logging, VRAM telemetry
├── scripts/
│   ├── generate_sample_data.py     # Generates synthetic MVTec bottle data for tests
│   └── download_mvtec_bottle.py    # Official MVTec bottle dataset downloader
├── tests/
│   ├── test_feature_extractor.py   # Unit test for patch embedding
│   ├── test_memory_bank.py         # Unit test for coreset subsampling & distances
│   ├── test_severity_confidence.py # Unit test for heuristic severity & confidence
│   ├── test_dataset.py             # Unit test for zero-leakage dataset loader
│   ├── test_dataset_adapter.py     # Unit test for dataset inspection, validation, & custom adapters
│   └── test_smoke.py               # Full end-to-end integration & contract smoke test
├── inspect_dataset.py              # CLI tool for inspecting and validating organiser datasets
├── train.py                        # Model training CLI
├── predict.py                      # Production inference CLI for PHP backend
├── evaluate.py                     # Model evaluation CLI & metrics report
├── api.py                          # Programmatic Python API and local HTTP microservice
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation & backend contract
```

---

## 4. Setup & Windows PowerShell Commands

### Step 1: Install Dependencies
Open Windows PowerShell inside `ai-engine/`:
```powershell
# Install required Python packages
python -m pip install -r requirements.txt
```

### Step 2: Prepare Dataset
You have two options:
**Option A: Generate Sample Industrial Bottle Dataset (Immediate, Zero Download, 100% Offline):**
```powershell
python scripts/generate_sample_data.py --output-dir data/mvtec --num-train 25
```
**Option B: Download Official MVTec Bottle Dataset (~125 MB):**
```powershell
python scripts/download_mvtec_bottle.py --dest-dir data/mvtec
```

### Step 2b: Inspect & Validate Dataset (Organiser Dataset Ready)
Run the dataset inspection utility on any dataset directory to analyze image counts, formats, dimensions, defect classes, ground-truth masks, and detect corrupted files or label inconsistencies:
```powershell
# Inspect default MVTec bottle dataset
python inspect_dataset.py --dataset-root data/mvtec --category bottle

# Inspect future organiser dataset (custom layout or auto-detected)
python inspect_dataset.py --dataset-root "path/to/organiser_dataset" --adapter auto

# Output machine-readable JSON report
python inspect_dataset.py --dataset-root data/mvtec --category bottle --json --output-report results/dataset_inspection.json
```

### Step 3: Run Unit & Smoke Tests
Verify all modules and the end-to-end pipeline:
```powershell
python -m unittest discover -s tests -v
```

### Step 4: Train the Engine
Trains ResNet-18 memory bank on `train/good` normal bottles and fits thresholds on `val/good`:
```powershell
python train.py --config config/default_config.yaml --category bottle --device cuda
```
*Note: Pass `--device cpu` if running on a system without an NVIDIA GPU.*

### Step 5: Evaluate on Held-Out Test Defect Categories
Evaluates Image AUROC, Pixel AUROC, precision, recall, F1, and FPR:
```powershell
python evaluate.py --config config/default_config.yaml --checkpoint checkpoints/defectx_bottle.pt --device cuda
```

### Step 6: Single-Image Inference CLI
```powershell
python predict.py --image data/mvtec/bottle/test/broken_large/000.png --checkpoint checkpoints/defectx_bottle.pt --device cuda
```

---

## 5. Integration Contract for PHP Backend Developer

The PHP backend communicates with DEFECTX AI via **CLI Execution** (recommended for production worker jobs) or **Local HTTP Microservice**.

### Standard Prediction JSON Schema

The CLI prints strictly valid JSON to `stdout` upon completion:

```json
{
  "inspection_id": "insp_20261125_144025_9a1f2b3c",
  "prediction": "anomaly",
  "anomaly_score": 4.1829,
  "severity": "CRITICAL",
  "confidence": null,
  "heatmap_path": "C:\\Users\\...\\DefectX-AI\\ai-engine\\results\\heatmaps\\insp_20261125_144025_9a1f2b3c_000_heatmap.png",
  "overlay_path": "C:\\Users\\...\\DefectX-AI\\ai-engine\\results\\heatmaps\\insp_20261125_144025_9a1f2b3c_000_overlay.png",
  "model_version": "1.0.0",
  "processing_time_ms": 14.85,
  "shift_detected": null,
  "details": {
    "image_path": "C:\\Users\\...\\DefectX-AI\\ai-engine\\data\\mvtec\\bottle\\test\\broken_large\\000.png",
    "image_threshold": 0.6366,
    "pixel_threshold": 0.5060,
    "anomalous_area_ratio": 0.368668,
    "severity_index": 0.737335,
    "severity_disclaimer": "Heuristic visual estimate prototype only. Not a verified physical or structural damage measurement.",
    "device": "cuda:0"
  }
}
```

### JSON Fields Explained:
- `inspection_id` (`string`): Unique identifier generated for every inspection (`insp_<timestamp>_<uuid>`). Guarantees no overwrites even if files have identical names across runs.
- `prediction` (`string`): `"normal"` or `"anomaly"`.
- `anomaly_score` (`float`): Image-level nearest-neighbor distance.
- `severity` (`string`): Categorized heuristic rating: `"NONE"`, `"LOW"`, `"MEDIUM"`, `"HIGH"`, or `"CRITICAL"`.
- `confidence` (`null` or `float`): Defaults to `null` because nearest-neighbor distance is not calibrated. Reserved for Member 4's calibration validator.
- `heatmap_path` (`string`): Absolute path to the saved JET colormap anomaly localization PNG.
- `overlay_path` (`string`): Absolute path to the saved blended heatmap + original image PNG.
- `model_version` (`string`): Model SemVer (e.g., `"1.0.0"`).
- `processing_time_ms` (`float`): Inference latency in milliseconds.
- `shift_detected` (`null`): Reserved for Member 4's distribution shift detector (do not assume false).

---

### Exit Codes & Error Handling Contract

When an error occurs, `predict.py` prints a clean JSON error response to `stdout`, logs debugging context to `stderr`, and terminates with a non-zero exit code:

| Exit Code | `error_code` | Cause | Action for PHP Backend |
| :---: | :--- | :--- | :--- |
| **0** | `None` | Inference Success | Parse prediction results from JSON. |
| **1** | `MISSING_IMAGE` | Image path does not exist on disk. | Verify file upload/storage path. |
| **2** | `MISSING_CHECKPOINT` | Model checkpoint file not found. | Verify model checkpoint path or train model. |
| **3** | `INVALID_IMAGE` | Corrupted image file or unsupported extension (allowed: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`). | Reject upload with client-side 400 error. |
| **4** | `INFERENCE_ERROR` | Model loading or PyTorch execution failure. | Check GPU/RAM telemetry or fallback to CPU. |

#### Error JSON Response Example (Exit Code > 0):
```json
{
  "error": "Input image not found: 'uploads/missing.png'",
  "error_code": "MISSING_IMAGE",
  "prediction": null,
  "anomaly_score": null,
  "severity": null,
  "confidence": null,
  "heatmap_path": null,
  "overlay_path": null,
  "model_version": null,
  "processing_time_ms": 0.0,
  "shift_detected": null
}
```

---

### Health Checking Contract (`--health`)

The PHP backend can perform an automated health check before accepting uploads:

```powershell
python predict.py --health --checkpoint checkpoints/defectx_bottle.pt
```

Returns:
```json
{
  "status": "healthy",
  "checkpoint_exists": true,
  "checkpoint_path": "C:\\Users\\...\\DefectX-AI\\ai-engine\\checkpoints\\defectx_bottle.pt",
  "config_exists": true,
  "model_loadable": true,
  "model_version": "1.0.0",
  "device": "cuda:0",
  "cuda_available": true,
  "vram": {
    "allocated_mb": 0.0,
    "reserved_mb": 0.0,
    "max_allocated_mb": 153.9
  },
  "image_threshold": 0.6366,
  "pixel_threshold": 0.5060
}
```

---

### PHP Integration Examples

#### Option 1: Robust Execution via `proc_open` (Separating stdout & stderr)
```php
<?php
function runDefectXInference(string $imagePath, string $engineDir): array {
    $pythonBin = 'python';
    $script = $engineDir . DIRECTORY_SEPARATOR . 'predict.py';
    $checkpoint = $engineDir . DIRECTORY_SEPARATOR . 'checkpoints' . DIRECTORY_SEPARATOR . 'defectx_bottle.pt';

    // Safely quote arguments to protect against injection and handle file paths containing spaces
    $cmd = sprintf(
        '%s %s --image %s --checkpoint %s',
        escapeshellcmd($pythonBin),
        escapeshellarg($script),
        escapeshellarg($imagePath),
        escapeshellarg($checkpoint)
    );

    $descriptors = [
        0 => ['pipe', 'r'], // stdin
        1 => ['pipe', 'w'], // stdout (clean JSON)
        2 => ['pipe', 'w']  // stderr (logs/debug)
    ];

    $process = proc_open($cmd, $descriptors, $pipes, $engineDir);

    if (!is_resource($process)) {
        throw new RuntimeException("Failed to launch DefectX AI engine process.");
    }

    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);

    $exitCode = proc_close($process);

    $json = json_decode($stdout, true);
    if ($json === null) {
        throw new RuntimeException("AI Engine invalid JSON output. Stderr: " . $stderr);
    }

    if ($exitCode !== 0) {
        $errorCode = $json['error_code'] ?? 'UNKNOWN_ERROR';
        throw new RuntimeException("AI Engine error [{$errorCode}]: " . ($json['error'] ?? $stderr));
    }

    return $json;
}

// Example usage:
try {
    $result = runDefectXInference('C:/uploads/inspected bottle.png', 'C:/Users/.../DefectX-AI/ai-engine');
    echo "Inspection ID: " . $result['inspection_id'] . "\n";
    echo "Decision:      " . $result['prediction'] . "\n";
    echo "Severity:      " . $result['severity'] . "\n";
    echo "Heatmap saved: " . $result['heatmap_path'] . "\n";
} catch (Exception $e) {
    echo "Inspection Failed: " . $e->getMessage() . "\n";
}
?>
```

#### Option 2: Lightweight Execution via `shell_exec`
```php
<?php
$imagePath = 'C:/uploads/sample bottle.png';
$predictScript = 'C:/Users/.../DefectX-AI/ai-engine/predict.py';
$checkpoint = 'C:/Users/.../DefectX-AI/ai-engine/checkpoints/defectx_bottle.pt';

$command = "python " . escapeshellarg($predictScript) .
           " --image " . escapeshellarg($imagePath) .
           " --checkpoint " . escapeshellarg($checkpoint);

$output = shell_exec($command);
$result = json_decode($output, true);

if ($result && !isset($result['error'])) {
    echo "Prediction: " . $result['prediction'] . " (Score: " . $result['anomaly_score'] . ")\n";
    echo "Heatmap:    " . $result['heatmap_path'] . "\n";
} else {
    echo "Error: " . ($result['error'] ?? 'Execution failed') . "\n";
}
?>
```

---

## 6. Python API Usage

Teammates or Python services can also import and use `predict()` directly:

```python
from predict import predict, check_health

# 1. Health check
status = check_health(checkpoint_path="checkpoints/defectx_bottle.pt")
print(status["status"])  # "healthy"

# 2. Run inference
result = predict(
    image_path="data/mvtec/bottle/test/broken_large/000.png",
    checkpoint_path="checkpoints/defectx_bottle.pt"
)
print(result["inspection_id"])
print(result["prediction"])      # "anomaly"
print(result["severity"])        # "CRITICAL"
print(result["heatmap_path"])    # Absolute path to non-overwritten heatmap
```

---

## 7. Integration Status Note

> **Note on End-to-End Integration:**  
> The Python AI Engine interface, CLI, health checks, and error contracts are fully implemented and verified with 100% test pass rate inside `ai-engine/`. Full end-to-end integration with the application will be marked complete once the PHP team connects and tests their backend controllers in `backend/` against this CLI/API contract.

---

## 8. Documented Limitations & Quality Safeguards

1. **Open-Set Anomaly Assumption:** The engine trains exclusively on normal images. Microscopic flaws with anomaly scores below the normal variance distribution cannot be guaranteed detectable.
2. **Severity Is a Visual Heuristic:** The severity metric measures visual feature deviation and anomalous surface area. It must **not** be presented to regulatory bodies as a physical strength or chemical containment rating.
3. **Distribution Shift:** Changes in camera optics, ambient factory lighting, or conveyor speed can elevate baseline anomaly scores. Member 4 is assigned to develop the distribution shift detector.
