<?php
/**
 * DEFECTX AI - Real-Time Inspection Ingestion & Python AI Inference Gateway
 * CLOCKIT 2026 Hackathon
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/db.php';

setApiHeaders();

if (($_SERVER['REQUEST_METHOD'] ?? '') === 'GET' && !empty($_GET['id'])) {
    $item = DefectXDatabase::getInspectionById($_GET['id']);
    if ($item) {
        echo json_encode($item, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    } else {
        http_response_code(404);
        echo json_encode(['error' => 'Inspection record not found.'], JSON_PRETTY_PRINT);
    }
    exit;
}

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed. Use POST.'], JSON_PRETTY_PRINT);
    exit;
}

// 0. Resolve Product Category and Checkpoint
// Category is optional: missing category defaults to 'bottle' for legacy backward compatibility.
// Explicit unsupported category returns HTTP 400.
// Registered category with missing/untrained checkpoint returns HTTP 503.
$requestedCategory = $_POST['category'] ?? null;
$categoryResolution = resolveCategoryCheckpoint($requestedCategory);

if (!$categoryResolution['success']) {
    http_response_code($categoryResolution['status_code']);
    echo json_encode([
        'error' => $categoryResolution['error'],
        'category' => $categoryResolution['category'],
        'code' => ($categoryResolution['status_code'] === 503) ? 'MODEL_NOT_AVAILABLE' : 'INVALID_CATEGORY',
        'message' => 'INSPECTION FAILED – ' . (($categoryResolution['status_code'] === 503) ? 'MODEL NOT AVAILABLE' : 'INVALID CATEGORY')
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

$resolvedCategory = $categoryResolution['category'];
$resolvedCheckpoint = $categoryResolution['checkpoint'];
$resolvedCategoryInfo = $categoryResolution['category_info'];

$rawFilename = 'mobile_capture.png';
$tempPath = null;

// 1. Process Image Ingestion (Multipart File or Base64 Data URL)
if (isset($_FILES['image']) && is_uploaded_file($_FILES['image']['tmp_name'])) {
    $file = $_FILES['image'];
    $rawFilename = basename($file['name']);
    
    // Validate file size (15MB limit)
    if ($file['size'] > 15 * 1024 * 1024) {
        http_response_code(400);
        echo json_encode([
            'error' => 'File size exceeds 15MB limit.',
            'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
        ]);
        exit;
    }

    $ext = strtolower(pathinfo($rawFilename, PATHINFO_EXTENSION));
    if (!in_array($ext, ['png', 'jpg', 'jpeg', 'webp', 'bmp'])) {
        http_response_code(400);
        echo json_encode([
            'error' => 'Unsupported image format. Allowed: PNG, JPG, WEBP, BMP.',
            'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
        ]);
        exit;
    }

    $uniqueName = 'capture_' . date('Ymd_His') . '_' . substr(md5(uniqid()), 0, 8) . '.' . $ext;
    $targetPath = UPLOAD_DIR . DIRECTORY_SEPARATOR . $uniqueName;

    if (!move_uploaded_file($file['tmp_name'], $targetPath)) {
        http_response_code(500);
        echo json_encode([
            'error' => 'Failed to save uploaded image.',
            'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
        ]);
        exit;
    }
} elseif (!empty($_POST['image_data'])) {
    $dataUri = $_POST['image_data'];
    if (!empty($_POST['filename'])) {
        $rawFilename = basename($_POST['filename']);
    }

    // Decode base64
    if (preg_match('/^data:image\/(\w+);base64,/', $dataUri, $matches)) {
        $ext = strtolower($matches[1]);
        if ($ext === 'jpeg') $ext = 'jpg';
        $base64Data = substr($dataUri, strpos($dataUri, ',') + 1);
        $binary = base64_decode($base64Data);
    } else {
        $binary = base64_decode($dataUri);
        $ext = 'png';
    }

    if (!$binary) {
        http_response_code(400);
        echo json_encode([
            'error' => 'Invalid Base64 image payload.',
            'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
        ]);
        exit;
    }

    $uniqueName = 'capture_' . date('Ymd_His') . '_' . substr(md5(uniqid()), 0, 8) . '.' . $ext;
    $targetPath = UPLOAD_DIR . DIRECTORY_SEPARATOR . $uniqueName;
    file_put_contents($targetPath, $binary);
} else {
    http_response_code(400);
    echo json_encode([
        'error' => 'No image file or image_data provided.',
        'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
    ]);
    exit;
}

// 2. Validate saved file with getimagesize
$imgInfo = @getimagesize($targetPath);
if (!$imgInfo) {
    @unlink($targetPath);
    http_response_code(400);
    echo json_encode([
        'error' => 'File is not a valid optical image.',
        'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN'
    ]);
    exit;
}

$inspectionId = 'insp_' . date('Ymd_His') . '_' . substr(md5(uniqid()), 0, 8);

// 3. Execute Python AI Inference safely via CLI
$cmd = sprintf(
    '%s %s --config %s --checkpoint %s --image %s --output-dir %s --inspection-id %s',
    escapeshellarg(PYTHON_BIN),
    escapeshellarg(PREDICT_SCRIPT),
    escapeshellarg(CONFIG_PATH),
    escapeshellarg($resolvedCheckpoint),
    escapeshellarg($targetPath),
    escapeshellarg(HEATMAP_OUTPUT_DIR),
    escapeshellarg($inspectionId)
);

// Execute and capture stdout & stderr separately
$descriptors = [
    0 => ['pipe', 'r'],
    1 => ['pipe', 'w'],
    2 => ['pipe', 'w']
];

$proc = proc_open($cmd, $descriptors, $pipes, ROOT_DIR);
$stdout = '';
$stderr = '';

if (is_resource($proc)) {
    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    fclose($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[2]);
    $exitCode = proc_close($proc);
} else {
    $exitCode = -1;
}

if ($exitCode !== 0 || empty($stdout)) {
    http_response_code(500);
    echo json_encode([
        'error' => 'Python AI inference failed with exit code ' . $exitCode,
        'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN',
        'details' => $stderr
    ], JSON_PRETTY_PRINT);
    exit;
}

$pyResult = json_decode($stdout, true);
if (!$pyResult || isset($pyResult['error']) && $pyResult['error']) {
    http_response_code(500);
    echo json_encode([
        'error' => $pyResult['error'] ?? 'Malformed JSON output from Python AI engine.',
        'message' => 'INSPECTION FAILED – PLEASE TRY AGAIN',
        'details' => $stderr
    ], JSON_PRETTY_PRINT);
    exit;
}

// 4. Transform Paths to Web-Accessible Relative URLs
$webOriginalPath = 'uploads/' . basename($targetPath);
$webHeatmapPath = !empty($pyResult['heatmap_path']) ? 'uploads/heatmaps/' . basename($pyResult['heatmap_path']) : null;
$webOverlayPath = !empty($pyResult['overlay_path']) ? 'uploads/heatmaps/' . basename($pyResult['overlay_path']) : null;

$predictionRaw = strtoupper($pyResult['prediction'] ?? 'NORMAL');
$isAnomaly = ($predictionRaw === 'ANOMALY');

$verdictMessage = $isAnomaly
    ? 'ANOMALY DETECTED – DEFECT FOUND'
    : 'INSPECTION PASSED – NO ANOMALY DETECTED';

$severity = strtoupper($pyResult['severity'] ?? ($isAnomaly ? 'HIGH' : 'NONE'));
$score = floatval($pyResult['anomaly_score'] ?? 0);
$threshold = floatval($pyResult['details']['image_threshold'] ?? 0.6366);
$pixelThreshold = floatval($pyResult['details']['pixel_threshold'] ?? 0.5060);
$areaRatio = isset($pyResult['details']['anomalous_area_ratio']) ? floatval($pyResult['details']['anomalous_area_ratio']) : 0.0;
$areaPct = round($areaRatio * 100, 2);
$scoreDelta = round($score - $threshold, 4);

$decisionMsg = $isAnomaly
    ? sprintf("Anomaly detected because the image anomaly score (%.4f) exceeded the configured decision threshold (%.4f).", $score, $threshold)
    : sprintf("No anomaly detected because the image anomaly score (%.4f) did not exceed the configured decision threshold (%.4f).", $score, $threshold);

$categoryName = $resolvedCategoryInfo['name'] ?? ucfirst($resolvedCategory);

$explanation = [
    'category'              => $resolvedCategory,
    'category_name'         => $categoryName,
    'decision_message'      => $decisionMsg,
    'prediction'            => $predictionRaw,
    'anomaly_score'         => $score,
    'threshold'             => $threshold,
    'pixel_threshold'       => $pixelThreshold,
    'score_delta'           => $scoreDelta,
    'is_above_threshold'    => ($score >= $threshold),
    'severity'              => $severity,
    'anomalous_area_ratio'  => $areaRatio,
    'anomalous_area_pct'    => $areaPct,
    'confidence'            => isset($pyResult['confidence']) && $pyResult['confidence'] !== null ? floatval($pyResult['confidence']) : null,
    'confidence_display'    => (isset($pyResult['confidence']) && $pyResult['confidence'] !== null)
        ? sprintf('%.1f%%', floatval($pyResult['confidence']) * 100)
        : 'Uncalibrated (Metric is Euclidean distance, not probability)',
    'model_version'         => $pyResult['model_version'] ?? '1.0.0',
    'processing_time_ms'    => floatval($pyResult['processing_time_ms'] ?? 14.2),
    'device'                => $pyResult['details']['device'] ?? 'cuda:0',
    'metric_type'           => 'Euclidean Nearest-Neighbor Distance (Unbounded [0, ∞))',
    'score_description'     => sprintf('The image anomaly score is computed as the maximum Euclidean distance of patch feature representations to their nearest neighbor in the nominal %s memory coreset.', $categoryName),
    'heatmap_description'   => 'Heatmap highlights spatial regions where patch feature distance to the normal memory bank is highest. It does not classify specific defect categories.',
    'severity_disclaimer'   => $pyResult['details']['severity_disclaimer'] ?? 'Heuristic visual estimate prototype only. Not a verified physical or structural damage measurement.'
];

$record = [
    'inspection_id'        => $pyResult['inspection_id'] ?? $inspectionId,
    'category'             => $resolvedCategory,
    'category_name'        => $categoryName,
    'filename'             => $rawFilename,
    'original_image'       => $webOriginalPath,
    'prediction'           => $predictionRaw,
    'verdict_message'      => $verdictMessage,
    'anomaly_score'        => $score,
    'threshold'            => $threshold,
    'pixel_threshold'      => $pixelThreshold,
    'score_delta'          => $scoreDelta,
    'severity'             => $severity,
    'severity_score'       => floatval($pyResult['details']['severity_index'] ?? 0),
    'anomalous_area_ratio' => $areaRatio,
    'anomalous_area_pct'   => $areaPct,
    'confidence'           => isset($pyResult['confidence']) && $pyResult['confidence'] !== null ? floatval($pyResult['confidence']) : null,
    'heatmap_path'         => $webHeatmapPath,
    'overlay_path'         => $webOverlayPath,
    'processing_time_ms'   => floatval($pyResult['processing_time_ms'] ?? 14.2),
    'defect_type'          => $isAnomaly ? 'detected_defect' : 'normal',
    'notes'                => $isAnomaly
        ? sprintf('Defect localized. Anomalous area ratio: %.2f%%. Severity: %s.', $areaPct, $severity)
        : 'Inspection passed. Normal industrial component within verified coreset boundary.',
    'device'               => $pyResult['details']['device'] ?? 'cuda:0',
    'model_version'        => $pyResult['model_version'] ?? '1.0.0',
    'source'               => !empty($_POST['source']) ? $_POST['source'] : 'MOBILE_CAMERA',
    'timestamp'            => date('c'),
    'explanation'          => $explanation,
    'details'              => $pyResult['details'] ?? []
];

// 5. Persist to MySQL Database (with auto-fallback)
DefectXDatabase::saveInspection($record);

// Return full result to frontend
echo json_encode($record, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
