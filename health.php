<?php
/**
 * DEFECTX AI - System Health & Hardware Telemetry Endpoint
 * CLOCKIT 2026 Hackathon
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/db.php';

setApiHeaders();

// Resolve requested category or default to 'bottle'
$requestedCategory = $_GET['category'] ?? null;
$categoryResolution = resolveCategoryCheckpoint($requestedCategory);

$activeCategory = $categoryResolution['success'] ? $categoryResolution['category'] : DEFAULT_CATEGORY;
$activeCheckpoint = $categoryResolution['success'] ? $categoryResolution['checkpoint'] : CHECKPOINT_PATH;

// Catalog all categories and their checkpoint status
$categoryCatalog = [];
$availableCategories = [];
foreach (SUPPORTED_CATEGORIES as $slug => $catMeta) {
    $cPath = AI_ENGINE_DIR . DIRECTORY_SEPARATOR . 'checkpoints' . DIRECTORY_SEPARATOR . $catMeta['checkpoint_file'];
    $exists = file_exists($cPath);
    $size = $exists ? filesize($cPath) : 0;
    $isAvailable = $catMeta['available'] && $exists;
    
    $entry = [
        'category' => $slug,
        'name' => $catMeta['name'],
        'dataset' => $catMeta['dataset'],
        'available' => $isAvailable,
        'checkpoint_file' => $catMeta['checkpoint_file'],
        'checkpoint_exists' => $exists,
        'checkpoint_size_bytes' => $size,
        'description' => $catMeta['description']
    ];
    $categoryCatalog[$slug] = $entry;
    if ($isAvailable) {
        $availableCategories[] = $slug;
    }
}

// 1. Check Python AI Engine for the selected/active checkpoint
$cmd = sprintf(
    '%s %s --config %s --checkpoint %s --health',
    escapeshellarg(PYTHON_BIN),
    escapeshellarg(PREDICT_SCRIPT),
    escapeshellarg(CONFIG_PATH),
    escapeshellarg($activeCheckpoint)
);

$output = @exec($cmd, $outLines, $code);
$pyHealth = json_decode(implode("\n", $outLines), true);

// 2. Check Database
$dbStatus = DefectXDatabase::getStatus();

$response = [
    'status' => ($code === 0 && !empty($pyHealth) && $pyHealth['status'] === 'healthy') ? 'ONLINE' : 'DEGRADED',
    'engine' => 'PatchCore-ResNet18-Baseline',
    'version' => $pyHealth['model_version'] ?? '1.0.0',
    'device' => $pyHealth['device'] ?? 'cuda:0',
    'cuda_available' => $pyHealth['cuda_available'] ?? true,
    'vram' => $pyHealth['vram'] ?? ['allocated_mb' => 0, 'reserved_mb' => 0],
    'active_category' => $activeCategory,
    'active_category_valid' => $categoryResolution['success'],
    'category_resolution_error' => $categoryResolution['error'],
    'threshold' => $pyHealth['image_threshold'] ?? ($activeCategory === 'candle' ? 1.5599732 : 0.6366),
    'pixel_threshold' => $pyHealth['pixel_threshold'] ?? ($activeCategory === 'candle' ? 0.8939507 : 0.506),
    'checkpoint_exists' => $pyHealth['checkpoint_exists'] ?? file_exists($activeCheckpoint),
    'checkpoint_path' => 'ai-engine/checkpoints/' . basename($activeCheckpoint),
    'checkpoint_size_bytes' => file_exists($activeCheckpoint) ? filesize($activeCheckpoint) : 0,
    'available_categories' => $availableCategories,
    'categories' => $categoryCatalog,
    'database' => $dbStatus,
    'timestamp' => date('c')
];

echo json_encode($response, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
