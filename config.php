<?php
/**
 * DEFECTX AI - Backend Configuration & Environment Setup
 * CLOCKIT 2026 Hackathon
 */

// Error reporting for production/dev
error_reporting(E_ALL & ~E_NOTICE & ~E_DEPRECATED);
ini_set('display_errors', '0');

// Set JSON headers and CORS
function setApiHeaders() {
    if (php_sapi_name() !== 'cli') {
        header('Content-Type: application/json; charset=utf-8');
        header('Access-Control-Allow-Origin: *');
        header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
        header('Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With');
        header('Cache-Control: no-cache, no-store, must-revalidate');
    }
    
    if (($_SERVER['REQUEST_METHOD'] ?? '') === 'OPTIONS') {
        http_response_code(200);
        exit;
    }
}

// Project Root Directories
define('ROOT_DIR', dirname(__DIR__));
define('FRONTEND_DIR', ROOT_DIR . DIRECTORY_SEPARATOR . 'frontend');
define('AI_ENGINE_DIR', ROOT_DIR . DIRECTORY_SEPARATOR . 'ai-engine');

// Python AI Engine Executable and Scripts
$possiblePythons = [
    'python',
    'python3',
    'C:\\Program Files\\Python313\\python.exe',
    'C:\\Program Files\\Python311\\python.exe',
    'C:\\Python311\\python.exe'
];
$pythonBin = 'python';
foreach ($possiblePythons as $py) {
    if ($py === 'python' || $py === 'python3') {
        $test = @exec("$py -V 2>&1");
        if ($test && stripos($test, 'Python') !== false) {
            $pythonBin = $py;
            break;
        }
    } elseif (file_exists($py)) {
        $pythonBin = $py;
        break;
    }
}
define('PYTHON_BIN', $pythonBin);

define('PREDICT_SCRIPT', AI_ENGINE_DIR . DIRECTORY_SEPARATOR . 'predict.py');
define('CHECKPOINT_PATH', AI_ENGINE_DIR . DIRECTORY_SEPARATOR . 'checkpoints' . DIRECTORY_SEPARATOR . 'defectx_bottle.pt');
define('CONFIG_PATH', AI_ENGINE_DIR . DIRECTORY_SEPARATOR . 'config' . DIRECTORY_SEPARATOR . 'default_config.yaml');

// Central Category Registry
// Available categories are strictly verified against existing trained checkpoints.
// Missing category input defaults to 'bottle' for legacy backward compatibility.
// Explicit unsupported category => 400 Bad Request.
// Registered but untrained category => 503 Service Unavailable (MODEL_NOT_AVAILABLE).
define('DEFAULT_CATEGORY', 'bottle');

const SUPPORTED_CATEGORIES = [
    'bottle' => [
        'name' => 'MVTec Bottle',
        'dataset' => 'mvtec',
        'checkpoint_file' => 'defectx_bottle.pt',
        'available' => true,
        'description' => 'MVTec AD industrial bottle benchmark checkpoint'
    ],
    'candle' => [
        'name' => 'VisA Candle',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_candle.pt',
        'available' => true,
        'description' => 'VisA industrial candle dataset checkpoint'
    ],
    'capsules' => [
        'name' => 'VisA Capsules',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_capsules.pt',
        'available' => false,
        'description' => 'VisA industrial capsules dataset checkpoint (untrained)'
    ],
    'cashew' => [
        'name' => 'VisA Cashew',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_cashew.pt',
        'available' => false,
        'description' => 'VisA industrial cashew dataset checkpoint (untrained)'
    ],
    'chewinggum' => [
        'name' => 'VisA Chewing Gum',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_chewinggum.pt',
        'available' => false,
        'description' => 'VisA industrial chewinggum dataset checkpoint (untrained)'
    ],
    'fryum' => [
        'name' => 'VisA Fryum',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_fryum.pt',
        'available' => false,
        'description' => 'VisA industrial fryum dataset checkpoint (untrained)'
    ],
    'macaroni1' => [
        'name' => 'VisA Macaroni 1',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_macaroni1.pt',
        'available' => false,
        'description' => 'VisA industrial macaroni1 dataset checkpoint (untrained)'
    ],
    'macaroni2' => [
        'name' => 'VisA Macaroni 2',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_macaroni2.pt',
        'available' => false,
        'description' => 'VisA industrial macaroni2 dataset checkpoint (untrained)'
    ],
    'pcb1' => [
        'name' => 'VisA PCB 1',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_pcb1.pt',
        'available' => false,
        'description' => 'VisA industrial pcb1 dataset checkpoint (untrained)'
    ],
    'pcb2' => [
        'name' => 'VisA PCB 2',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_pcb2.pt',
        'available' => false,
        'description' => 'VisA industrial pcb2 dataset checkpoint (untrained)'
    ],
    'pcb3' => [
        'name' => 'VisA PCB 3',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_pcb3.pt',
        'available' => false,
        'description' => 'VisA industrial pcb3 dataset checkpoint (untrained)'
    ],
    'pcb4' => [
        'name' => 'VisA PCB 4',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_pcb4.pt',
        'available' => false,
        'description' => 'VisA industrial pcb4 dataset checkpoint (untrained)'
    ],
    'pipe_fryum' => [
        'name' => 'VisA Pipe Fryum',
        'dataset' => 'visa',
        'checkpoint_file' => 'defectx_pipe_fryum.pt',
        'available' => false,
        'description' => 'VisA industrial pipe_fryum dataset checkpoint (untrained)'
    ]
];

/**
 * Resolve product category and return checkpoint info or error status.
 *
 * @param string|null $categoryInput
 * @return array ['success' => bool, 'category' => string, 'checkpoint' => string|null, 'status_code' => int, 'error' => string|null, 'category_info' => array|null]
 */
function resolveCategoryCheckpoint(?string $categoryInput): array {
    if ($categoryInput === null || trim($categoryInput) === '') {
        $cat = DEFAULT_CATEGORY;
    } else {
        $cat = strtolower(trim($categoryInput));
    }

    if (!array_key_exists($cat, SUPPORTED_CATEGORIES)) {
        return [
            'success' => false,
            'category' => $cat,
            'checkpoint' => null,
            'status_code' => 400,
            'error' => "Unsupported category '{$cat}'. Supported categories: " . implode(', ', array_keys(SUPPORTED_CATEGORIES)),
            'category_info' => null
        ];
    }

    $info = SUPPORTED_CATEGORIES[$cat];
    $checkpointPath = AI_ENGINE_DIR . DIRECTORY_SEPARATOR . 'checkpoints' . DIRECTORY_SEPARATOR . $info['checkpoint_file'];

    if (!$info['available'] || !file_exists($checkpointPath)) {
        return [
            'success' => false,
            'category' => $cat,
            'checkpoint' => null,
            'status_code' => 503,
            'error' => "Model checkpoint for category '{$cat}' is not currently available.",
            'category_info' => $info
        ];
    }

    return [
        'success' => true,
        'category' => $cat,
        'checkpoint' => $checkpointPath,
        'status_code' => 200,
        'error' => null,
        'category_info' => $info
    ];
}

// Upload and Output Directories
define('UPLOAD_DIR', FRONTEND_DIR . DIRECTORY_SEPARATOR . 'uploads');
define('HEATMAP_OUTPUT_DIR', FRONTEND_DIR . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . 'heatmaps');

if (!is_dir(UPLOAD_DIR)) {
    @mkdir(UPLOAD_DIR, 0777, true);
}
if (!is_dir(HEATMAP_OUTPUT_DIR)) {
    @mkdir(HEATMAP_OUTPUT_DIR, 0777, true);
}

// Database Credentials (Standard XAMPP / Local MySQL)
define('DB_HOST', getenv('DEFECTX_DB_HOST') ?: '127.0.0.1');
define('DB_PORT', getenv('DEFECTX_DB_PORT') ?: '3306');
define('DB_USER', getenv('DEFECTX_DB_USER') ?: 'root');
define('DB_PASS', getenv('DEFECTX_DB_PASS') ?: '');
define('DB_NAME', getenv('DEFECTX_DB_NAME') ?: 'defectx_db');

// Fallback JSON Store when MySQL is temporarily unavailable
define('FALLBACK_STORE_PATH', FRONTEND_DIR . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . 'inspections_history.json');
