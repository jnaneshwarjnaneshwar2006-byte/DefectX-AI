<?php
/**
 * DEFECTX AI - Inspection History Endpoint
 * CLOCKIT 2026 Hackathon
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/db.php';

setApiHeaders();

$filters = [
    'search' => $_GET['search'] ?? '',
    'prediction' => $_GET['prediction'] ?? 'ALL',
    'severity' => $_GET['severity'] ?? 'ALL'
];

if (isset($_GET['id']) && !empty($_GET['id'])) {
    $item = DefectXDatabase::getInspectionById($_GET['id']);
    if ($item) {
        echo json_encode($item, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    } else {
        http_response_code(404);
        echo json_encode(['error' => 'Inspection record not found.'], JSON_PRETTY_PRINT);
    }
    exit;
}

$items = DefectXDatabase::getHistory($filters);

echo json_encode([
    'items' => $items,
    'total' => count($items),
    'filters' => $filters,
    'database_connected' => DefectXDatabase::isMysqlAvailable()
], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
