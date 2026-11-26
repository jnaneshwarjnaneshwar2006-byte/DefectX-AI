<?php
/**
 * DEFECTX AI - MySQL PDO Database Layer with Resilient Fallback
 * CLOCKIT 2026 Hackathon
 */

require_once __DIR__ . '/config.php';

class DefectXDatabase {
    private static ?PDO $pdo = null;
    private static bool $mysqlAvailable = false;
    private static ?string $initError = null;

    public static function getConnection(): ?PDO {
        if (self::$pdo !== null) {
            return self::$pdo;
        }

        try {
            // First attempt connecting directly to the database
            $dsn = sprintf('mysql:host=%s;port=%s;dbname=%s;charset=utf8mb4', DB_HOST, DB_PORT, DB_NAME);
            self::$pdo = new PDO($dsn, DB_USER, DB_PASS, [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false,
                PDO::ATTR_TIMEOUT => 2
            ]);
            self::$mysqlAvailable = true;
            self::ensureTables();
            return self::$pdo;
        } catch (PDOException $e) {
            // Attempt to connect to server without dbname to create database if missing
            try {
                $rootDsn = sprintf('mysql:host=%s;port=%s;charset=utf8mb4', DB_HOST, DB_PORT);
                $rootPdo = new PDO($rootDsn, DB_USER, DB_PASS, [
                    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                    PDO::ATTR_TIMEOUT => 2
                ]);
                $rootPdo->exec("CREATE DATABASE IF NOT EXISTS `" . DB_NAME . "` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
                
                // Reconnect with dbname
                $dsn = sprintf('mysql:host=%s;port=%s;dbname=%s;charset=utf8mb4', DB_HOST, DB_PORT, DB_NAME);
                self::$pdo = new PDO($dsn, DB_USER, DB_PASS, [
                    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                    PDO::ATTR_EMULATE_PREPARES => false
                ]);
                self::$mysqlAvailable = true;
                self::ensureTables();
                return self::$pdo;
            } catch (Exception $e2) {
                self::$mysqlAvailable = false;
                self::$initError = $e2->getMessage();
                return null;
            }
        }
    }

    public static function isMysqlAvailable(): bool {
        self::getConnection();
        return self::$mysqlAvailable;
    }

    public static function getStatus(): array {
        $available = self::isMysqlAvailable();
        return [
            'mysql_connected' => $available,
            'database' => DB_NAME,
            'host' => DB_HOST,
            'port' => DB_PORT,
            'error' => self::$initError
        ];
    }

    private static function ensureTables(): void {
        if (!self::$pdo) return;
        $sql = "CREATE TABLE IF NOT EXISTS `inspections` (
            `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `inspection_id` VARCHAR(64) NOT NULL UNIQUE,
            `category` VARCHAR(64) NOT NULL DEFAULT 'bottle',
            `filename` VARCHAR(255) NOT NULL,
            `original_image` TEXT NOT NULL,
            `prediction` ENUM('NORMAL', 'ANOMALY') NOT NULL,
            `anomaly_score` DECIMAL(8, 4) NOT NULL,
            `threshold` DECIMAL(8, 4) NOT NULL DEFAULT 0.6366,
            `severity` ENUM('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL DEFAULT 'NONE',
            `severity_score` DECIMAL(6, 4) DEFAULT 0.0000,
            `anomalous_area_ratio` DECIMAL(6, 4) DEFAULT 0.0000,
            `confidence` DECIMAL(6, 4) NULL,
            `heatmap_path` TEXT NULL,
            `overlay_path` TEXT NULL,
            `processing_time_ms` DECIMAL(8, 2) NOT NULL DEFAULT 0.00,
            `defect_type` VARCHAR(64) DEFAULT 'unspecified',
            `notes` TEXT NULL,
            `device` VARCHAR(128) DEFAULT 'cuda:0',
            `model_version` VARCHAR(32) DEFAULT '1.0.0',
            `source` ENUM('MOBILE_CAMERA', 'WEB_UPLOAD', 'BENCHMARK_PRESET') DEFAULT 'MOBILE_CAMERA',
            `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX `idx_category` (`category`),
            INDEX `idx_prediction` (`prediction`),
            INDEX `idx_severity` (`severity`),
            INDEX `idx_created_at` (`created_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;";
        self::$pdo->exec($sql);

        // Migrate column if table existed without anomalous_area_ratio
        try {
            self::$pdo->exec("ALTER TABLE `inspections` ADD COLUMN `anomalous_area_ratio` DECIMAL(6, 4) DEFAULT 0.0000 AFTER `severity_score`");
        } catch (Exception $e) {
            // Column already exists or table cannot be altered
        }

        // Migrate column if table existed without category
        try {
            self::$pdo->exec("ALTER TABLE `inspections` ADD COLUMN `category` VARCHAR(64) NOT NULL DEFAULT 'bottle' AFTER `inspection_id`");
            self::$pdo->exec("CREATE INDEX `idx_category` ON `inspections` (`category`)");
        } catch (Exception $e) {
            // Column/index already exists or table cannot be altered
        }
    }

    public static function formatRecord(array $row): array {
        $score = floatval($row['anomaly_score'] ?? 0);
        $threshold = floatval($row['threshold'] ?? 0.6366);
        $pixelThreshold = isset($row['pixel_threshold']) ? floatval($row['pixel_threshold']) : 0.5060;
        $category = $row['category'] ?? 'bottle';
        $isAnomaly = strtoupper($row['prediction'] ?? '') === 'ANOMALY';
        $severity = strtoupper($row['severity'] ?? ($isAnomaly ? 'HIGH' : 'NONE'));
        $areaRatio = isset($row['anomalous_area_ratio']) ? floatval($row['anomalous_area_ratio']) : 0.0;
        $areaPct = round($areaRatio * 100, 2);
        $scoreDelta = round($score - $threshold, 4);

        $decisionMsg = $isAnomaly
            ? sprintf("Anomaly detected because the image anomaly score (%.4f) exceeded the configured decision threshold (%.4f).", $score, $threshold)
            : sprintf("No anomaly detected because the image anomaly score (%.4f) did not exceed the configured decision threshold (%.4f).", $score, $threshold);

        $row['category'] = $category;
        $row['anomaly_score'] = $score;
        $row['threshold'] = $threshold;
        $row['pixel_threshold'] = $pixelThreshold;
        $row['score_delta'] = $scoreDelta;
        $row['anomalous_area_ratio'] = $areaRatio;
        $row['anomalous_area_pct'] = $areaPct;

        if (empty($row['explanation']) || !is_array($row['explanation'])) {
            $categoryName = (defined('SUPPORTED_CATEGORIES') && isset(SUPPORTED_CATEGORIES[$category]))
                ? SUPPORTED_CATEGORIES[$category]['name']
                : ucfirst($category);
            $row['explanation'] = [
                'category'              => $category,
                'category_name'         => $categoryName,
                'decision_message'      => $decisionMsg,
                'prediction'            => $isAnomaly ? 'ANOMALY' : 'NORMAL',
                'anomaly_score'         => $score,
                'threshold'             => $threshold,
                'pixel_threshold'       => $pixelThreshold,
                'score_delta'           => $scoreDelta,
                'is_above_threshold'    => ($score >= $threshold),
                'severity'              => $severity,
                'anomalous_area_ratio'  => $areaRatio,
                'anomalous_area_pct'    => $areaPct,
                'confidence'            => isset($row['confidence']) && $row['confidence'] !== null ? floatval($row['confidence']) : null,
                'confidence_display'    => (isset($row['confidence']) && $row['confidence'] !== null)
                    ? sprintf('%.1f%%', floatval($row['confidence']) * 100)
                    : 'Uncalibrated (Metric is Euclidean distance, not probability)',
                'model_version'         => $row['model_version'] ?? '1.0.0',
                'processing_time_ms'    => floatval($row['processing_time_ms'] ?? 14.2),
                'device'                => $row['device'] ?? 'cuda:0',
                'metric_type'           => 'Euclidean Nearest-Neighbor Distance (Unbounded [0, ∞))',
                'score_description'     => sprintf('The image anomaly score is computed as the maximum Euclidean distance of patch feature representations to their nearest neighbor in the nominal %s memory coreset.', $categoryName),
                'heatmap_description'   => 'Heatmap highlights spatial regions where patch feature distance to the normal memory bank is highest. It does not classify specific defect categories.',
                'severity_disclaimer'   => 'Heuristic visual estimate prototype only. Not a verified physical or structural damage measurement.'
            ];
        }

        return $row;
    }

    public static function saveInspection(array $data): bool {
        $pdo = self::getConnection();
        $category = $data['category'] ?? 'bottle';
        if ($pdo) {
            try {
                $sql = "INSERT INTO `inspections` (
                    `inspection_id`, `category`, `filename`, `original_image`, `prediction`,
                    `anomaly_score`, `threshold`, `severity`, `severity_score`, `anomalous_area_ratio`,
                    `confidence`, `heatmap_path`, `overlay_path`, `processing_time_ms`,
                    `defect_type`, `notes`, `device`, `model_version`, `source`
                ) VALUES (
                    :inspection_id, :category, :filename, :original_image, :prediction,
                    :anomaly_score, :threshold, :severity, :severity_score, :anomalous_area_ratio,
                    :confidence, :heatmap_path, :overlay_path, :processing_time_ms,
                    :defect_type, :notes, :device, :model_version, :source
                ) ON DUPLICATE KEY UPDATE
                    `category` = VALUES(`category`),
                    `prediction` = VALUES(`prediction`),
                    `anomaly_score` = VALUES(`anomaly_score`),
                    `threshold` = VALUES(`threshold`),
                    `severity` = VALUES(`severity`),
                    `anomalous_area_ratio` = VALUES(`anomalous_area_ratio`),
                    `heatmap_path` = VALUES(`heatmap_path`),
                    `overlay_path` = VALUES(`overlay_path`)";

                $stmt = $pdo->prepare($sql);
                $stmt->execute([
                    ':inspection_id'        => $data['inspection_id'],
                    ':category'             => $category,
                    ':filename'             => $data['filename'] ?? 'captured_specimen.png',
                    ':original_image'       => $data['original_image'] ?? '',
                    ':prediction'           => strtoupper($data['prediction'] ?? 'NORMAL'),
                    ':anomaly_score'        => floatval($data['anomaly_score'] ?? 0),
                    ':threshold'            => floatval($data['threshold'] ?? 0.6366),
                    ':severity'             => strtoupper($data['severity'] ?? 'NONE'),
                    ':severity_score'       => floatval($data['severity_score'] ?? 0),
                    ':anomalous_area_ratio' => floatval($data['anomalous_area_ratio'] ?? 0),
                    ':confidence'           => isset($data['confidence']) && $data['confidence'] !== null ? floatval($data['confidence']) : null,
                    ':heatmap_path'         => $data['heatmap_path'] ?? null,
                    ':overlay_path'         => $data['overlay_path'] ?? null,
                    ':processing_time_ms'   => floatval($data['processing_time_ms'] ?? 0),
                    ':defect_type'          => $data['defect_type'] ?? 'unspecified',
                    ':notes'                => $data['notes'] ?? '',
                    ':device'               => $data['device'] ?? 'cuda:0',
                    ':model_version'        => $data['model_version'] ?? '1.0.0',
                    ':source'               => $data['source'] ?? 'MOBILE_CAMERA'
                ]);
            } catch (Exception $e) {
                error_log("MySQL save error: " . $e->getMessage());
            }
        }

        // Ensure category is in array for fallback store
        $data['category'] = $category;
        // Always save to fallback JSON history for resilience and offline capability
        self::saveToFallbackStore($data);
        return true;
    }

    public static function getHistory(array $filters = []): array {
        $pdo = self::getConnection();
        if ($pdo) {
            try {
                $where = [];
                $params = [];

                if (!empty($filters['category']) && $filters['category'] !== 'ALL') {
                    $where[] = "`category` = :cat";
                    $params[':cat'] = strtolower($filters['category']);
                }

                if (!empty($filters['category']) && $filters['category'] !== 'ALL') {
            $catFilter = strtolower($filters['category']);
            $items = array_filter($items, fn($i) => strtolower($i['category'] ?? 'bottle') === $catFilter);
        }
        if (!empty($filters['search'])) {
                    $where[] = "(`inspection_id` LIKE :search OR `filename` LIKE :search2 OR `notes` LIKE :search3)";
                    $params[':search'] = '%' . $filters['search'] . '%';
                    $params[':search2'] = '%' . $filters['search'] . '%';
                    $params[':search3'] = '%' . $filters['search'] . '%';
                }

                if (!empty($filters['prediction']) && $filters['prediction'] !== 'ALL') {
                    $where[] = "`prediction` = :pred";
                    $params[':pred'] = strtoupper($filters['prediction']);
                }

                if (!empty($filters['severity']) && $filters['severity'] !== 'ALL') {
                    $where[] = "`severity` = :sev";
                    $params[':sev'] = strtoupper($filters['severity']);
                }

                $whereSql = count($where) ? 'WHERE ' . implode(' AND ', $where) : '';
                $sql = "SELECT * FROM `inspections` $whereSql ORDER BY `created_at` DESC LIMIT 100";
                
                $stmt = $pdo->prepare($sql);
                $stmt->execute($params);
                $rows = $stmt->fetchAll();
                
                if (count($rows) > 0) {
                    return array_map([self::class, 'formatRecord'], $rows);
                }
            } catch (Exception $e) {
                error_log("MySQL fetch error: " . $e->getMessage());
            }
        }

        // Fallback to JSON store
        $fallback = self::getFromFallbackStore($filters);
        return array_map([self::class, 'formatRecord'], $fallback);
    }

    public static function getInspectionById(string $id): ?array {
        $pdo = self::getConnection();
        if ($pdo) {
            try {
                $stmt = $pdo->prepare("SELECT * FROM `inspections` WHERE `inspection_id` = :id LIMIT 1");
                $stmt->execute([':id' => $id]);
                $row = $stmt->fetch();
                if ($row) return self::formatRecord($row);
            } catch (Exception $e) {
                error_log("MySQL getById error: " . $e->getMessage());
            }
        }

        $all = self::getFromFallbackStore();
        foreach ($all as $item) {
            if ($item['inspection_id'] === $id) {
                return self::formatRecord($item);
            }
        }
        return null;
    }

    private static function saveToFallbackStore(array $record): void {
        $items = self::getFromFallbackStore();
        // Remove duplicate if exists
        $items = array_values(array_filter($items, fn($i) => $i['inspection_id'] !== $record['inspection_id']));
        array_unshift($items, $record);
        if (count($items) > 100) {
            $items = array_slice($items, 0, 100);
        }
        @file_put_contents(FALLBACK_STORE_PATH, json_encode($items, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
    }

    private static function getFromFallbackStore(array $filters = []): array {
        if (!file_exists(FALLBACK_STORE_PATH)) {
            return [];
        }
        $data = @file_get_contents(FALLBACK_STORE_PATH);
        $items = json_decode($data, true) ?: [];

        if (!empty($filters['category']) && $filters['category'] !== 'ALL') {
            $catFilter = strtolower($filters['category']);
            $items = array_filter($items, fn($i) => strtolower($i['category'] ?? 'bottle') === $catFilter);
        }
        if (!empty($filters['search'])) {
            $s = strtolower($filters['search']);
            $items = array_filter($items, function($item) use ($s) {
                return (isset($item['inspection_id']) && stripos($item['inspection_id'], $s) !== false) ||
                       (isset($item['filename']) && stripos($item['filename'], $s) !== false);
            });
        }
        if (!empty($filters['prediction']) && $filters['prediction'] !== 'ALL') {
            $p = strtoupper($filters['prediction']);
            $items = array_filter($items, fn($i) => strtoupper($i['prediction'] ?? '') === $p);
        }
        if (!empty($filters['severity']) && $filters['severity'] !== 'ALL') {
            $sev = strtoupper($filters['severity']);
            $items = array_filter($items, fn($i) => strtoupper($i['severity'] ?? '') === $sev);
        }

        return array_values($items);
    }
}
