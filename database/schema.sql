-- ============================================================================
-- DEFECTX AI - MySQL Production Database Schema
-- CLOCKIT 2026 Hackathon
-- ============================================================================

CREATE DATABASE IF NOT EXISTS `defectx_db`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `defectx_db`;

-- ----------------------------------------------------------------------------
-- Table: inspections
-- Stores full telemetry, image references, and AI inference verdicts
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `inspections` (
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Initial seed inspections (MVTec Bottle benchmark baseline)
-- ----------------------------------------------------------------------------
INSERT INTO `inspections` 
  (`inspection_id`, `category`, `filename`, `original_image`, `prediction`, `anomaly_score`, `threshold`, `severity`, `severity_score`, `anomalous_area_ratio`, `confidence`, `heatmap_path`, `overlay_path`, `processing_time_ms`, `defect_type`, `notes`, `device`, `model_version`, `source`, `created_at`)
VALUES
  ('insp_seed_001', 'bottle', 'bottle_broken_large_001.png', 'assets/images/sample-broken-large.png', 'ANOMALY', 4.1829, 0.6366, 'CRITICAL', 0.7373, 0.3687, NULL, 'ai-engine/results/heatmaps/insp_seed_001_heatmap.png', 'ai-engine/results/heatmaps/insp_seed_001_overlay.png', 14.80, 'broken_large', 'Structural fracture detected at neck and body interface.', 'cuda:0 (NVIDIA RTX 4050)', '1.0.0', 'BENCHMARK_PRESET', '2026-11-25 15:42:10'),
  ('insp_seed_002', 'bottle', 'bottle_contamination_004.png', 'assets/images/sample-contamination.png', 'ANOMALY', 3.2450, 0.6366, 'MEDIUM', 0.4500, 0.1420, NULL, 'ai-engine/results/heatmaps/insp_seed_002_heatmap.png', 'ai-engine/results/heatmaps/insp_seed_002_overlay.png', 13.90, 'contamination', 'Foreign particulate matter clustered in lower liquid chamber.', 'cuda:0 (NVIDIA RTX 4050)', '1.0.0', 'BENCHMARK_PRESET', '2026-11-25 15:38:22'),
  ('insp_seed_003', 'bottle', 'bottle_normal_018.png', 'assets/images/sample-normal.png', 'NORMAL', 0.4120, 0.6366, 'NONE', 0.0000, 0.0000, NULL, 'ai-engine/results/heatmaps/insp_seed_003_heatmap.png', 'ai-engine/results/heatmaps/insp_seed_003_overlay.png', 13.50, 'normal', 'Component conforms to industrial tolerance standards.', 'cuda:0 (NVIDIA RTX 4050)', '1.0.0', 'BENCHMARK_PRESET', '2026-11-25 15:32:05')
ON DUPLICATE KEY UPDATE `inspection_id` = `inspection_id`;
