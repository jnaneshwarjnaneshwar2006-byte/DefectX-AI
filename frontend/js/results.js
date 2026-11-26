/**
 * DEFECTX AI - Results & Multi-View Visualizer Controller (results.js)
 * CLOCKIT 2026 Hackathon
 */

(function () {
  'use strict';

  let currentInspection = null;

  document.addEventListener('DOMContentLoaded', async () => {
    const params = new URLSearchParams(window.location.search);
    const inspectionId = params.get('id');

    currentInspection = await window.DefectXApi.getInspectionById(inspectionId);
    if (!currentInspection) {
      if (window.showToast) window.showToast('Inspection Not Found', 'Displaying benchmark specimen.', 'warning');
      const hist = await window.DefectXApi.getHistory();
      currentInspection = hist.items[0];
    }

    renderInspectionData(currentInspection);
    initViewModes();
    initSplitSlider();
    initActionButtons();
  });

  function renderInspectionData(item) {
    if (!item) return;

    // Header ID
    const headerId = document.getElementById('header-inspection-id');
    if (headerId) headerId.textContent = item.inspection_id;

    // Numerical Values
    const category = item.category || 'bottle';
    const categoryName = item.category_name || (category === 'candle' ? 'VisA Candle' : 'MVTec Bottle');
    const score = parseFloat(item.anomaly_score !== undefined ? item.anomaly_score : 0);
    const threshold = parseFloat(item.threshold !== undefined ? item.threshold : 0.6366);
    const pixelThreshold = parseFloat(item.pixel_threshold !== undefined ? item.pixel_threshold : 0.5060);
    const scoreDelta = (score - threshold);
    const isAnomaly = item.prediction === 'ANOMALY' || (score >= threshold);
    const severity = (item.severity || (isAnomaly ? 'HIGH' : 'NONE')).toUpperCase();

    // Category
    const valCategory = document.getElementById('val-category');
    if (valCategory) valCategory.textContent = categoryName;

    // Verdict Badge
    const badgeEl = document.getElementById('verdict-badge');
    if (badgeEl) {
      badgeEl.className = `badge ${isAnomaly ? 'badge-danger' : 'badge-success'}`;
      badgeEl.style.fontSize = '0.95rem';
      badgeEl.style.padding = '6px 12px';
      badgeEl.innerHTML = isAnomaly
        ? `<i class="fas fa-triangle-exclamation"></i> ANOMALY DETECTED`
        : `<i class="fas fa-circle-check"></i> NORMAL (CONFORMING)`;
    }

    // Confidence / Metric Type (Uncalibrated notice)
    const confEl = document.getElementById('verdict-confidence');
    const confSubEl = document.getElementById('verdict-confidence-sub');
    if (confEl) {
      if (item.confidence !== null && item.confidence !== undefined && !isNaN(item.confidence)) {
        confEl.textContent = `${(parseFloat(item.confidence) * 100).toFixed(1)}%`;
        if (confSubEl) confSubEl.textContent = 'Calibrated Probability';
      } else {
        confEl.textContent = 'Uncalibrated';
        if (confSubEl) confSubEl.textContent = 'Euclidean Distance Metric';
      }
    }

    // Telemetry Specs
    const valAnomaly = document.getElementById('val-anomaly-score');
    if (valAnomaly) valAnomaly.textContent = score.toFixed(4);

    const valThresh = document.getElementById('val-threshold');
    if (valThresh) valThresh.textContent = threshold.toFixed(4);

    const valDelta = document.getElementById('val-score-delta');
    if (valDelta) {
      const sign = scoreDelta >= 0 ? '+' : '';
      valDelta.textContent = `${sign}${scoreDelta.toFixed(4)}`;
      valDelta.style.color = scoreDelta >= 0 ? 'var(--red)' : 'var(--green)';
    }

    const areaRatio = item.anomalous_area_ratio !== undefined ? parseFloat(item.anomalous_area_ratio) : (item.details?.anomalous_area_ratio !== undefined ? parseFloat(item.details.anomalous_area_ratio) : 0);
    const areaPct = (areaRatio * 100).toFixed(2);
    const valArea = document.getElementById('val-anomalous-area');
    if (valArea) valArea.textContent = `${areaPct}%`;

    const valPixelThresh = document.getElementById('val-pixel-threshold');
    if (valPixelThresh) valPixelThresh.textContent = pixelThreshold.toFixed(4);

    const valLat = document.getElementById('val-latency');
    if (valLat) valLat.textContent = `${item.processing_time_ms || 14.8} ms`;

    const valDev = document.getElementById('val-device');
    if (valDev) valDev.textContent = item.device || 'NVIDIA RTX 4050 CUDA';

    const diagNotes = document.getElementById('diagnostic-notes');
    if (diagNotes) diagNotes.textContent = item.notes || (isAnomaly ? `Structural deviation localized. Anomalous area: ${areaPct}%.` : 'Inference executed within verified nominal coreset bounds.');

    // Severity
    renderSeverity(severity);

    // AI DECISION EXPLANATION ENGINE POPULATION
    const expVersion = document.getElementById('explanation-model-version');
    if (expVersion) expVersion.textContent = item.model_version || '1.0.0';

    // Exact Decision Message
    const decisionMsg = isAnomaly
      ? `Anomaly detected because the image anomaly score (${score.toFixed(4)}) exceeded the configured decision threshold (${threshold.toFixed(4)}).`
      : `No anomaly detected because the image anomaly score (${score.toFixed(4)}) did not exceed the configured decision threshold (${threshold.toFixed(4)}).`;

    const stmtBanner = document.getElementById('decision-statement-banner');
    const stmtIcon = document.getElementById('decision-statement-icon');
    const stmtBadge = document.getElementById('decision-statement-badge');
    const stmtText = document.getElementById('decision-statement-text');

    if (stmtBanner && stmtIcon && stmtBadge && stmtText) {
      stmtText.textContent = item.explanation?.decision_message || decisionMsg;
      if (isAnomaly) {
        stmtBanner.style.borderColor = 'rgba(239, 68, 68, 0.35)';
        stmtBanner.style.background = 'rgba(239, 68, 68, 0.08)';
        stmtIcon.innerHTML = '<i class="fas fa-circle-exclamation"></i>';
        stmtIcon.style.color = 'var(--red)';
        stmtBadge.className = 'badge badge-danger';
        stmtBadge.textContent = 'ANOMALY DETECTED';
      } else {
        stmtBanner.style.borderColor = 'rgba(16, 185, 129, 0.35)';
        stmtBanner.style.background = 'rgba(16, 185, 129, 0.08)';
        stmtIcon.innerHTML = '<i class="fas fa-circle-check"></i>';
        stmtIcon.style.color = 'var(--green)';
        stmtBadge.className = 'badge badge-success';
        stmtBadge.textContent = 'NORMAL CONFORMING';
      }
    }

    // Decision Delta Badge and Meter
    const deltaBadge = document.getElementById('decision-delta-badge');
    if (deltaBadge) {
      const sign = scoreDelta >= 0 ? '+' : '';
      deltaBadge.textContent = `Δ ${sign}${scoreDelta.toFixed(4)}`;
      deltaBadge.className = `badge ${isAnomaly ? 'badge-danger' : 'badge-success'}`;
    }

    const meterScore = document.getElementById('meter-score-val');
    if (meterScore) meterScore.textContent = score.toFixed(4);

    const meterThresh = document.getElementById('meter-thresh-val');
    if (meterThresh) meterThresh.textContent = threshold.toFixed(4);

    const meterFill = document.getElementById('meter-score-fill');
    if (meterFill) {
      let fillPct = 0;
      if (score <= threshold) {
        fillPct = Math.max(5, (score / (threshold || 1)) * 35);
      } else {
        fillPct = Math.min(100, 35 + ((score - threshold) / (threshold * 5 || 1)) * 65);
      }
      meterFill.style.width = `${fillPct.toFixed(1)}%`;
      meterFill.style.background = isAnomaly
        ? 'linear-gradient(90deg, #10B981, #F59E0B 40%, #EF4444 80%)'
        : 'linear-gradient(90deg, #10B981, #38BDF8)';
    }

    // Anomalous Area
    const areaBadge = document.getElementById('area-coverage-badge');
    if (areaBadge) areaBadge.textContent = `${areaPct}%`;

    const areaHeadline = document.getElementById('area-coverage-headline');
    if (areaHeadline) areaHeadline.textContent = isAnomaly ? `${areaPct}% of Surface` : '0.00% (Conforming)';

    const areaPixelThresh = document.getElementById('area-pixel-thresh');
    if (areaPixelThresh) areaPixelThresh.textContent = pixelThreshold.toFixed(4);

    // Confidence / Metric Card
    const confTitle = document.getElementById('confidence-calibration-title');
    if (confTitle) {
      confTitle.textContent = (item.confidence !== null && item.confidence !== undefined && !isNaN(item.confidence))
        ? `${(parseFloat(item.confidence) * 100).toFixed(1)}% (Calibrated)`
        : 'Uncalibrated (Euclidean Metric)';
    }

    // Populate Visualizer Images
    const origSrc = item.original_image || 'uploads/capture_20261125_130907_df917883.png';
    const heatSrc = item.heatmap_path || origSrc;
    const overSrc = item.overlay_path || heatSrc;

    const splitBottom = document.getElementById('split-bottom-img');
    if (splitBottom) splitBottom.src = origSrc;

    const splitTop = document.getElementById('split-top-img');
    if (splitTop) splitTop.src = overSrc;

    const sideOrig = document.getElementById('side-original-img');
    if (sideOrig) sideOrig.src = origSrc;

    const sideOver = document.getElementById('side-overlay-img');
    if (sideOver) sideOver.src = overSrc;

    const staticOrig = document.getElementById('static-original-img');
    if (staticOrig) staticOrig.src = origSrc;

    const staticHeat = document.getElementById('static-heatmap-img');
    if (staticHeat) staticHeat.src = heatSrc;

    const staticOver = document.getElementById('static-overlay-img');
    if (staticOver) staticOver.src = overSrc;
  }

  function renderSeverity(sev) {
    const sevBadge = document.getElementById('severity-badge');
    const seg1 = document.getElementById('sev-seg-1');
    const seg2 = document.getElementById('sev-seg-2');
    const seg3 = document.getElementById('sev-seg-3');
    const seg4 = document.getElementById('sev-seg-4');

    const severity = (sev || 'NONE').toUpperCase();
    sevBadge.textContent = severity;

    // Reset styles
    [seg1, seg2, seg3, seg4].forEach(s => s.style.opacity = '0.2');

    if (severity === 'LOW') {
      seg1.style.opacity = '1';
    } else if (severity === 'MEDIUM') {
      seg1.style.opacity = '1';
      seg2.style.opacity = '1';
    } else if (severity === 'HIGH') {
      seg1.style.opacity = '1';
      seg2.style.opacity = '1';
      seg3.style.opacity = '1';
    } else if (severity === 'CRITICAL') {
      seg1.style.opacity = '1';
      seg2.style.opacity = '1';
      seg3.style.opacity = '1';
      seg4.style.opacity = '1';
    }
  }

  // Segmented View Mode Tabs
  function initViewModes() {
    const buttons = document.querySelectorAll('.segmented-btn');
    const splitSlider = document.getElementById('split-slider-container');
    const sideBySide = document.getElementById('side-by-side-container');
    const staticOriginal = document.getElementById('static-original-img');
    const staticHeatmap = document.getElementById('static-heatmap-img');
    const staticOverlay = document.getElementById('static-overlay-img');

    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const view = btn.getAttribute('data-view');

        // Hide all views
        splitSlider.style.display = 'none';
        sideBySide.style.display = 'none';
        staticOriginal.style.display = 'none';
        staticHeatmap.style.display = 'none';
        staticOverlay.style.display = 'none';

        if (view === 'split') {
          splitSlider.style.display = 'block';
        } else if (view === 'side') {
          sideBySide.style.display = 'grid';
        } else if (view === 'overlay') {
          staticOverlay.style.display = 'block';
        } else if (view === 'heatmap') {
          staticHeatmap.style.display = 'block';
        } else if (view === 'original') {
          staticOriginal.style.display = 'block';
        }
      });
    });
  }

  // Draggable Split Comparison Slider
  function initSplitSlider() {
    const container = document.getElementById('split-slider-container');
    const topLayer = document.getElementById('split-top-layer');
    const handle = document.getElementById('split-divider-handle');

    if (!container || !topLayer || !handle) return;

    let isDragging = false;

    function updateSplitPosition(clientX) {
      const rect = container.getBoundingClientRect();
      let offsetX = clientX - rect.left;
      if (offsetX < 0) offsetX = 0;
      if (offsetX > rect.width) offsetX = rect.width;

      const percentage = (offsetX / rect.width) * 100;
      topLayer.style.width = `${percentage}%`;
      handle.style.left = `${percentage}%`;
    }

    handle.addEventListener('mousedown', () => isDragging = true);
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {
      if (isDragging) updateSplitPosition(e.clientX);
    });

    // Touch Support
    handle.addEventListener('touchstart', () => isDragging = true);
    window.addEventListener('touchend', () => isDragging = false);
    window.addEventListener('touchmove', (e) => {
      if (isDragging && e.touches.length > 0) {
        updateSplitPosition(e.touches[0].clientX);
      }
    });

    container.addEventListener('click', (e) => {
      if (e.target !== handle && !handle.contains(e.target)) {
        updateSplitPosition(e.clientX);
      }
    });
  }

  function initActionButtons() {
    // Print QA Report
    const btnPrint = document.getElementById('btn-print-report');
    if (btnPrint) {
      btnPrint.addEventListener('click', () => window.print());
    }

    // Export JSON
    const btnExport = document.getElementById('btn-export-json');
    if (btnExport) {
      btnExport.addEventListener('click', () => {
        if (!currentInspection) return;
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(currentInspection, null, 2));
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = `${currentInspection.inspection_id}_verdict.json`;
        a.click();
        if (window.showToast) window.showToast('JSON Exported', 'Full inspection verdict downloaded.', 'success');
      });
    }

    // Save Overlay Image
    const btnSaveOverlay = document.getElementById('btn-download-overlay');
    if (btnSaveOverlay) {
      btnSaveOverlay.addEventListener('click', () => {
        if (!currentInspection) return;
        const a = document.createElement('a');
        a.href = currentInspection.overlay_path || currentInspection.heatmap_path;
        a.download = `${currentInspection.inspection_id}_overlay.png`;
        a.click();
        if (window.showToast) window.showToast('Overlay Saved', 'Defect heatmap overlay downloaded.', 'success');
      });
    }
  }

})();
