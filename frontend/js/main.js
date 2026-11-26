/**
 * DEFECTX AI - Enterprise Global Application Controller (main.js)
 * CLOCKIT 2026 Hackathon
 *
 * Provides shell controllers:
 * - Desktop collapsible sidebar with localStorage persistence
 * - Mobile drawer toggle and backdrop
 * - Active page route highlighting
 * - Global command search (Ctrl+K shortcut)
 * - Inspection Detail Drawer (slide-over viewer)
 * - Global toast notification system
 * - Modal controllers
 */

(function (window, document) {
  'use strict';

  document.addEventListener('DOMContentLoaded', () => {
    initSidebarState();
    initSidebarCollapsing();
    initMobileDrawer();
    initActiveNavHighlight();
    initCommandSearch();
    initToastEngine();
    initDrawerBackdrop();
  });

  // 1. Sidebar Collapsing & Persistence
  function initSidebarState() {
    const isCollapsed = localStorage.getItem('defectx_sidebar_collapsed') === 'true';
    if (isCollapsed) {
      document.body.classList.add('sidebar-collapsed');
    }
  }

  function initSidebarCollapsing() {
    const toggleBtn = document.getElementById('sidebar-collapse-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('sidebar-collapsed');
        const isCollapsed = document.body.classList.contains('sidebar-collapsed');
        localStorage.setItem('defectx_sidebar_collapsed', isCollapsed ? 'true' : 'false');
      });
    }
  }

  // 2. Mobile Drawer Navigation
  function initMobileDrawer() {
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.getElementById('app-sidebar');
    const backdrop = document.getElementById('sidebar-overlay-backdrop');

    if (mobileBtn && sidebar) {
      mobileBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        if (backdrop) backdrop.classList.toggle('active');
      });
    }

    if (backdrop && sidebar) {
      backdrop.addEventListener('click', () => {
        sidebar.classList.remove('open');
        backdrop.classList.remove('active');
      });
    }
  }

  // 3. Active Nav Route Highlighting
  function initActiveNavHighlight() {
    const currentPath = window.location.pathname;
    const links = document.querySelectorAll('.nav-item-link');

    links.forEach(link => {
      const href = link.getAttribute('href');
      if (!href) return;

      if (currentPath.endsWith(href) ||
        (href === 'dashboard.html' && (currentPath.endsWith('/') || currentPath.endsWith('index.html')))) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  // 4. Command Search (Ctrl+K / Cmd+K)
  function initCommandSearch() {
    const searchInput = document.getElementById('global-command-search');
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (searchInput) searchInput.focus();
      }
    });

    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const val = searchInput.value.trim();
          if (val) {
            window.location.href = `history.html?search=${encodeURIComponent(val)}`;
          }
        }
      });
    }
  }

  // 5. Global Toast Notification System
  let toastContainer = null;

  function initToastEngine() {
    toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
      toastContainer = document.createElement('div');
      toastContainer.id = 'toast-container';
      toastContainer.className = 'toast-container';
      document.body.appendChild(toastContainer);
    }
  }

  window.showToast = function (title, message, type = 'info', duration = 4000) {
    if (!toastContainer) initToastEngine();

    const toast = document.createElement('div');
    toast.className = `toast-item toast-${type}`;

    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-check-circle';
    if (type === 'error') icon = 'fa-triangle-exclamation';
    if (type === 'warning') icon = 'fa-exclamation-triangle';

    toast.innerHTML = `
      <div class="toast-icon"><i class="fas ${icon}"></i></div>
      <div class="toast-content">
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close" aria-label="Close">&times;</button>
    `;

    toast.querySelector('.toast-close').addEventListener('click', () => dismissToast(toast));
    toastContainer.appendChild(toast);

    if (duration > 0) {
      setTimeout(() => dismissToast(toast), duration);
    }
  };

  function dismissToast(toast) {
    toast.style.animation = 'toastSlideOut 0.25s ease-in forwards';
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 250);
  }

  // 6. Inspection Detail Drawer Controller
  function initDrawerBackdrop() {
    let backdrop = document.getElementById('drawer-backdrop');
    if (!backdrop) {
      backdrop = document.createElement('div');
      backdrop.id = 'drawer-backdrop';
      backdrop.className = 'drawer-backdrop';
      document.body.appendChild(backdrop);
    }
    backdrop.addEventListener('click', () => window.closeInspectionDrawer());
  }

  window.openInspectionDrawer = function (item) {
    if (!item) return;

    let drawer = document.getElementById('inspection-detail-drawer');
    const backdrop = document.getElementById('drawer-backdrop');

    if (!drawer) {
      drawer = document.createElement('div');
      drawer.id = 'inspection-detail-drawer';
      drawer.className = 'detail-drawer';
      document.body.appendChild(drawer);
    }

    const score = parseFloat(item.anomaly_score !== undefined ? item.anomaly_score : 0);
    const threshold = parseFloat(item.threshold !== undefined ? item.threshold : 0.6366);
    const scoreDelta = (score - threshold);
    const deltaSign = scoreDelta >= 0 ? '+' : '';
    const deltaStr = `${deltaSign}${scoreDelta.toFixed(4)}`;

    const isAnomaly = item.prediction === 'ANOMALY' || (score >= threshold);
    const badgeClass = isAnomaly ? 'badge-danger' : 'badge-success';
    const badgeIcon = isAnomaly ? 'fa-triangle-exclamation' : 'fa-check';

    const areaRatio = item.anomalous_area_ratio !== undefined ? parseFloat(item.anomalous_area_ratio) : (item.details?.anomalous_area_ratio !== undefined ? parseFloat(item.details.anomalous_area_ratio) : 0);
    const areaPct = (areaRatio * 100).toFixed(2);

    const decisionMsg = isAnomaly
      ? `Anomaly detected because the image anomaly score (${score.toFixed(4)}) exceeded the configured decision threshold (${threshold.toFixed(4)}).`
      : `No anomaly detected because the image anomaly score (${score.toFixed(4)}) did not exceed the configured decision threshold (${threshold.toFixed(4)}).`;

    const confDisplay = (item.confidence !== null && item.confidence !== undefined && !isNaN(item.confidence))
      ? `${(parseFloat(item.confidence) * 100).toFixed(1)}%`
      : 'Uncalibrated (Euclidean Metric)';

    drawer.innerHTML = `
      <div class="drawer-header">
        <div>
          <span style="font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono);">INSPECTION RECORD</span>
          <div class="drawer-title font-mono text-cyan">${item.inspection_id}</div>
        </div>
        <button class="drawer-close-btn" onclick="window.closeInspectionDrawer()">&times;</button>
      </div>

      <div class="drawer-body">
        <!-- Visual Comparison Preview -->
        <div style="background: #040810; border: 1px solid var(--border-panel); border-radius: var(--radius-md); padding: 12px; display: flex; flex-direction: column; gap: 8px;">
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">IMAGE HEATMAP PREVIEW</div>
          <img src="${item.overlay_path || item.heatmap_path || item.original_image}" style="width: 100%; max-height: 220px; object-fit: contain; border-radius: var(--radius-sm);" alt="Inspection Preview">
        </div>

        <!-- Verdict Banner -->
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: var(--bg-panel-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
          <div>
            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">AI Verdict</div>
            <span class="badge ${badgeClass}" style="font-size: 0.85rem; margin-top: 4px;">
              <i class="fas ${badgeIcon}"></i> ${item.prediction}
            </span>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Severity Grade</div>
            <span class="badge badge-warning" style="font-size: 0.85rem; margin-top: 4px;">${item.severity || 'NONE'}</span>
          </div>
        </div>

        <!-- AI Decision Explanation Statement -->
        <div style="background: ${isAnomaly ? 'rgba(239, 68, 68, 0.08)' : 'rgba(16, 185, 129, 0.08)'}; border: 1px solid ${isAnomaly ? 'rgba(239, 68, 68, 0.28)' : 'rgba(16, 185, 129, 0.28)'}; border-radius: var(--radius-sm); padding: 12px 14px; font-size: 0.82rem; line-height: 1.5;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <strong style="color: ${isAnomaly ? 'var(--red)' : 'var(--green)'}; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Decision Explanation</strong>
            <span class="badge ${badgeClass} font-mono" style="font-size: 0.7rem;">Δ ${deltaStr}</span>
          </div>
          <div style="color: #fff; font-weight: 500;">
            ${item.explanation?.decision_message || decisionMsg}
          </div>
        </div>

        <!-- Telemetry Details Table -->
        <div class="specs-meta-table">
          <div class="specs-meta-row">
            <span class="specs-lbl">Filename:</span>
            <span class="specs-val font-mono" style="font-size: 0.78rem;">${item.filename}</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Anomaly Score (s):</span>
            <span class="specs-val font-mono text-cyan">${score.toFixed(4)}</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Decision Threshold (τ):</span>
            <span class="specs-val font-mono">${threshold.toFixed(4)}</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Score Delta (Δ):</span>
            <span class="specs-val font-mono" style="color: ${scoreDelta >= 0 ? 'var(--red)' : 'var(--green)'};">${deltaStr}</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Anomalous Area:</span>
            <span class="specs-val font-mono">${areaPct}%</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Confidence:</span>
            <span class="specs-val font-mono" style="font-size: 0.75rem;">${confDisplay}</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Inference Latency:</span>
            <span class="specs-val font-mono">${item.processing_time_ms || 14.8} ms</span>
          </div>
          <div class="specs-meta-row">
            <span class="specs-lbl">Timestamp:</span>
            <span class="specs-val font-mono" style="font-size: 0.75rem;">${item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}</span>
          </div>
        </div>

        <!-- Diagnostics Notes -->
        <div style="background: var(--bg-input); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5;">
          <div style="font-weight: 700; color: #fff; margin-bottom: 2px;">Triage Diagnostics:</div>
          <div>${item.notes || 'Component evaluated against verified normal coreset distribution.'}</div>
          <div style="margin-top: 6px; font-size: 0.7rem; color: var(--text-muted); font-style: italic;">
            * Heuristic visual estimate prototype only. Not a verified physical damage measurement.
          </div>
        </div>

        <!-- Action Links -->
        <div style="display: flex; gap: 10px; margin-top: auto;">
          <a href="results.html?id=${item.inspection_id}" class="btn btn-primary" style="flex: 1;">
            <i class="fas fa-arrows-left-right"></i> Open Report & Visualizer
          </a>
          <button class="btn btn-secondary" onclick="window.downloadRecordJson('${item.inspection_id}')">
            <i class="fas fa-file-code"></i> JSON
          </button>
        </div>
      </div>
    `;

    drawer.classList.add('active');
    if (backdrop) backdrop.classList.add('active');
  };

  window.openInspectionDrawerById = async function (id) {
    if (!id) return;
    const item = await window.DefectXApi.getInspectionById(id);
    if (item) {
      window.openInspectionDrawer(item);
    } else if (window.showToast) {
      window.showToast('Not Found', `Record ${id} not located.`, 'warning');
    }
  };

  window.closeInspectionDrawer = function () {
    const drawer = document.getElementById('inspection-detail-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (drawer) drawer.classList.remove('active');
    if (backdrop) backdrop.classList.remove('active');
  };

  window.downloadRecordJson = async function (id) {
    const record = await window.DefectXApi.getInspectionById(id);
    if (!record) return;
    const str = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(record, null, 2));
    const a = document.createElement('a');
    a.href = str;
    a.download = `${record.inspection_id}_verdict.json`;
    a.click();
  };

  // 7. Modal Handlers
  window.openModal = function (id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add('active');
  };

  window.closeModal = function (id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove('active');
  };

})(window, document);
