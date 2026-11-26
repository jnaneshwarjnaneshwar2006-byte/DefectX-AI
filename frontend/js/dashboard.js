/**
 * DEFECTX AI - Enterprise Dashboard Controller (dashboard.js)
 * CLOCKIT 2026 Hackathon
 */

(function () {
  'use strict';

  let trendChartInstance = null;
  let severityChartInstance = null;

  document.addEventListener('DOMContentLoaded', async () => {
    await loadDashboardData();
    setupDashboardActions();
  });

  async function loadDashboardData() {
    try {
      const data = await window.DefectXApi.getAnalytics();
      renderStats(data);
      renderCharts(data);
      renderRecentTable(data.recentItems);
      renderSpotlight(data.recentItems);
    } catch (err) {
      console.error('Failed to load dashboard telemetry:', err);
      if (window.showToast) {
        window.showToast('Telemetry Error', 'Unable to retrieve real-time analytics.', 'error');
      }
    }
  }

  function renderStats(data) {
    document.getElementById('stat-total-count').textContent = data.totalInspections;
    document.getElementById('stat-normal-count').textContent = data.normalCount;
    document.getElementById('stat-defect-count').textContent = data.defectCount;
    document.getElementById('stat-avg-latency').textContent = `${data.averageLatencyMs} ms`;

    const total = data.totalInspections || 1;
    const normalPct = ((data.normalCount / total) * 100).toFixed(1);
    const defectPct = ((data.defectCount / total) * 100).toFixed(1);

    document.getElementById('stat-normal-percent').textContent = `${normalPct}%`;
    document.getElementById('stat-defect-percent').textContent = `${defectPct}%`;
  }

  function renderCharts(data) {
    // 1. Line Trend Chart
    const ctxTrend = document.getElementById('dashboardTrendChart');
    if (ctxTrend) {
      if (trendChartInstance) trendChartInstance.destroy();

      const days = ['Nov 19', 'Nov 20', 'Nov 21', 'Nov 22', 'Nov 23', 'Nov 24', 'Nov 25'];
      const volumeData = [45, 62, 58, 74, 80, 95, Math.max(data.totalInspections * 6, 85)];
      const defectData = [3, 4, 2, 6, 5, 8, Math.max(data.defectCount * 2, 6)];

      trendChartInstance = new Chart(ctxTrend, {
        type: 'line',
        data: {
          labels: days,
          datasets: [
            {
              label: 'Total Inspected',
              data: volumeData,
              borderColor: '#00F2FE',
              backgroundColor: 'rgba(0, 242, 254, 0.08)',
              borderWidth: 2,
              tension: 0.35,
              fill: true,
              pointBackgroundColor: '#00F2FE',
              pointBorderColor: '#090E1A',
              pointRadius: 4
            },
            {
              label: 'Defects Detected',
              data: defectData,
              borderColor: '#EF4444',
              backgroundColor: 'rgba(239, 68, 68, 0.05)',
              borderWidth: 2,
              tension: 0.35,
              fill: true,
              pointBackgroundColor: '#EF4444',
              pointBorderColor: '#090E1A',
              pointRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: {
                color: '#94A3B8',
                font: { family: 'Inter', size: 11 },
                boxWidth: 14
              }
            },
            tooltip: {
              backgroundColor: 'rgba(14, 22, 38, 0.95)',
              titleColor: '#F8FAFC',
              bodyColor: '#94A3B8',
              borderColor: 'rgba(0, 242, 254, 0.3)',
              borderWidth: 1
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } }
            },
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } }
            }
          }
        }
      });
    }

    // 2. Donut Chart
    const ctxSeverity = document.getElementById('dashboardSeverityChart');
    if (ctxSeverity) {
      if (severityChartInstance) severityChartInstance.destroy();

      const sev = data.severities || { NONE: 1, LOW: 1, MEDIUM: 1, HIGH: 0, CRITICAL: 1 };

      severityChartInstance = new Chart(ctxSeverity, {
        type: 'doughnut',
        data: {
          labels: ['Normal (Pass)', 'Low Severity', 'Medium', 'Critical Defect'],
          datasets: [
            {
              data: [sev.NONE || 1, sev.LOW || 1, sev.MEDIUM || 1, (sev.CRITICAL || 0) + (sev.HIGH || 0) || 1],
              backgroundColor: [
                '#10B981', // Green
                '#F59E0B', // Amber
                '#F97316', // Orange
                '#EF4444'  // Red
              ],
              borderColor: '#090E1A',
              borderWidth: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '70%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                color: '#94A3B8',
                font: { family: 'Inter', size: 11 },
                padding: 10,
                boxWidth: 12
              }
            }
          }
        }
      });
    }
  }

  function renderRecentTable(items) {
    const tbody = document.getElementById('recent-inspections-tbody');
    if (!tbody) return;

    if (!items || items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">No inspections recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = items.map(item => {
      const isAnomaly = item.prediction === 'ANOMALY';
      const badgeClass = isAnomaly ? 'badge-danger' : 'badge-success';
      const badgeIcon = isAnomaly ? 'fa-triangle-exclamation' : 'fa-check';

      let sevBadge = '<span class="badge badge-success">NONE</span>';
      if (item.severity === 'LOW') sevBadge = '<span class="badge badge-warning">LOW</span>';
      if (item.severity === 'MEDIUM') sevBadge = '<span class="badge" style="background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.3);">MED</span>';
      if (item.severity === 'HIGH' || item.severity === 'CRITICAL') sevBadge = '<span class="badge badge-danger">CRITICAL</span>';

      const timeFormatted = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Recent';

      return `
        <tr>
          <td>
            <a href="javascript:void(0)" onclick="window.openInspectionDrawerById('${item.inspection_id}')" class="font-mono text-cyan" style="font-weight: 700; text-decoration: none;">
              ${item.inspection_id}
            </a>
          </td>
          <td>
            <div style="font-weight: 600; font-size: 0.85rem; color: #fff;">${item.filename}</div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">${timeFormatted}</div>
          </td>
          <td>
            <span class="badge ${badgeClass}">
              <i class="fas ${badgeIcon}"></i> ${item.prediction}
            </span>
          </td>
          <td>
            <span class="font-mono" style="font-weight: 700; font-size: 0.82rem;">${(item.anomaly_score || 0).toFixed(2)}</span>
            <span style="font-size: 0.7rem; color: var(--text-muted);">/ 3.50</span>
          </td>
          <td>${sevBadge}</td>
          <td>
            <span class="font-mono" style="font-size: 0.78rem; color: var(--text-secondary);">
              ${item.processing_time_ms || 14.2}ms
            </span>
          </td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-secondary btn-sm" onclick="window.openInspectionDrawerById('${item.inspection_id}')" title="Quick Slide Drawer">
                <i class="fas fa-sidebar"></i> Details
              </button>
              <a href="results.html?id=${item.inspection_id}" class="btn btn-secondary btn-sm" title="Open Full Wipe Visualizer">
                <i class="fas fa-eye"></i>
              </a>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  function renderSpotlight(items) {
    if (!items || !items.length) return;
    const latestDefect = items.find(i => i.prediction === 'ANOMALY') || items[0];

    const filenameEl = document.getElementById('spotlight-filename');
    const scoreEl = document.getElementById('spotlight-score');
    const sevEl = document.getElementById('spotlight-severity');
    const linkEl = document.getElementById('spotlight-link');
    const imgEl = document.getElementById('spotlight-img');

    if (filenameEl) filenameEl.textContent = latestDefect.filename;
    if (scoreEl) scoreEl.textContent = `${(latestDefect.anomaly_score || 0).toFixed(4)} (Thresh 3.50)`;
    if (sevEl) sevEl.textContent = latestDefect.severity || 'CRITICAL';
    if (linkEl) linkEl.href = `results.html?id=${latestDefect.inspection_id}`;
    if (imgEl && (latestDefect.overlay_path || latestDefect.heatmap_path)) {
      imgEl.src = latestDefect.overlay_path || latestDefect.heatmap_path;
    }
  }

  // Quick lookup helper for drawer
  window.openInspectionDrawerById = async function (id) {
    const item = await window.DefectXApi.getInspectionById(id);
    if (item && window.openInspectionDrawer) {
      window.openInspectionDrawer(item);
    }
  };

  function setupDashboardActions() {
    // Reset Seed
    const btnReset = document.getElementById('btn-reset-demo');
    if (btnReset) {
      btnReset.addEventListener('click', async () => {
        if (confirm('Reset to standard MVTec benchmark inspection seed?')) {
          window.DefectXApi.resetSeedData();
          await loadDashboardData();
          if (window.showToast) {
            window.showToast('Seed Restored', 'Benchmark inspection records re-initialized.', 'info');
          }
        }
      });
    }

    // Export Telemetry
    const btnExport = document.getElementById('btn-export-dashboard-json');
    if (btnExport) {
      btnExport.addEventListener('click', async () => {
        const data = await window.DefectXApi.getAnalytics();
        const jsonStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
        const a = document.createElement('a');
        a.href = jsonStr;
        a.download = `defectx_daily_telemetry_${Date.now()}.json`;
        a.click();
        if (window.showToast) {
          window.showToast('Report Exported', 'Telemetry snapshot downloaded as JSON.', 'success');
        }
      });
    }
  }

})();
