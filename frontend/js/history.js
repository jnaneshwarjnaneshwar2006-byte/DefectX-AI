/**
 * DEFECTX AI - History Controller with Detail Drawer (history.js)
 * CLOCKIT 2026 Hackathon
 */

(function () {
  'use strict';

  let allItems = [];
  let filteredItems = [];
  let currentPage = 1;
  const PAGE_SIZE = 10;

  document.addEventListener('DOMContentLoaded', async () => {
    // Check URL parameters for preset search (e.g. from Ctrl+K)
    const urlParams = new URLSearchParams(window.location.search);
    const initialSearch = urlParams.get('search');
    if (initialSearch) {
      const searchInput = document.getElementById('filter-search');
      if (searchInput) searchInput.value = initialSearch;
    }

    await fetchAndRenderHistory();
    setupFilterListeners();
    setupExportListeners();
  });

  async function fetchAndRenderHistory() {
    const searchVal = document.getElementById('filter-search').value.trim();
    const predVal = document.getElementById('filter-prediction').value;
    const sevVal = document.getElementById('filter-severity').value;
    const sortVal = document.getElementById('filter-sort').value;

    const res = await window.DefectXApi.getHistory({
      search: searchVal,
      prediction: predVal,
      severity: sevVal
    });

    allItems = res.items || [];

    // Sorting
    if (sortVal === 'score_desc') {
      allItems.sort((a, b) => (b.anomaly_score || 0) - (a.anomaly_score || 0));
    } else if (sortVal === 'latency_asc') {
      allItems.sort((a, b) => (a.processing_time_ms || 14) - (b.processing_time_ms || 14));
    }

    filteredItems = allItems;
    currentPage = 1;
    renderTable();
  }

  function renderTable() {
    const tbody = document.getElementById('history-tbody');
    const countEl = document.getElementById('table-record-count');
    const infoEl = document.getElementById('pagination-info');
    const btnPrev = document.getElementById('btn-prev-page');
    const btnNext = document.getElementById('btn-next-page');

    countEl.textContent = filteredItems.length;

    if (filteredItems.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 30px;">No matching records found.</td></tr>`;
      infoEl.textContent = 'Showing 0 to 0 of 0 entries';
      btnPrev.disabled = true;
      btnNext.disabled = true;
      return;
    }

    const totalPages = Math.ceil(filteredItems.length / PAGE_SIZE);
    const startIdx = (currentPage - 1) * PAGE_SIZE;
    const endIdx = Math.min(startIdx + PAGE_SIZE, filteredItems.length);
    const pageItems = filteredItems.slice(startIdx, endIdx);

    infoEl.textContent = `Showing ${startIdx + 1} to ${endIdx} of ${filteredItems.length} entries`;
    btnPrev.disabled = currentPage <= 1;
    btnNext.disabled = currentPage >= totalPages;

    tbody.innerHTML = pageItems.map(item => {
      const isAnomaly = item.prediction === 'ANOMALY';
      const badgeClass = isAnomaly ? 'badge-danger' : 'badge-success';
      const badgeIcon = isAnomaly ? 'fa-triangle-exclamation' : 'fa-check';

      let sevBadge = '<span class="badge badge-success">NONE</span>';
      if (item.severity === 'LOW') sevBadge = '<span class="badge badge-warning">LOW</span>';
      if (item.severity === 'MEDIUM') sevBadge = '<span class="badge" style="background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.3);">MED</span>';
      if (item.severity === 'HIGH' || item.severity === 'CRITICAL') sevBadge = '<span class="badge badge-danger">CRITICAL</span>';

      const timeFormatted = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A';
      const thumbSrc = item.original_image || 'assets/images/sample-broken-large.png';

      return `
        <tr>
          <td>
            <a href="javascript:void(0)" onclick="window.openInspectionDrawerById('${item.inspection_id}')" class="font-mono text-cyan" style="font-weight: 700; text-decoration: none;">
              ${item.inspection_id}
            </a>
          </td>
          <td>
            <div style="display: flex; align-items: center; gap: 10px;">
              <img src="${thumbSrc}" alt="Specimen Thumbnail" style="width: 38px; height: 38px; object-fit: cover; border-radius: 4px; border: 1px solid var(--border-panel); background: #000; flex-shrink: 0;" onerror="this.src='assets/icons/favicon.svg'">
              <div style="min-width: 0;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px;">${item.filename}</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">${item.source || item.defect_type || 'industrial'}</div>
              </div>
            </div>
          </td>
          <td>
            <span style="font-size: 0.75rem; color: var(--text-secondary);">${timeFormatted}</span>
          </td>
          <td>
            <span class="badge ${badgeClass}">
              <i class="fas ${badgeIcon}"></i> ${item.prediction}
            </span>
          </td>
          <td>
            <span class="font-mono" style="font-weight: 700;">${(item.anomaly_score || 0).toFixed(4)}</span>
            <span style="font-size: 0.7rem; color: var(--text-muted);">/ ${(item.threshold || 0.6366).toFixed(2)}</span>
          </td>
          <td>${sevBadge}</td>
          <td>
            <span class="font-mono" style="font-size: 0.75rem; color: var(--text-secondary);">
              ${item.device || 'cuda:0'} • ${item.processing_time_ms || 14.2}ms
            </span>
          </td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-primary btn-sm" onclick="window.openInspectionDrawerById('${item.inspection_id}')" title="View Inspection Details">
                <i class="fas fa-eye"></i> View Details
              </button>
              <a href="results.html?id=${item.inspection_id}" class="btn btn-secondary btn-sm" title="Open Full Report">
                <i class="fas fa-file-invoice"></i>
              </a>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  function setupFilterListeners() {
    const searchInput = document.getElementById('filter-search');
    const predSelect = document.getElementById('filter-prediction');
    const sevSelect = document.getElementById('filter-severity');
    const sortSelect = document.getElementById('filter-sort');
    const btnClear = document.getElementById('btn-clear-filters');
    const btnPrev = document.getElementById('btn-prev-page');
    const btnNext = document.getElementById('btn-next-page');

    searchInput.addEventListener('input', () => fetchAndRenderHistory());
    predSelect.addEventListener('change', () => fetchAndRenderHistory());
    sevSelect.addEventListener('change', () => fetchAndRenderHistory());
    sortSelect.addEventListener('change', () => fetchAndRenderHistory());

    btnClear.addEventListener('click', () => {
      searchInput.value = '';
      predSelect.value = 'ALL';
      sevSelect.value = 'ALL';
      sortSelect.value = 'newest';
      fetchAndRenderHistory();
      if (window.showToast) window.showToast('Filters Reset', 'Displaying all recorded inspections.', 'info');
    });

    btnPrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderTable();
      }
    });

    btnNext.addEventListener('click', () => {
      const totalPages = Math.ceil(filteredItems.length / PAGE_SIZE);
      if (currentPage < totalPages) {
        currentPage++;
        renderTable();
      }
    });
  }

  function setupExportListeners() {
    // Export CSV
    const btnCsv = document.getElementById('btn-export-csv');
    if (btnCsv) {
      btnCsv.addEventListener('click', () => {
        if (!filteredItems.length) return;
        const headers = ['inspection_id', 'filename', 'timestamp', 'prediction', 'anomaly_score', 'threshold', 'severity', 'confidence', 'latency_ms'];
        const csvRows = [headers.join(',')];

        filteredItems.forEach(item => {
          const row = [
            item.inspection_id,
            `"${item.filename}"`,
            item.timestamp,
            item.prediction,
            item.anomaly_score,
            item.threshold,
            item.severity,
            item.confidence,
            item.processing_time_ms
          ];
          csvRows.push(row.join(','));
        });

        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `defectx_inspections_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        if (window.showToast) window.showToast('CSV Exported', 'Downloaded inspection log table.', 'success');
      });
    }

    // Export JSON
    const btnJson = document.getElementById('btn-export-all-json');
    if (btnJson) {
      btnJson.addEventListener('click', () => {
        if (!filteredItems.length) return;
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(filteredItems, null, 2));
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = `defectx_inspections_${Date.now()}.json`;
        a.click();
        if (window.showToast) window.showToast('JSON Exported', 'Downloaded full records dataset.', 'success');
      });
    }
  }

})();
