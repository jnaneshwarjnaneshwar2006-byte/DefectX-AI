/**
 * DEFECTX AI - Analytics & Telemetry Controller (analytics.js)
 * CLOCKIT 2026 Hackathon
 */

(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', async () => {
    initScoreDistChart();
    initDefectTypeChart();
    initLatencyBenchmarkChart();
  });

  // 1. Anomaly Score Distribution Histogram
  function initScoreDistChart() {
    const ctx = document.getElementById('scoreDistChart');
    if (!ctx) return;

    const scoreBins = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5 (Thresh)', '4.0', '5.0', '6.0', '7.0', '8.0+'];
    const normalDistribution = [2, 6, 11, 5, 2, 0, 0, 0, 0, 0, 0, 0];
    const defectDistribution = [0, 0, 0, 0, 0, 0, 0, 4, 12, 19, 16, 12];

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: scoreBins,
        datasets: [
          {
            label: 'Normal Validation Samples (Within Tolerance)',
            data: normalDistribution,
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#10B981',
            borderWidth: 1,
            borderRadius: 3
          },
          {
            label: 'Defective Test Samples (Unseen Anomalies)',
            data: defectDistribution,
            backgroundColor: 'rgba(239, 68, 68, 0.75)',
            borderColor: '#EF4444',
            borderWidth: 1,
            borderRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
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
            ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } },
            title: {
              display: true,
              text: 'Image Frequency',
              color: '#64748B',
              font: { family: 'Inter', size: 10 }
            }
          }
        }
      }
    });
  }

  // 2. Defect Morphology Distribution (Polar Area / Doughnut)
  function initDefectTypeChart() {
    const ctx = document.getElementById('defectTypeChart');
    if (!ctx) return;

    new Chart(ctx, {
      type: 'polarArea',
      data: {
        labels: ['Broken Large (Fracture)', 'Broken Small (Chipping)', 'Contamination (Foreign Object)', 'Normal (Conforming)'],
        datasets: [
          {
            data: [22, 21, 20, 20],
            backgroundColor: [
              'rgba(239, 68, 68, 0.75)',
              'rgba(245, 158, 11, 0.75)',
              'rgba(249, 115, 22, 0.75)',
              'rgba(16, 185, 129, 0.75)'
            ],
            borderColor: '#090E1A',
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { display: false }
          }
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#94A3B8',
              font: { family: 'Inter', size: 10 },
              padding: 10,
              boxWidth: 10
            }
          }
        }
      }
    });
  }

  // 3. Hardware Latency Profile (GPU vs CPU)
  function initLatencyBenchmarkChart() {
    const ctx = document.getElementById('latencyChart');
    if (!ctx) return;

    const stages = ['Preprocessing', 'ResNet Embed', 'Coreset Search', 'Heatmap Gen', 'Total Pipeline'];
    const gpuTimes = [1.8, 4.2, 4.9, 3.3, 14.2];
    const cpuTimes = [3.5, 32.4, 48.0, 18.2, 102.1];

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: stages,
        datasets: [
          {
            label: 'NVIDIA RTX 4050 CUDA (ms)',
            data: gpuTimes,
            backgroundColor: 'rgba(0, 242, 254, 0.8)',
            borderColor: '#00F2FE',
            borderWidth: 1,
            borderRadius: 3
          },
          {
            label: 'Standard x86 CPU Baseline (ms)',
            data: cpuTimes,
            backgroundColor: 'rgba(148, 163, 184, 0.35)',
            borderColor: '#94A3B8',
            borderWidth: 1,
            borderRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#94A3B8',
              font: { family: 'Inter', size: 10 },
              boxWidth: 12
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 9 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } },
            title: {
              display: true,
              text: 'Latency (ms)',
              color: '#64748B',
              font: { family: 'Inter', size: 10 }
            }
          }
        }
      }
    });
  }

})();
