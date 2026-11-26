/**
 * DEFECTX AI - Mobile Camera Live Inspection Studio Controller
 * CLOCKIT 2026 Hackathon
 */

(function () {
  'use strict';

  // Sample SVG Data URLs representing genuine MVTec bottles for evaluation
  const SAMPLE_PRESETS = {
    normal: {
      filename: 'bottle_normal_test_018.png',
      type: 'normal',
      isAnomaly: false,
      svg: generateBottleSvg('normal')
    },
    broken_large: {
      filename: 'bottle_broken_large_001.png',
      type: 'broken_large',
      isAnomaly: true,
      svg: generateBottleSvg('broken_large')
    },
    broken_small: {
      filename: 'bottle_broken_small_007.png',
      type: 'broken_small',
      isAnomaly: true,
      svg: generateBottleSvg('broken_small')
    },
    contamination: {
      filename: 'bottle_contamination_004.png',
      type: 'contamination',
      isAnomaly: true,
      svg: generateBottleSvg('contamination')
    }
  };

  function generateBottleSvg(variant) {
    let defectOverlay = '';
    let mainColor = '#00F2FE';

    if (variant === 'broken_large') {
      mainColor = '#EF4444';
      defectOverlay = `
        <path d="M120 180 L95 230 L145 260 L110 320" stroke="#FF4757" stroke-width="6" stroke-linecap="round"/>
        <circle cx="115" cy="245" r="28" fill="rgba(255, 71, 87, 0.35)"/>
      `;
    } else if (variant === 'broken_small') {
      mainColor = '#F59E0B';
      defectOverlay = `
        <circle cx="170" cy="55" r="14" fill="#FF4757" opacity="0.8"/>
        <path d="M165 48 L175 62" stroke="#FFF" stroke-width="3"/>
      `;
    } else if (variant === 'contamination') {
      mainColor = '#F97316';
      defectOverlay = `
        <circle cx="150" cy="290" r="16" fill="#00F2FE"/>
        <circle cx="165" cy="305" r="10" fill="#EF4444"/>
        <circle cx="140" cy="315" r="8" fill="#F59E0B"/>
      `;
    }

    const svgString = `
      <svg width="400" height="400" viewBox="0 0 320 400" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="320" height="400" fill="#060A14"/>
        <line x1="20" y1="200" x2="300" y2="200" stroke="rgba(0,242,254,0.1)" stroke-width="1"/>
        <line x1="160" y1="20" x2="160" y2="380" stroke="rgba(0,242,254,0.1)" stroke-width="1"/>
        <rect x="135" y="40" width="50" height="55" rx="3" fill="#0D182E" stroke="${mainColor}" stroke-width="2.5"/>
        <rect x="130" y="32" width="60" height="12" rx="3" fill="${mainColor}" opacity="0.8"/>
        <path d="M135 95 C135 125 70 155 70 200 L70 340 C70 355 85 370 100 370 L220 370 C235 370 250 355 250 340 L250 200 C250 155 185 125 185 95 Z" fill="#091122" stroke="${mainColor}" stroke-width="3"/>
        <path d="M74 230 C120 240 200 220 246 230 L246 340 C246 352 232 366 220 366 L100 366 C88 366 74 352 74 340 Z" fill="rgba(0, 242, 254, 0.12)"/>
        ${defectOverlay}
      </svg>
    `;
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgString);
  }

  // State Management
  let activeMode = 'camera'; // 'camera' | 'upload'
  let currentImagePayload = null;
  let currentFilename = 'SPEC-BTL-001_good_000.png';
  let isCurrentDefectHint = null;
  let capturedBlob = null;
  let lastInspectionResult = null;
  const selectedCategory = 'bottle'; // Active production AI model is exclusively MVTec Bottle

  // Industrial Specimen Demonstration Library State
  let specimenManifest = [];
  let selectedSpecimen = null;
  let activeCategoryTab = 'bottle'; // 'bottle' | 'candle' | 'wood'
  let activeFilter = 'all'; // 'all' | 'normal' | 'defective'
  let currentSearchQuery = '';
  const inspectionHistoryLog = new Map(); // specimenId -> { groundTruth, aiPrediction, isMatch }

  // Media Stream State
  let activeMediaStream = null;
  let currentFacingMode = 'environment'; // environment for mobile rear, fallback to laptop webcam
  let availableVideoDevices = [];
  let selectedDeviceId = null;

  document.addEventListener('DOMContentLoaded', () => {
    initCameraControls();
    initModeSwitching();
    initSpecimenLibrary();
    initDropzone();
    initRunInspectionTrigger();
    initStageViewTabs();

    // Default: Check if on mobile
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (isMobile) {
      setMode('camera');
    }
  });

  // =========================================================================
  // 1. Mode Switching (Camera vs Upload Library)
  // =========================================================================
  function initModeSwitching() {
    const tabCamera = document.getElementById('tab-mode-camera');
    const tabUpload = document.getElementById('tab-mode-upload');

    if (tabCamera) {
      tabCamera.addEventListener('click', () => setMode('camera'));
    }
    if (tabUpload) {
      tabUpload.addEventListener('click', () => setMode('upload'));
    }
  }

  function setMode(mode) {
    activeMode = mode;
    const tabCamera = document.getElementById('tab-mode-camera');
    const tabUpload = document.getElementById('tab-mode-upload');
    const secCamera = document.getElementById('section-camera-mode');
    const secUpload = document.getElementById('section-upload-mode');
    const btnRunDesktop = document.getElementById('btn-run-inspection');

    if (mode === 'camera') {
      tabCamera.classList.add('active');
      tabUpload.classList.remove('active');
      secCamera.style.display = 'flex';
      secUpload.style.display = 'none';
      if (btnRunDesktop) btnRunDesktop.style.display = 'none'; // in camera mode, scan button is inside camera card
    } else {
      tabCamera.classList.remove('active');
      tabUpload.classList.add('active');
      secCamera.style.display = 'none';
      secUpload.style.display = 'flex';
      if (btnRunDesktop) btnRunDesktop.style.display = 'flex';
      stopCameraStream();
    }
  }

  // =========================================================================
  // 2. Optical Camera Implementation (Laptop Webcam + Mobile Rear Sensor)
  // =========================================================================
  function initCameraControls() {
    const btnStart = document.getElementById('btn-start-camera');
    const btnCapture = document.getElementById('btn-capture-frame');
    const btnSwitch = document.getElementById('btn-switch-camera');
    const btnStop = document.getElementById('btn-stop-camera');
    const btnRetake = document.getElementById('btn-retake-frame');
    const btnScan = document.getElementById('btn-scan-frame');
    const deviceSelect = document.getElementById('camera-device-select');

    if (btnStart) btnStart.addEventListener('click', startCamera);
    if (btnCapture) btnCapture.addEventListener('click', captureFrame);
    if (btnSwitch) btnSwitch.addEventListener('click', switchCamera);
    if (btnStop) btnStop.addEventListener('click', stopCameraStream);
    if (btnRetake) btnRetake.addEventListener('click', retakeFrame);
    if (btnScan) btnScan.addEventListener('click', () => executeInspection('MOBILE_CAMERA'));

    if (deviceSelect) {
      deviceSelect.addEventListener('change', (e) => {
        selectedDeviceId = e.target.value;
        startCamera();
      });
    }

    refreshDeviceList();

    // Stop camera when navigating away
    window.addEventListener('beforeunload', stopCameraStream);
    window.addEventListener('pagehide', stopCameraStream);
  }

  async function refreshDeviceList() {
    const btnSwitch = document.getElementById('btn-switch-camera');
    const deviceSelect = document.getElementById('camera-device-select');

    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      availableVideoDevices = devices.filter(d => d.kind === 'videoinput');

      if (deviceSelect) {
        if (availableVideoDevices.length > 1) {
          deviceSelect.innerHTML = '';
          availableVideoDevices.forEach((d, idx) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || `Camera ${idx + 1}`;
            if (selectedDeviceId && d.deviceId === selectedDeviceId) opt.selected = true;
            deviceSelect.appendChild(opt);
          });
          deviceSelect.style.display = 'inline-block';
        } else {
          deviceSelect.style.display = 'none';
        }
      }

      if (btnSwitch) {
        btnSwitch.style.display = (availableVideoDevices.length > 1) ? 'inline-flex' : 'none';
      }
    } catch (e) {
      // Ignore enumeration errors until permission is granted
    }
  }

  async function startCamera() {
    const errorBanner = document.getElementById('camera-error-banner');
    const errorText = document.getElementById('camera-error-text');
    const promptCard = document.getElementById('camera-prompt-card');
    const videoElem = document.getElementById('camera-video');
    const reticle = document.getElementById('camera-reticle');
    const liveControls = document.getElementById('camera-live-controls');
    const reviewArea = document.getElementById('captured-review-area');
    const statusText = document.getElementById('camera-status-text');
    const statusDot = document.getElementById('camera-status-dot');
    const feedBadge = document.getElementById('camera-feed-badge');

    if (errorBanner) errorBanner.style.display = 'none';

    // 1. Verify Browser Support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraError('Your browser does not support the MediaDevices camera API. Please use Microsoft Edge or Google Chrome.');
      return;
    }

    // 2. Verify Secure Context (Chrome/Edge require HTTPS or localhost)
    const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!window.isSecureContext && !isLocalhost) {
      showCameraError('Camera access requires HTTPS or localhost. For mobile testing, run run_mobile.ps1 (which configures an ADB reverse tunnel to http://localhost:8080) or serve over HTTPS.');
      return;
    }

    // Stop existing stream and tracks before requesting a new one
    stopCameraStream();

    // 3. Build constraints: prefer explicit deviceId if selected, otherwise ideal facingMode
    const videoConstraints = {
      width: { ideal: 1920, min: 640 },
      height: { ideal: 1080, min: 480 }
    };

    if (selectedDeviceId) {
      videoConstraints.deviceId = { exact: selectedDeviceId };
    } else {
      videoConstraints.facingMode = { ideal: currentFacingMode };
    }

    const constraints = {
      video: videoConstraints,
      audio: false
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      activeMediaStream = stream;

      // Configure video element properly per browser video standards
      videoElem.autoplay = true;
      videoElem.muted = true;
      videoElem.playsInline = true;
      videoElem.setAttribute('playsinline', '');
      videoElem.setAttribute('webkit-playsinline', '');
      videoElem.srcObject = stream;

      // Wait for loadedmetadata and video play
      await new Promise((resolve, reject) => {
        const onLoaded = () => {
          videoElem.removeEventListener('loadedmetadata', onLoaded);
          resolve();
        };
        videoElem.addEventListener('loadedmetadata', onLoaded);
        setTimeout(resolve, 1500); // safety fallback
      });

      await videoElem.play();

      // Verify that actual video frames are streaming (dimensions > 0)
      if (videoElem.videoWidth === 0 || videoElem.videoHeight === 0) {
        // Poll for up to 1 second for valid dimensions
        await new Promise(r => {
          let count = 0;
          const interval = setInterval(() => {
            count++;
            if (videoElem.videoWidth > 0 || count > 10) {
              clearInterval(interval);
              r();
            }
          }, 100);
        });
      }

      if (videoElem.videoWidth === 0 || videoElem.videoHeight === 0) {
        throw new Error('Camera feed connected but returned 0x0 frame dimensions.');
      }

      // UI State: Live Streaming Verified
      videoElem.style.display = 'block';
      if (reticle) reticle.style.display = 'flex';
      if (promptCard) promptCard.style.display = 'none';
      if (reviewArea) reviewArea.style.display = 'none';
      if (liveControls) liveControls.style.display = 'flex';

      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isRear = (currentFacingMode === 'environment');
      const labelText = isMobile
        ? (isRear ? 'LIVE REAR SENSOR' : 'LIVE FRONT SENSOR')
        : 'LIVE WEBCAM STREAM';

      if (statusText) statusText.textContent = labelText;
      if (statusDot) {
        statusDot.className = 'fas fa-circle text-green';
        statusDot.style.boxShadow = '0 0 8px #10B981';
      }
      if (feedBadge) {
        feedBadge.textContent = `${videoElem.videoWidth}x${videoElem.videoHeight}`;
      }

      // Refresh device labels now that permission has been granted
      refreshDeviceList();

      if (window.showToast) {
        window.showToast('Camera Connected', `${labelText} active (${videoElem.videoWidth}x${videoElem.videoHeight}). Align bottle specimen.`, 'success', 3000);
      }
    } catch (err) {
      console.error('Camera access error:', err);
      let msg = 'Failed to access camera: ' + (err.message || err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission was denied. Please allow camera permissions in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera hardware found on this system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = 'Camera is already in use by another application. Please close other camera programs and try again.';
      } else if (err.name === 'OverconstrainedError') {
        // Fallback to minimal unconstrained stream if ideal resolution failed
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          activeMediaStream = fallbackStream;
          videoElem.srcObject = fallbackStream;
          await videoElem.play();
          videoElem.style.display = 'block';
          if (promptCard) promptCard.style.display = 'none';
          if (liveControls) liveControls.style.display = 'flex';
          if (statusText) statusText.textContent = 'LIVE OPTICAL FEED';
          if (statusDot) {
            statusDot.className = 'fas fa-circle text-green';
            statusDot.style.boxShadow = '0 0 8px #10B981';
          }
          return;
        } catch (fallbackErr) {
          msg = 'Could not initialize camera feed: ' + fallbackErr.message;
        }
      }
      showCameraError(msg);
      stopCameraStream();
    }
  }

  function showCameraError(msg) {
    const errorBanner = document.getElementById('camera-error-banner');
    const errorText = document.getElementById('camera-error-text');
    if (errorBanner && errorText) {
      errorText.textContent = msg;
      errorBanner.style.display = 'flex';
    }
    if (window.showToast) {
      window.showToast('Camera Notice', msg, 'warning', 6000);
    }
  }

  function switchCamera() {
    selectedDeviceId = null;
    currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
    startCamera();
  }

  function stopCameraStream() {
    if (activeMediaStream) {
      activeMediaStream.getTracks().forEach(track => {
        try { track.stop(); } catch (e) { }
      });
      activeMediaStream = null;
    }

    const videoElem = document.getElementById('camera-video');
    const reticle = document.getElementById('camera-reticle');
    const liveControls = document.getElementById('camera-live-controls');
    const promptCard = document.getElementById('camera-prompt-card');
    const statusText = document.getElementById('camera-status-text');
    const statusDot = document.getElementById('camera-status-dot');
    const feedBadge = document.getElementById('camera-feed-badge');

    if (videoElem) {
      try { videoElem.pause(); } catch (e) { }
      videoElem.srcObject = null;
      videoElem.style.display = 'none';
    }
    if (reticle) reticle.style.display = 'none';
    if (liveControls) liveControls.style.display = 'none';
    if (promptCard) promptCard.style.display = 'flex';

    if (statusText) statusText.textContent = 'CAMERA STANDBY';
    if (statusDot) {
      statusDot.className = 'fas fa-circle text-muted';
      statusDot.style.boxShadow = 'none';
    }
    if (feedBadge) feedBadge.textContent = 'SENSOR STANDBY';
  }

  // =========================================================================
  // 3. Capture Frame & Review Stage
  // =========================================================================
  function captureFrame() {
    const videoElem = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const reviewArea = document.getElementById('captured-review-area');
    const liveControls = document.getElementById('camera-live-controls');
    const previewImg = document.getElementById('captured-preview-img');
    const chamberImg = document.getElementById('preview-stage-img');
    const dimsEl = document.getElementById('captured-dims');
    const sizeEl = document.getElementById('captured-size');
    const metaFilename = document.getElementById('meta-filename');
    const placeholder = document.getElementById('preview-placeholder-box');

    if (!videoElem || !videoElem.videoWidth) {
      if (window.showToast) window.showToast('Capture Error', 'Video stream not ready for capture.', 'error');
      return;
    }

    const width = videoElem.videoWidth;
    const height = videoElem.videoHeight;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElem, 0, 0, width, height);

    // Convert frame to Blob and DataURL
    canvas.toBlob((blob) => {
      capturedBlob = blob;
      const sizeKb = (blob.size / 1024).toFixed(1);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.95);

      currentImagePayload = blob;
      currentFilename = `mobile_capture_${Date.now()}.jpg`;
      isCurrentDefectHint = null;

      // Update Review UI
      previewImg.src = dataUrl;
      if (chamberImg) {
        chamberImg.src = dataUrl;
        chamberImg.style.display = 'block';
      }
      if (placeholder) placeholder.style.display = 'none';

      if (dimsEl) dimsEl.textContent = `${width} x ${height} px`;
      if (sizeEl) sizeEl.textContent = `${sizeKb} KB`;
      if (metaFilename) metaFilename.textContent = currentFilename;

      // Switch from live video to captured preview
      videoElem.style.display = 'none';
      const reticle = document.getElementById('camera-reticle');
      if (reticle) reticle.style.display = 'none';
      if (liveControls) liveControls.style.display = 'none';
      if (reviewArea) reviewArea.style.display = 'flex';

      if (window.showToast) {
        window.showToast('Frame Captured', `Resolution: ${width}x${height} (${sizeKb} KB). Inspect and tap SCAN IMAGE.`, 'info', 3000);
      }
    }, 'image/jpeg', 0.95);
  }

  function retakeFrame() {
    capturedBlob = null;
    currentImagePayload = null;
    const reviewArea = document.getElementById('captured-review-area');
    if (reviewArea) reviewArea.style.display = 'none';

    // Clear previous verdict banners
    resetResultsView();

    // Re-engage live camera
    const videoElem = document.getElementById('camera-video');
    const reticle = document.getElementById('camera-reticle');
    const liveControls = document.getElementById('camera-live-controls');

    if (activeMediaStream && videoElem) {
      videoElem.style.display = 'block';
      if (reticle) reticle.style.display = 'flex';
      if (liveControls) liveControls.style.display = 'flex';
    } else {
      startCamera();
    }
  }

  // =========================================================================
  // 4. Run AI Defect Inspection (PHP Backend + Python AI Engine)
  // =========================================================================
  function initRunInspectionTrigger() {
    const btnRunDesktop = document.getElementById('btn-run-inspection');
    if (btnRunDesktop) {
      btnRunDesktop.addEventListener('click', () => executeInspection('WEB_UPLOAD'));
    }
  }

  async function executeInspection(sourceType = 'MOBILE_CAMERA') {
    if (!currentImagePayload) {
      if (window.showToast) {
        window.showToast('No Specimen Loaded', 'Please capture a camera frame or load an industrial specimen.', 'warning');
      }
      return;
    }

    // Strict AI Model Guard: Only Bottle is enabled for live AI inference
    if (selectedSpecimen && selectedSpecimen.category !== 'bottle') {
      const catName = selectedSpecimen.category_display || selectedSpecimen.category;
      if (window.showToast) {
        window.showToast('Model Not Available', `${catName} is currently in PREVIEW ONLY mode. Only MVTec Bottle is trained and enabled for production AI inference.`, 'warning', 5000);
      }
      renderPreviewOnlyNotice(selectedSpecimen);
      return;
    }

    const modal = document.getElementById('inspection-modal');
    if (modal) modal.classList.add('active');

    // Animate 5-step stepper for responsive visual feedback
    animateStepper();

    try {
      const startTime = performance.now();

      // Upload image to PHP backend (/api/inspect.php) -> Python inference
      const result = await window.DefectXApi.inspectImage(currentImagePayload, currentFilename, isCurrentDefectHint, 'bottle');

      const elapsed = (performance.now() - startTime).toFixed(1);
      lastInspectionResult = result;

      // Close modal
      setTimeout(() => {
        if (modal) modal.classList.remove('active');
        renderInspectionResult(result);
      }, 700);

    } catch (err) {
      console.error('Inspection failed:', err);
      if (modal) modal.classList.remove('active');
      renderFailedResult(err.message || 'Inspection failed.');
    }
  }

  function renderPreviewOnlyNotice(spec) {
    const bannerContainer = document.getElementById('result-verdict-banner');
    if (bannerContainer) {
      bannerContainer.style.display = 'block';
      const catName = spec.category_display || spec.category;
      const isUnavailable = (spec.available === false);
      const subMsg = isUnavailable
        ? `The official ${catName} dataset images were not packaged in the local repository. DEFECTX AI does not fabricate fake images or test metrics.`
        : `Genuine ${catName} dataset sample loaded for visual inspection preview. The active production AI model checkpoint (ai-engine/checkpoints/defectx_bottle.pt) is strictly dedicated to MVTec Bottle inspection to prevent category hallucination or uncalibrated false predictions.`;

      bannerContainer.innerHTML = `
        <div class="verdict-banner" style="border-left: 4px solid var(--amber); background: rgba(245, 158, 11, 0.08); flex-direction: column; gap: 10px;">
          <div style="display: flex; align-items: flex-start; gap: 14px;">
            <div class="verdict-icon" style="background: rgba(245, 158, 11, 0.2); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.4);"><i class="fas fa-eye"></i></div>
            <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                <h3 class="verdict-title" style="color: #fde68a;">PREVIEW ONLY — MODEL NOT ACTIVE</h3>
                <span class="badge badge-warning font-mono">SPECIMEN PREVIEW</span>
              </div>
              <p class="verdict-sub" style="margin-top: 6px; font-size: 0.85rem; line-height: 1.5; color: #fff;">
                ${subMsg}
              </p>
              <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.78rem; color: var(--text-secondary);">
                <span><strong>Specimen ID:</strong> ${spec.id}</span>
                <span><strong>Product:</strong> ${catName}</span>
                <span><strong>Ground Truth:</strong> <span class="badge ${spec.label === 'NORMAL' ? 'badge-success' : 'badge-danger'}">${spec.label} (${spec.defect_display || spec.defect_type})</span></span>
                <span><strong>Dataset Source:</strong> ${spec.dataset_source}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }
  }

  function animateStepper() {
    const steps = [
      document.getElementById('step-1'),
      document.getElementById('step-2'),
      document.getElementById('step-3'),
      document.getElementById('step-4'),
      document.getElementById('step-5')
    ];

    steps.forEach((s, idx) => {
      if (!s) return;
      s.className = 'stepper-row';
      setTimeout(() => {
        steps.forEach((el, i) => {
          if (!el) return;
          if (i < idx) el.className = 'stepper-row completed';
          else if (i === idx) el.className = 'stepper-row active';
        });
      }, idx * 180);
    });
  }

  function renderInspectionResult(result) {
    if (!result) return;

    const bannerContainer = document.getElementById('result-verdict-banner');
    const stageTabs = document.getElementById('stage-view-tabs');
    const actionsRow = document.getElementById('post-inspection-actions');
    const scoreRow = document.getElementById('meta-score-row');
    const reportLink = document.getElementById('btn-view-full-report');
    const stageImg = document.getElementById('preview-stage-img');

    const score = parseFloat(result.anomaly_score || 0);
    const threshold = parseFloat(result.threshold || 0.6366);
    const pixelThreshold = parseFloat(result.pixel_threshold || 0.5060);
    const scoreDelta = (score - threshold);
    const isAnomaly = (result.prediction === 'ANOMALY') || (score >= threshold);
    const severity = (result.severity || (isAnomaly ? 'HIGH' : 'NONE')).toUpperCase();
    const anomalyScoreStr = score.toFixed(4);
    const thresholdStr = threshold.toFixed(4);
    const deltaSign = scoreDelta >= 0 ? '+' : '';
    const deltaStr = `${deltaSign}${scoreDelta.toFixed(4)}`;

    const areaRatio = result.anomalous_area_ratio !== undefined ? parseFloat(result.anomalous_area_ratio) : (result.details?.anomalous_area_ratio !== undefined ? parseFloat(result.details.anomalous_area_ratio) : 0);
    const areaPct = (areaRatio * 100).toFixed(2);

    const decisionMsg = isAnomaly
      ? `Anomaly detected because the image anomaly score (${anomalyScoreStr}) exceeded the configured decision threshold (${thresholdStr}).`
      : `No anomaly detected because the image anomaly score (${anomalyScoreStr}) did not exceed the configured decision threshold (${thresholdStr}).`;

    // Dynamic ground-truth match evaluation
    let matchHtml = '';
    const aiVerdictLabel = isAnomaly ? 'DEFECTIVE' : 'NORMAL';
    if (selectedSpecimen && selectedSpecimen.label) {
      const gtLabel = selectedSpecimen.label.toUpperCase();
      const isMatch = (gtLabel === aiVerdictLabel);

      inspectionHistoryLog.set(selectedSpecimen.id, {
        groundTruth: gtLabel,
        aiPrediction: aiVerdictLabel,
        isMatch: isMatch
      });
      updateLibrarySummaryStats();

      const matchBadge = isMatch
        ? '<span class="badge badge-success" style="font-size: 0.8rem;"><i class="fas fa-check-circle"></i> Match: CORRECT</span>'
        : '<span class="badge badge-danger" style="font-size: 0.8rem;"><i class="fas fa-times-circle"></i> Match: DISCREPANCY</span>';

      matchHtml = `
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
          <span style="font-size: 0.78rem; color: var(--text-secondary);"><strong>Specimen ID:</strong> ${selectedSpecimen.id}</span>
          <span style="font-size: 0.78rem; color: var(--text-secondary);"><strong>Ground Truth:</strong> <span class="badge ${gtLabel === 'NORMAL' ? 'badge-success' : 'badge-danger'}">${gtLabel} (${selectedSpecimen.display_type})</span></span>
          <span style="font-size: 0.78rem; color: var(--text-secondary);"><strong>AI Prediction:</strong> <span class="badge ${isAnomaly ? 'badge-danger' : 'badge-success'}">${aiVerdictLabel}</span></span>
          ${matchBadge}
        </div>
      `;
    }

    // 1. Prominent Verdict & AI Decision Explanation Banner
    if (bannerContainer) {
      bannerContainer.style.display = 'block';
      const borderClass = isAnomaly ? 'danger' : 'success';
      const iconClass = isAnomaly ? 'fa-triangle-exclamation' : 'fa-circle-check';
      const title = isAnomaly ? 'ANOMALY DETECTED – DEFECT FOUND' : 'INSPECTION PASSED – NO ANOMALY DETECTED';
      const deltaBadgeClass = isAnomaly ? 'badge-danger' : 'badge-success';

      bannerContainer.innerHTML = `
        <div class="verdict-banner ${borderClass}" style="flex-direction: column; gap: 12px; align-items: stretch;">
          <div style="display: flex; align-items: flex-start; gap: 14px;">
            <div class="verdict-icon"><i class="fas ${iconClass}"></i></div>
            <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                <h3 class="verdict-title">${title}</h3>
                <span class="badge ${deltaBadgeClass} font-mono" style="font-size: 0.75rem;">Δ ${deltaStr}</span>
              </div>
              <p class="verdict-sub" style="margin-top: 6px; font-size: 0.88rem; line-height: 1.45; color: #fff;">
                <strong>Decision Logic:</strong> ${decisionMsg}
              </p>
              ${matchHtml}
            </div>
          </div>

          <!-- Comparative Telemetry Pill Row -->
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; background: rgba(0, 0, 0, 0.25); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); font-size: 0.75rem;">
            <div>
              <span style="color: var(--text-muted); display: block; font-size: 0.68rem; text-transform: uppercase;">Score (s)</span>
              <strong class="font-mono text-cyan" style="font-size: 0.95rem;">${anomalyScoreStr}</strong>
            </div>
            <div>
              <span style="color: var(--text-muted); display: block; font-size: 0.68rem; text-transform: uppercase;">Threshold (τ)</span>
              <strong class="font-mono" style="color: var(--amber); font-size: 0.95rem;">${thresholdStr}</strong>
            </div>
            <div>
              <span style="color: var(--text-muted); display: block; font-size: 0.68rem; text-transform: uppercase;">Area Exceeding</span>
              <strong class="font-mono" style="color: #fff; font-size: 0.95rem;">${areaPct}%</strong>
            </div>
            <div>
              <span style="color: var(--text-muted); display: block; font-size: 0.68rem; text-transform: uppercase;">Severity Tier</span>
              <span class="badge ${isAnomaly ? 'badge-danger' : 'badge-success'}" style="font-size: 0.7rem; padding: 2px 6px;">${severity}</span>
            </div>
            <div>
              <span style="color: var(--text-muted); display: block; font-size: 0.68rem; text-transform: uppercase;">Confidence</span>
              <span style="color: var(--text-secondary); font-size: 0.72rem;">Uncalibrated (Metric)</span>
            </div>
          </div>

          <!-- Heatmap & Severity Prototype Note -->
          <div style="font-size: 0.7rem; color: var(--text-muted); line-height: 1.4; display: flex; flex-direction: column; gap: 2px;">
            <span><i class="fas fa-info-circle text-cyan"></i> Heatmap highlights localized nearest-neighbor feature patch distance; it does not classify defect categories.</span>
            <span><i class="fas fa-triangle-exclamation" style="color: var(--amber);"></i> Heuristic visual estimate prototype only. Not a verified physical damage measurement.</span>
          </div>
        </div>
      `;
    }

    // 2. Enable Stage View Tabs (Overlay / Heatmap / Original)
    if (stageTabs) stageTabs.style.display = 'flex';
    if (actionsRow) actionsRow.style.display = 'flex';

    if (reportLink && result.inspection_id) {
      reportLink.href = `results.html?id=${encodeURIComponent(result.inspection_id)}`;
    }

    // 3. Update Image Display with Blended Overlay
    const overlaySrc = result.overlay_path || result.heatmap_path || result.original_image;
    if (stageImg && overlaySrc) {
      stageImg.src = overlaySrc;
    }

    // 4. Update Specs Table
    if (scoreRow) {
      scoreRow.innerHTML = isAnomaly
        ? `<span class="badge badge-danger">${anomalyScore}</span> <span class="badge badge-danger">${severity}</span>`
        : `<span class="badge badge-success">${anomalyScore}</span> <span class="badge badge-success">PASS</span>`;
    }

    // Toast Notification
    if (window.showToast) {
      const toastType = isAnomaly ? 'error' : 'success';
      const toastTitle = isAnomaly ? 'ANOMALY DETECTED' : 'INSPECTION PASSED';
      window.showToast(toastTitle, `Score: ${anomalyScore} | Severity: ${severity} | ${result.processing_time_ms}ms`, toastType, 5000);
    }
  }

  function renderFailedResult(errMessage) {
    const bannerContainer = document.getElementById('result-verdict-banner');
    if (bannerContainer) {
      bannerContainer.style.display = 'block';
      bannerContainer.innerHTML = `
        <div class="verdict-banner danger">
          <div class="verdict-icon"><i class="fas fa-circle-xmark"></i></div>
          <div style="flex: 1;">
            <h3 class="verdict-title">INSPECTION FAILED – PLEASE TRY AGAIN</h3>
            <p class="verdict-sub">${errMessage || 'An error occurred during Python AI inference execution.'}</p>
          </div>
        </div>
      `;
    }
    if (window.showToast) {
      window.showToast('Inspection Failed', 'INSPECTION FAILED – PLEASE TRY AGAIN', 'error', 5000);
    }
  }

  function resetResultsView() {
    const bannerContainer = document.getElementById('result-verdict-banner');
    const stageTabs = document.getElementById('stage-view-tabs');
    const actionsRow = document.getElementById('post-inspection-actions');
    const scoreRow = document.getElementById('meta-score-row');

    if (bannerContainer) bannerContainer.style.display = 'none';
    if (stageTabs) stageTabs.style.display = 'none';
    if (actionsRow) actionsRow.style.display = 'none';
    if (scoreRow) scoreRow.textContent = 'Awaiting Scan';
  }

  // =========================================================================
  // 5. Stage View Modes (Overlay / Heatmap / Original)
  // =========================================================================
  function initStageViewTabs() {
    const buttons = document.querySelectorAll('[data-stage-view]');
    const stageImg = document.getElementById('preview-stage-img');

    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const view = btn.getAttribute('data-stage-view');
        if (!lastInspectionResult) return;

        if (view === 'overlay') {
          stageImg.src = lastInspectionResult.overlay_path || lastInspectionResult.original_image;
        } else if (view === 'heatmap') {
          stageImg.src = lastInspectionResult.heatmap_path || lastInspectionResult.original_image;
        } else if (view === 'original') {
          stageImg.src = lastInspectionResult.original_image;
        }
      });
    });
  }

  // =========================================================================
  // 6. Industrial Specimen Library (14 MVTec / VisA Samples) & File Dropzone
  // =========================================================================
  let activeCategoryTab = 'bottle';
  let activeFilter = 'all';
  let currentSearchQuery = '';

  async function initSpecimenLibrary() {
    try {
      const resp = await fetch('assets/specimen-library/manifest.json');
      if (resp.ok) {
        specimenManifest = await resp.json();
      }
    } catch (e) {
      console.warn('Could not load assets/specimen-library/manifest.json; fallback to static entries.', e);
    }

    if (!specimenManifest || specimenManifest.length === 0) {
      // Fallback built-in manifest in case of static local file origin
      specimenManifest = generateFallbackManifest();
    }

    initCategoryTabs();
    initSpecimenFilters();
    updateCategoryBanner();
    renderSpecimenGrid(getCurrentFilteredList());
    updateLibrarySummaryStats();

    // Auto-select first specimen by default
    const initialList = getCurrentFilteredList();
    if (initialList.length > 0) {
      selectSpecimen(initialList[0]);
    }
  }

  function generateFallbackManifest() {
    return [
      {
        "id": "SPEC-BTL-001",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Specimen",
        "dataset_source": "MVTec AD Bottle (test/good)",
        "filename": "SPEC-BTL-001_good_000.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-001_good_000.png",
        "available": true
      },
      {
        "id": "SPEC-BTL-002",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Specimen",
        "dataset_source": "MVTec AD Bottle (test/good)",
        "filename": "SPEC-BTL-002_good_001.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-002_good_001.png",
        "available": true
      },
      {
        "id": "SPEC-BTL-003",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "DEFECTIVE",
        "defect_type": "broken_large",
        "defect_display": "Broken Large (Crack)",
        "dataset_source": "MVTec AD Bottle (test/broken_large)",
        "filename": "SPEC-BTL-003_broken_large_000.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-003_broken_large_000.png",
        "available": true
      },
      {
        "id": "SPEC-BTL-004",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "DEFECTIVE",
        "defect_type": "broken_small",
        "defect_display": "Broken Small (Chip)",
        "dataset_source": "MVTec AD Bottle (test/broken_small)",
        "filename": "SPEC-BTL-004_broken_small_000.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-004_broken_small_000.png",
        "available": true
      },
      {
        "id": "SPEC-BTL-005",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "DEFECTIVE",
        "defect_type": "contamination",
        "defect_display": "Contamination (Debris)",
        "dataset_source": "MVTec AD Bottle (test/contamination)",
        "filename": "SPEC-BTL-005_contamination_000.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-005_contamination_000.png",
        "available": true
      },
      {
        "id": "SPEC-BTL-006",
        "category": "bottle",
        "category_display": "Bottle",
        "label": "DEFECTIVE",
        "defect_type": "contamination",
        "defect_display": "Contamination (Foreign Object)",
        "dataset_source": "MVTec AD Bottle (test/contamination)",
        "filename": "SPEC-BTL-006_contamination_001.png",
        "image_path": "assets/specimen-library/bottle/SPEC-BTL-006_contamination_001.png",
        "available": true
      },
      {
        "id": "SPEC-CND-001",
        "category": "candle",
        "category_display": "Candle",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Specimen",
        "dataset_source": "VisA Candle (Data/Images/Normal)",
        "filename": "SPEC-CND-001_normal_0000.jpg",
        "image_path": "assets/specimen-library/candle/SPEC-CND-001_normal_0000.jpg",
        "available": true
      },
      {
        "id": "SPEC-CND-002",
        "category": "candle",
        "category_display": "Candle",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Specimen",
        "dataset_source": "VisA Candle (Data/Images/Normal)",
        "filename": "SPEC-CND-002_normal_0001.jpg",
        "image_path": "assets/specimen-library/candle/SPEC-CND-002_normal_0001.jpg",
        "available": true
      },
      {
        "id": "SPEC-CND-003",
        "category": "candle",
        "category_display": "Candle",
        "label": "DEFECTIVE",
        "defect_type": "structural_defect",
        "defect_display": "Surface / Shape Anomaly",
        "dataset_source": "VisA Candle (Data/Images/Anomaly)",
        "filename": "SPEC-CND-003_anomaly_000.jpg",
        "image_path": "assets/specimen-library/candle/SPEC-CND-003_anomaly_000.jpg",
        "available": true
      },
      {
        "id": "SPEC-CND-004",
        "category": "candle",
        "category_display": "Candle",
        "label": "DEFECTIVE",
        "defect_type": "structural_defect",
        "defect_display": "Wax Imperfection / Discoloration",
        "dataset_source": "VisA Candle (Data/Images/Anomaly)",
        "filename": "SPEC-CND-004_anomaly_001.jpg",
        "image_path": "assets/specimen-library/candle/SPEC-CND-004_anomaly_001.jpg",
        "available": true
      },
      {
        "id": "SPEC-WOD-001",
        "category": "wood",
        "category_display": "Wood",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Wood Specimen",
        "dataset_source": "MVTec AD Wood (test/good)",
        "filename": "wood_good_001.png",
        "image_path": "assets/icons/favicon.svg",
        "available": false
      },
      {
        "id": "SPEC-WOD-002",
        "category": "wood",
        "category_display": "Wood",
        "label": "NORMAL",
        "defect_type": "none",
        "defect_display": "Normal Wood Specimen",
        "dataset_source": "MVTec AD Wood (test/good)",
        "filename": "wood_good_002.png",
        "image_path": "assets/icons/favicon.svg",
        "available": false
      },
      {
        "id": "SPEC-WOD-003",
        "category": "wood",
        "category_display": "Wood",
        "label": "DEFECTIVE",
        "defect_type": "scratch",
        "defect_display": "Surface Scratch Defect",
        "dataset_source": "MVTec AD Wood (test/scratch)",
        "filename": "wood_scratch_001.png",
        "image_path": "assets/icons/favicon.svg",
        "available": false
      },
      {
        "id": "SPEC-WOD-004",
        "category": "wood",
        "category_display": "Wood",
        "label": "DEFECTIVE",
        "defect_type": "hole",
        "defect_display": "Hole / Void Defect",
        "dataset_source": "MVTec AD Wood (test/hole)",
        "filename": "wood_hole_001.png",
        "image_path": "assets/icons/favicon.svg",
        "available": false
      }
    ];
  }

  function initCategoryTabs() {
    const tabs = [
      { id: 'tab-cat-bottle', cat: 'bottle' },
      { id: 'tab-cat-candle', cat: 'candle' },
      { id: 'tab-cat-wood', cat: 'wood' }
    ];

    tabs.forEach(tab => {
      const btn = document.getElementById(tab.id);
      if (!btn) return;
      btn.addEventListener('click', () => {
        tabs.forEach(t => {
          const b = document.getElementById(t.id);
          if (b) b.classList.remove('active');
        });
        btn.classList.add('active');
        activeCategoryTab = tab.cat;
        updateCategoryBanner();
        const filtered = getCurrentFilteredList();
        renderSpecimenGrid(filtered);
        if (filtered.length > 0) {
          selectSpecimen(filtered[0]);
        }
      });
    });
  }

  function updateCategoryBanner() {
    const descEl = document.getElementById('category-status-desc');
    const badgeEl = document.getElementById('category-ai-badge');
    if (!descEl || !badgeEl) return;

    if (activeCategoryTab === 'bottle') {
      descEl.innerHTML = '<span class="text-cyan font-mono"><strong>Active Inspection:</strong> MVTec Bottle Checkpoint (ai-engine/checkpoints/defectx_bottle.pt)</span>';
      badgeEl.className = 'badge badge-success font-mono';
      badgeEl.textContent = 'AI INFERENCE READY';
    } else if (activeCategoryTab === 'candle') {
      descEl.innerHTML = '<span class="text-amber font-mono"><strong>VisA Candle Dataset Specimen:</strong> Genuine dataset samples loaded (PREVIEW ONLY). Model routing isolated.</span>';
      badgeEl.className = 'badge badge-warning font-mono';
      badgeEl.textContent = 'MODEL NOT AVAILABLE';
    } else if (activeCategoryTab === 'wood') {
      descEl.innerHTML = '<span class="text-muted font-mono"><strong>MVTec Wood Dataset Specimen:</strong> Missing in local repository (Reported per specification, no synthetic fabrication).</span>';
      badgeEl.className = 'badge badge-secondary font-mono';
      badgeEl.textContent = 'DATASET UNAVAILABLE';
    }
  }

  function renderSpecimenGrid(items) {
    const container = document.getElementById('specimen-library-container');
    if (!container) return;

    if (items.length === 0) {
      container.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 24px;">No specimens matching current category/filter.</div>';
      return;
    }

    container.innerHTML = items.map(spec => {
      const isSelected = (selectedSpecimen && selectedSpecimen.id === spec.id);
      const isNormal = (spec.label === 'NORMAL');
      const badgeClass = isNormal ? 'badge-success' : 'badge-danger';
      const labelText = isNormal ? 'NORMAL' : (spec.defect_display || spec.display_type || spec.defect_type);
      const borderStyle = isSelected
        ? 'border-color: var(--cyan); box-shadow: 0 0 12px rgba(0, 242, 254, 0.35); background: rgba(0, 242, 254, 0.08);'
        : 'border-color: var(--border-subtle); background: var(--bg-panel-elevated);';

      // Past inspection result indicator
      let inspectedBadge = '';
      if (inspectionHistoryLog.has(spec.id)) {
        const res = inspectionHistoryLog.get(spec.id);
        inspectedBadge = res.isMatch
          ? '<span title="Match: Correct" style="position: absolute; top: 4px; right: 4px; width: 8px; height: 8px; border-radius: 50%; background: var(--green);"></span>'
          : '<span title="Discrepancy" style="position: absolute; top: 4px; right: 4px; width: 8px; height: 8px; border-radius: 50%; background: var(--red);"></span>';
      }

      // Preview / Unavailable ribbon
      let previewTag = '';
      if (spec.category !== 'bottle') {
        previewTag = spec.available === false
          ? '<span style="position: absolute; bottom: 4px; left: 4px; font-size: 0.55rem; background: rgba(100,116,139,0.85); color: #fff; padding: 1px 4px; border-radius: 2px;">MISSING</span>'
          : '<span style="position: absolute; bottom: 4px; left: 4px; font-size: 0.55rem; background: rgba(245,158,11,0.85); color: #000; font-weight: 700; padding: 1px 4px; border-radius: 2px;">PREVIEW ONLY</span>';
      }

      return `
        <div class="specimen-card-item ${isSelected ? 'active' : ''}" data-specimen-id="${spec.id}" style="position: relative; border: 1px solid; ${borderStyle} border-radius: var(--radius-sm); padding: 8px; display: flex; flex-direction: column; gap: 6px; cursor: pointer; transition: all 0.2s ease;">
          ${inspectedBadge}
          <div style="position: relative; width: 100%; aspect-ratio: 1; background: #08101E; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
            <img src="${spec.image_path}" alt="${spec.id}" style="max-width: 100%; max-height: 100%; object-fit: contain;" onerror="this.src='assets/icons/favicon.svg'">
            ${previewTag}
          </div>
          <div style="display: flex; flex-direction: column; gap: 2px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span class="font-mono text-cyan" style="font-size: 0.72rem; font-weight: 700;">${spec.id}</span>
              <span class="badge ${badgeClass}" style="font-size: 0.6rem; padding: 1px 4px;">${spec.label}</span>
            </div>
            <span style="font-size: 0.68rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${labelText}">${labelText}</span>
          </div>
          <button type="button" class="btn btn-secondary btn-sm" style="font-size: 0.7rem; padding: 4px 6px; justify-content: center; width: 100%; margin-top: auto;">
            ${isSelected ? '<i class="fas fa-check"></i> Selected' : 'Select'}
          </button>
        </div>
      `;
    }).join('');

    // Attach click listeners to cards
    container.querySelectorAll('.specimen-card-item').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-specimen-id');
        const spec = specimenManifest.find(s => s.id === id);
        if (spec) selectSpecimen(spec);
      });
    });
  }

  function selectSpecimen(spec) {
    selectedSpecimen = spec;
    currentFilename = spec.filename;
    isCurrentDefectHint = (spec.label === 'DEFECTIVE');
    capturedBlob = null;

    // Load actual image as payload via blob / URL
    currentImagePayload = spec.image_path;

    const imgEl = document.getElementById('preview-stage-img');
    const placeholder = document.getElementById('preview-placeholder-box');
    const loadingSpinner = document.getElementById('chamber-loading-spinner');
    const errorBox = document.getElementById('chamber-error-box');
    const errorTitle = document.getElementById('chamber-error-title');
    const errorMsg = document.getElementById('chamber-error-message');
    const errorUrl = document.getElementById('chamber-error-url');
    const metaFilename = document.getElementById('meta-filename');
    const metaSpecId = document.getElementById('meta-specimen-id');
    const metaCatRow = document.getElementById('meta-category-row');
    const metaGt = document.getElementById('meta-ground-truth');
    const metaSource = document.getElementById('meta-dataset-source');
    const metaScore = document.getElementById('meta-score-row');
    const metaCheckpoint = document.getElementById('meta-checkpoint-row');
    const btnRun = document.getElementById('btn-run-inspection');
    const badgeSpec = document.getElementById('meta-specimen-badge');

    // Reset view states
    if (placeholder) placeholder.style.display = 'none';
    if (errorBox) errorBox.style.display = 'none';

    if (metaFilename) metaFilename.textContent = spec.filename;
    if (metaSpecId) metaSpecId.textContent = spec.id;
    if (metaCatRow) metaCatRow.textContent = spec.category_display || spec.category;
    if (metaSource) metaSource.textContent = spec.dataset_source || 'Local Dataset';

    const defDisp = spec.defect_display || spec.display_type || (spec.label === 'NORMAL' ? 'Normal Specimen' : spec.defect_type);
    if (metaGt) {
      const badgeCls = (spec.label === 'NORMAL') ? 'badge-success' : 'badge-danger';
      metaGt.innerHTML = `<span class="badge ${badgeCls}">${spec.label} (${defDisp})</span>`;
    }

    // Handle unavailable dataset specimens (e.g. Wood)
    if (spec.available === false) {
      if (imgEl) {
        imgEl.style.display = 'none';
        imgEl.src = '';
      }
      if (loadingSpinner) loadingSpinner.style.display = 'none';
      if (errorBox) {
        errorBox.style.display = 'flex';
        if (errorTitle) errorTitle.textContent = 'DATASET NOT IN REPOSITORY';
        if (errorMsg) errorMsg.textContent = `Specimen ${spec.id} (${spec.category_display}) is cataloged from official ${spec.dataset_source}, but the image files are not packaged in this local environment. DefectX AI does not fabricate fake images.`;
        if (errorUrl) errorUrl.textContent = `Expected: ${spec.src_path || spec.filename}`;
      }
      if (metaCheckpoint) metaCheckpoint.innerHTML = '<span class="text-muted">DATASET UNAVAILABLE</span>';
      if (metaScore) metaScore.innerHTML = '<span class="text-muted">NO IMAGE TO SCAN</span>';
      if (badgeSpec) {
        badgeSpec.className = 'badge badge-secondary font-mono';
        badgeSpec.textContent = 'IMAGE UNAVAILABLE';
      }
      if (btnRun) {
        btnRun.innerHTML = '<i class="fas fa-ban"></i> Inspection Unavailable (Missing Image)';
        btnRun.className = 'btn btn-secondary';
        btnRun.disabled = true;
      }
    } else {
      // Normal or defect specimen with image file
      if (btnRun) btnRun.disabled = false;
      if (loadingSpinner) loadingSpinner.style.display = 'flex';
      if (imgEl) {
        imgEl.style.display = 'none';

        // Detach prior handlers
        imgEl.onload = null;
        imgEl.onerror = null;

        imgEl.onload = function () {
          if (loadingSpinner) loadingSpinner.style.display = 'none';
          if (errorBox) errorBox.style.display = 'none';
          imgEl.style.display = 'block';

          // Update specimen badge once image is confirmed loaded
          if (badgeSpec) {
            if (spec.category === 'bottle') {
              badgeSpec.className = 'badge badge-cyan font-mono';
              badgeSpec.textContent = 'LIVE SPECIMEN';
            } else {
              badgeSpec.className = 'badge badge-warning font-mono';
              badgeSpec.textContent = 'PREVIEW ONLY';
            }
          }
        };

        imgEl.onerror = function () {
          console.error(`[DefectX] Failed to load specimen image for ${spec.id}: ${spec.image_path}`);
          if (loadingSpinner) loadingSpinner.style.display = 'none';
          imgEl.style.display = 'none';
          if (errorBox) {
            errorBox.style.display = 'flex';
            if (errorTitle) errorTitle.textContent = 'SPECIMEN IMAGE LOAD FAILED';
            if (errorMsg) errorMsg.textContent = `Could not load image for ${spec.id}. Please check file permissions or web root routing.`;
            if (errorUrl) errorUrl.textContent = `URL: ${spec.image_path}`;
          }
          if (badgeSpec) {
            badgeSpec.className = 'badge badge-danger font-mono';
            badgeSpec.textContent = 'LOAD ERROR';
          }
          if (btnRun) {
            btnRun.innerHTML = '<i class="fas fa-triangle-exclamation"></i> Image Load Failed';
            btnRun.className = 'btn btn-secondary';
            btnRun.disabled = true;
          }
        };

        // Trigger load
        imgEl.src = spec.image_path;
      }

      if (spec.category === 'bottle') {
        if (metaCheckpoint) metaCheckpoint.textContent = 'checkpoints/defectx_bottle.pt (2.41 MB)';
        if (metaScore) metaScore.textContent = 'Awaiting Scan';
        if (btnRun) {
          btnRun.innerHTML = '<i class="fas fa-bolt"></i> Run AI Defect Inspection';
          btnRun.className = 'btn btn-primary';
        }
      } else {
        if (metaCheckpoint) metaCheckpoint.innerHTML = '<span class="text-amber">MODEL NOT AVAILABLE (PREVIEW ONLY)</span>';
        if (metaScore) metaScore.innerHTML = '<span class="text-muted">PREVIEW ONLY — NOT SCANNED</span>';
        if (btnRun) {
          btnRun.innerHTML = '<i class="fas fa-eye"></i> Preview Only (Model Not Active)';
          btnRun.className = 'btn btn-secondary';
        }
      }
    }

    resetResultsView();

    // Re-render grid selection state
    renderSpecimenGrid(getCurrentFilteredList());

    if (spec.category !== 'bottle') {
      renderPreviewOnlyNotice(spec);
    }

    if (window.showToast) {
      const toastType = (spec.category === 'bottle') ? 'info' : (spec.available === false ? 'warning' : 'warning');
      const toastSub = (spec.category === 'bottle')
        ? `${spec.id}: ${spec.label} (${defDisp}) ready for AI inspection.`
        : `${spec.id}: ${spec.category_display} specimen loaded (${spec.available === false ? 'DATASET UNAVAILABLE' : 'PREVIEW ONLY'}).`;
      window.showToast('Specimen Selected', toastSub, toastType, 3000);
    }
  }

  function initSpecimenFilters() {
    const filterBtns = document.querySelectorAll('#specimen-filter-tabs button');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.getAttribute('data-filter') || 'all';
        renderSpecimenGrid(getCurrentFilteredList());
      });
    });

    const searchInput = document.getElementById('specimen-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        currentSearchQuery = e.target.value.toLowerCase().trim();
        renderSpecimenGrid(getCurrentFilteredList());
      });
    }
  }

  function getCurrentFilteredList() {
    return specimenManifest.filter(spec => {
      // 0. Active Category Tab
      if (spec.category !== activeCategoryTab) return false;

      // 1. Label Filter
      let matchesFilter = true;
      if (activeFilter === 'normal') matchesFilter = (spec.label === 'NORMAL');
      else if (activeFilter === 'defective') matchesFilter = (spec.label === 'DEFECTIVE');

      if (!matchesFilter) return false;

      // 2. Search Query
      if (currentSearchQuery) {
        const idMatch = (spec.id || '').toLowerCase().includes(currentSearchQuery);
        const labelMatch = (spec.label || '').toLowerCase().includes(currentSearchQuery);
        const typeMatch = (spec.defect_display || spec.defect_type || '').toLowerCase().includes(currentSearchQuery);
        const fileMatch = (spec.filename || '').toLowerCase().includes(currentSearchQuery);
        return (idMatch || labelMatch || typeMatch || fileMatch);
      }

      return true;
    });
  }

  function updateLibrarySummaryStats() {
    const inspectedEl = document.getElementById('stat-inspected-count');
    const correctEl = document.getElementById('stat-correct-count');
    const incorrectEl = document.getElementById('stat-incorrect-count');

    let inspected = 0;
    let correct = 0;
    let incorrect = 0;

    inspectionHistoryLog.forEach(val => {
      inspected++;
      if (val.isMatch) correct++;
      else incorrect++;
    });

    if (inspectedEl) inspectedEl.textContent = `${inspected} / 14`;
    if (correctEl) correctEl.textContent = correct;
    if (incorrectEl) incorrectEl.textContent = incorrect;
  }

  function initDropzone() {
    const dropzone = document.getElementById('inspection-dropzone');
    const fileInput = document.getElementById('file-input-elem');
    const btnBrowse = document.getElementById('btn-browse-file');

    if (!dropzone || !fileInput) return;

    btnBrowse.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
      if (e.target !== btnBrowse && !btnBrowse.contains(e.target)) {
        fileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleUploadedFile(files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleUploadedFile(fileInput.files[0]);
      }
    });
  }

  function handleUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
      if (window.showToast) window.showToast('Invalid File', 'Please upload a valid optical image.', 'error');
      return;
    }

    currentImagePayload = file;
    currentFilename = file.name;
    isCurrentDefectHint = null;
    capturedBlob = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      const stageImg = document.getElementById('preview-stage-img');
      const placeholder = document.getElementById('preview-placeholder-box');
      const metaFilename = document.getElementById('meta-filename');

      if (stageImg) {
        stageImg.src = e.target.result;
        stageImg.style.display = 'block';
      }
      if (placeholder) placeholder.style.display = 'none';
      if (metaFilename) metaFilename.textContent = file.name;
      resetResultsView();

      if (window.showToast) {
        window.showToast('File Loaded', `${file.name} ready for inspection.`, 'info');
      }
    };
    reader.readAsDataURL(file);
  }

})();
