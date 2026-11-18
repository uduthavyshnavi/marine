/* AquaFlex AI – Reactive Dashboard Controller v2
   ── Phase 2: Full 4-quadrant reactive state binding
   ── CartoDB Voyager tiles, real-time layer cleanup, centroid flyTo,
      pulsing markers, base64 mask overlays, drift trajectory, triage ring
   ──────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  /* ── API base ─────────────────────────────────────────────────────── */
  var API_BASE =
    window.AQUAFLEX_API_BASE ||
    (function () {
      if (window.location.protocol === 'file:') return 'http://127.0.0.1:8000';
      var host = window.location.hostname || '127.0.0.1';
      var port = window.location.port;
      if (port === '8000' || !port) return '';
      return 'http://' + host + ':8000';
    })();

  /* ── Constants ─────────────────────────────────────────────────────── */
  var SINKING_SITE = { lat: 8.2177, lon: 76.752 };
  var MAP_CENTER   = { lat: 8.6, lon: 77.6 };
  var MAP_ZOOM     = 7;
  var CIRCUMFERENCE = 263.89; // SVG circle r=42 → 2πr

  /* ── State ─────────────────────────────────────────────────────────── */
  var map;
  var detectionLayerGroup;
  var centroidMarker = null;
  var maskOverlay    = null;
  var countdownInterval = null;
  var isProcessing   = false;

  // ── Forecast Slider State ───────────────────────────────────────────
  var currentForecastTrajectory = [];
  var currentForecastOrigin = null;
  var forecastMarker = null;
  var forecastAnimTimer = null;
  var isPlayingForecast = false;
  var currentEtaHours = 72;

  /* ── Init ──────────────────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {
    initMap();
    wireUploadControls();
    wireForecastSlider();
    wireAlertAndReportControls();
    wireCoordinateInspector();
    loadSimulation();
  });

  /* ────────────────────────── MAP SETUP ───────────────────────────── */
  function initMap() {
    map = L.map('map', {
      center: [MAP_CENTER.lat, MAP_CENTER.lon],
      zoom: MAP_ZOOM,
      zoomControl: true,
      worldCopyJump: true,
    });

    // 1. OpenStreetMap Standard (Default - 100% Free, No API Key Required)
    const osmBasemap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    });

    // 2. Esri World Imagery (Satellite)
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 18,
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });

    // 3. Esri Ocean Basemap (Marine Bathymetry)
    const esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 13,
      attribution: 'Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVOCEANO'
    });

    // 4. National Geographic World Map (Optional Terrain)
    const natGeoLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 16,
      attribution: 'Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC'
    });

    // 5. Esri World Topo Map (Detailed Place Names & Topo)
    const esriTopo = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 18,
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong)'
    });

    // Add default basemap to Leaflet map instance
    osmBasemap.addTo(map);

    // Add base layer selector control
    const baseLayers = {
      "OpenStreetMap (Default)": osmBasemap,
      "Esri Satellite": esriSatellite,
      "Esri Ocean Bathymetry": esriOcean,
      "National Geographic Terrain": natGeoLayer,
      "Esri Detailed Topo & Places": esriTopo
    };

    L.control.layers(baseLayers, null, { position: 'topright' }).addTo(map);

    // Layer group for all live-detection overlays (cleared on each new detection)
    detectionLayerGroup = L.layerGroup().addTo(map);

    // Map click → auto-populate Q8 Coordinate Inspector
    map.on('click', function (e) {
      var latInput = document.getElementById('coord-lat-input');
      var lonInput = document.getElementById('coord-lon-input');
      if (latInput) latInput.value = e.latlng.lat.toFixed(5);
      if (lonInput) lonInput.value = e.latlng.lng.toFixed(5);
      // Scroll to Q8 panel in sidebar
      var q8 = document.getElementById('q8-panel');
      if (q8) q8.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Show a brief toast
      showToast('📍 Coordinates captured: ' + e.latlng.lat.toFixed(4) + '°N, ' + e.latlng.lng.toFixed(4) + '°E — Click "Predict 72h Drift" to run.', 'info');
    });

    // Force Leaflet recalculation of container size after DOM renders
    setTimeout(function () {
      if (map) map.invalidateSize();
    }, 100);
    setTimeout(function () {
      if (map) map.invalidateSize();
    }, 400);
    setTimeout(function () {
      if (map) map.invalidateSize();
    }, 1000);
    window.addEventListener('resize', function () {
      if (map) map.invalidateSize();
    });

    // MSC ELSA 3 sinking site marker (permanent)
    addSinkingSiteMarker();
  }

  /* ────────────────── GEOMETRY & HUD HELPERS ─────────────────────── */
  function computeHexagonVertices(lat, lon, radiusKm) {
    var vertices = [];
    var r = radiusKm || 2.5;
    for (var i = 0; i < 6; i++) {
      var angle = (i * 60.0) * Math.PI / 180.0;
      var dlat = (r * Math.cos(angle)) / 111.32;
      var dlon = (r * Math.sin(angle)) / (111.32 * Math.cos(lat * Math.PI / 180.0));
      vertices.push([lat + dlat, lon + dlon]);
    }
    return vertices;
  }

  function computeBearing(lat1, lon1, lat2, lon2) {
    var y = Math.sin((lon2 - lon1) * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0);
    var x = Math.cos(lat1 * Math.PI / 180.0) * Math.sin(lat2 * Math.PI / 180.0) -
            Math.sin(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) * Math.cos((lon2 - lon1) * Math.PI / 180.0);
    var b = Math.atan2(y, x) * 180.0 / Math.PI;
    return (b + 360.0) % 360.0;
  }

  function getCardinal(deg) {
    var dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    var idx = Math.floor((deg + 11.25) / 22.5) % 16;
    return dirs[idx];
  }

  function updateTimerHUD(hour) {
    var digitsEl = document.getElementById('timer-hud-digits');
    var hourEl = document.getElementById('timer-hud-hour');
    var statusEl = document.getElementById('timer-hud-status');
    if (!digitsEl) return;

    var h = Math.floor(hour);
    var m = Math.floor((hour - h) * 60);
    var s = Math.floor((((hour - h) * 60) - m) * 60);
    digitsEl.textContent = pad(h) + ':' + pad(m) + ':' + pad(s);

    if (hourEl) hourEl.textContent = 'T+' + pad(h) + 'h';
    if (statusEl) {
      if (hour === 0) statusEl.textContent = 'ORIGIN RELEASE';
      else if (hour < 24) statusEl.textContent = 'OFFSHORE DRIFT';
      else if (hour < 48) statusEl.textContent = 'COASTAL APPROACH';
      else if (hour < 72) statusEl.textContent = 'IMMINENT LANDFALL';
      else statusEl.textContent = 'SHORELINE INTERCEPT';
    }
  }

  function addSinkingSiteMarker() {
    var site = L.circleMarker([SINKING_SITE.lat, SINKING_SITE.lon], {
      radius: 8,
      color: '#22d3ee',
      fillColor: '#22d3ee',
      fillOpacity: 0.85,
      weight: 2,
    }).addTo(map);
    site.bindPopup(
      '<strong style="color:#22d3ee">MSC ELSA 3 Sinking Site</strong><br>' +
      '~13 NM off Kerala<br>Sank 25 May 2025'
    );
  }

  /* ────────────────────────── UPLOAD CONTROLS ─────────────────────── */
  function wireUploadControls() {
    var fileInput = document.getElementById('unet-file');
    var runBtn    = document.getElementById('unet-run');
    var uploadZone = document.getElementById('upload-zone');

    // Drag-and-drop
    uploadZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      uploadZone.style.borderColor = '#22d3ee';
    });
    uploadZone.addEventListener('dragleave', function () {
      uploadZone.style.borderColor = '';
    });
    uploadZone.addEventListener('drop', function (e) {
      e.preventDefault();
      uploadZone.style.borderColor = '';
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) {
        fileInput.files = files;
        updateUploadLabel(files[0].name);
      }
    });

    fileInput.addEventListener('change', function () {
      if (fileInput.files[0]) updateUploadLabel(fileInput.files[0].name);
    });

    runBtn.addEventListener('click', runLiveDetection);
  }

  function updateUploadLabel(name) {
    var zone = document.getElementById('upload-zone');
    var span = zone.querySelector('span');
    if (span) span.textContent = name;
  }

  /* ────────────────────────── LIVE DETECTION ──────────────────────── */
  function runLiveDetection() {
    var fileInput = document.getElementById('unet-file');
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
      setStatus('Select an 11-band Sentinel-2 GeoTIFF (.tif) first.', 'error');
      return;
    }
    if (isProcessing) return;
    isProcessing = true;

    // ── Phase 2: Immediately clear stale state and show loading ───────
    clearDetectionLayers();
    clearAllQuadrantMetrics();
    showProcessingOverlay(true);
    setStatus('Running quality gate + U-Net sliding-window inference…', '');
    document.getElementById('unet-run').classList.add('busy');

    var form = new FormData();
    form.append('file', fileInput.files[0]);

    fetch(API_BASE + '/api/v1/detect/unet', { method: 'POST', body: form })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        });
      })
      .then(function (r) {
        if (!r.ok) {
          var detail = r.data.detail || ('HTTP ' + r.status);
          setStatus('Rejection: ' + detail, 'error');
          updateQ1QualityGate(null, false, detail);
          return;
        }
        renderLiveResponse(r.data);
        setStatus('Done — live segmentation complete.', 'ok');
      })
      .catch(function (err) {
        setStatus('Request failed: ' + err.message, 'error');
      })
      .finally(function () {
        isProcessing = false;
        showProcessingOverlay(false);
        document.getElementById('unet-run').classList.remove('busy');
      });
  }

  /* ── Master render: dispatches to all quadrants ───────────────────── */
  function renderLiveResponse(d) {
    renderQ1(d);
    renderQ2(d);
    renderQ3(d);
    renderQ4(d);
    renderQ5Spectral(d);
    renderQ6Logistics(d);
    renderVoiceAdvisories(d);
    renderMapLayers(d);
    updateSidebarStatistics(d);
  }

  /* ────────────────── Q1: Satellite Mask & Quality Gate ───────────── */
  function renderQ1(d) {
    var q1 = document.getElementById('q1-results');
    if (q1) q1.classList.remove('hidden');

    // Mask preview
    var maskImg = document.getElementById('unet-mask');
    if (maskImg && d.outputs && d.outputs.base64_preview_png) {
      maskImg.src = d.outputs.base64_preview_png;
    }

    var qg = d.quality_gate || {};
    var passed = qg.passed !== false;
    updateQ1QualityGate(qg, passed, null);

    // Saturation meters
    var glint = (qg.glint_fraction || 0) * 100;
    var satPx = qg.saturated_pixels || 0;
    var totPx = qg.total_pixels || 1;
    var satPct = (satPx / totPx) * 100;
    setMeter('meter-glint', 'meter-glint-pct', glint, 20);
    setMeter('meter-sat',   'meter-sat-pct',   satPct, 20);

    // Badge
    setBadge('q1-gate-badge', passed ? 'PASS' : 'FAIL',
             passed ? 'color:#39ff14;border-color:rgba(57,255,20,.4);background:rgba(57,255,20,.08)'
                    : 'color:#ff2d55;border-color:rgba(255,45,85,.4);background:rgba(255,45,85,.08)');
  }

  function updateQ1QualityGate(qg, passed, rejMsg) {
    var qEl = document.getElementById('unet-quality');
    if (!qEl) return;
    if (rejMsg) {
      qEl.textContent = 'REJECTED — ' + rejMsg;
      qEl.className = 'metric-val gate-val fail';
    } else if (qg) {
      var glint = ((qg.glint_fraction || 0) * 100).toFixed(2);
      var thr   = ((qg.threshold || 0.2) * 100).toFixed(0);
      qEl.textContent = (passed ? 'PASS' : 'REJECT') + ' · glint ' + glint + '% (≤' + thr + '%)';
      qEl.className   = 'metric-val gate-val ' + (passed ? 'pass' : 'fail');
    }
  }

  function setMeter(barId, pctId, value, threshold) {
    var bar  = document.getElementById(barId);
    var lbl  = document.getElementById(pctId);
    if (!bar || !lbl) return;
    var pct  = Math.min(value, 100);
    bar.style.width = pct + '%';
    if (value > threshold) bar.style.background = 'var(--danger)';
    lbl.textContent = pct.toFixed(1) + '%';
  }

  /* ──────────────── Q2: Detection & Spatial Metrics ──────────────── */
  function renderQ2(d) {
    var m   = d.live_detection_metrics || {};
    var sdm = d.spatial_detection_metrics || {};
    var dri = d.detection_resolution_info || {};
    var sc  = d.spatial_coordinates || {};

    setEl('q2-area-m2',  sdm.debris_area_sq_m   != null ? fmtNum(sdm.debris_area_sq_m) : '—');
    setEl('q2-area-km2', sdm.debris_area_sq_km   != null ? fmtNum(sdm.debris_area_sq_km, 6) : '—');
    setEl('q2-pixels',   m.debris_pixels          != null ? fmtNum(m.debris_pixels) + ' px' : '—');
    setEl('q2-coverage', m.debris_coverage_percentage != null
          ? m.debris_coverage_percentage.toFixed(3) + '%' : '—');
    setEl('q2-res-x',   dri.res_x_m != null ? dri.res_x_m + ' m' : '—');
    setEl('q2-res-y',   dri.res_y_m != null ? dri.res_y_m + ' m' : '—');
    setEl('q2-px-area', dri.pixel_area_m2 != null ? dri.pixel_area_m2 + ' m²' : '—');

    var clat = sc.centroid_latitude, clon = sc.centroid_longitude;
    setEl('q2-centroid', clat != null
          ? clat.toFixed(4) + '°, ' + clon.toFixed(4) + '°' : 'No debris');

    var scale = sdm.cluster_scale || m.cluster_scale || 'UNKNOWN';
    setEl('q2-cluster', scale);
    setBadge('q2-scale-badge', clusterAbbr(scale));
  }

  /* ─────────────── Q3: 72-Hour Drift & Source Attribution ─────────── */
  function renderQ3(d) {
    // ── Source attribution ────────────────────────────────────────────
    var sa = d.source_attribution || {};
    var srcType = sa.origin_type || sa.source_type || '—';
    var ori = sa.primary_origin || sa.origin_coordinates || {};

    setEl('q3-source-type', srcType);
    setEl('q3-source-desc', sa.description || 'Attribution data unavailable.');

    // Attribution confidence score
    if (typeof sa.confidence_score === 'number') {
      setEl('q3-source-conf', (sa.confidence_score * 100).toFixed(1) + '% Conf');
    } else {
      setEl('q3-source-conf', '—');
    }

    // Origin coordinates
    if (ori.latitude != null && ori.longitude != null) {
      setEl('q3-origin-coords', ori.latitude.toFixed(4) + '°N, ' + ori.longitude.toFixed(4) + '°E');
    } else {
      setEl('q3-origin-coords', '—');
    }

    // Nearest entity (River outlet or Shipping corridor)
    var nearest = (sa.nearest_source && sa.nearest_source.name) || sa.nearest_river_mouth || '—';
    var dist = (sa.nearest_source && sa.nearest_source.distance_km != null)
      ? ' (' + sa.nearest_source.distance_km.toFixed(1) + ' km)'
      : (sa.river_mouth_dist_km != null ? ' (' + sa.river_mouth_dist_km.toFixed(1) + ' km)' : '');
    setEl('q3-nearest-source', nearest + dist);

    // Estimated discharge timestamp
    if (sa.discharge_timestamp_estimate) {
      try {
        var dEst = new Date(sa.discharge_timestamp_estimate);
        setEl('q3-est-release', dEst.toUTCString().replace('GMT', 'UTC'));
      } catch (e) {
        setEl('q3-est-release', sa.discharge_timestamp_estimate);
      }
    } else {
      setEl('q3-est-release', '—');
    }

    // Badge styling
    var badgeText = '—';
    var badgeStyle = '';
    if (srcType === 'URBAN_RIVERINE_RUNOFF' || srcType === 'RIVER_RUNOFF') {
      badgeText = 'RIVERINE';
      badgeStyle = 'color:#38bdf8;border-color:rgba(56,189,248,.4);background:rgba(56,189,248,.1)';
    } else if (srcType === 'VESSEL_DISCHARGE') {
      badgeText = 'VESSEL';
      badgeStyle = 'color:#c084fc;border-color:rgba(192,132,252,.4);background:rgba(192,132,252,.1)';
    } else if (srcType === 'COASTAL_OUTFLOW') {
      badgeText = 'COASTAL';
      badgeStyle = 'color:#ffb020;border-color:rgba(255,176,32,.4);background:rgba(255,176,32,.1)';
    } else if (srcType === 'OFFSHORE_SPILL') {
      badgeText = 'OFFSHORE';
      badgeStyle = 'color:#ff2d55;border-color:rgba(255,45,85,.4);background:rgba(255,45,85,.1)';
    }
    setBadge('q3-source-badge', badgeText, badgeStyle);

    // ── Drift trajectory nodes ────────────────────────────────────────
    var df = d.drift_forecast || {};
    var positions = df.positions || [];
    var trajList  = document.getElementById('q3-traj-list');
    if (trajList) {
      trajList.innerHTML = '';
      if (!positions.length) {
        trajList.innerHTML = '<li class="traj-placeholder">No drift data available.</li>';
      } else {
        positions.forEach(function (pos) {
          var li = document.createElement('li');
          li.className = 'traj-item';
          li.innerHTML =
            '<span class="traj-hour">+' + pos.hour + 'h</span>' +
            '<span class="traj-coords">' + pos.latitude.toFixed(3) + '°, ' + pos.longitude.toFixed(3) + '°</span>' +
            '<span class="traj-speed">' + (pos.drift_speed_ms || 0).toFixed(2) + ' m/s</span>';
          trajList.appendChild(li);
        });
      }
    }
  }

  /* ─────────────── Q4: Risk Triage & Municipal Alerts ────────────── */
  function renderQ4(d) {
    var ts  = d.triage_score || {};
    var li  = d.live_intimation || {};
    var mrs = d.municipal_report_status || {};
    var cci = d.coastal_community_impact || {};

    // Score ring
    var score = ts.priority_score || 0;
    setEl('score-num', (score * 100).toFixed(0) + '%');
    var arc = document.getElementById('score-arc');
    if (arc) {
      var dashOffset = CIRCUMFERENCE * (1 - score);
      arc.style.strokeDashoffset = dashOffset;
      arc.style.stroke = score >= 0.65 ? 'var(--danger)'
                       : score >= 0.35 ? 'var(--warn)'
                       : 'var(--neon)';
    }

    // Threat level
    var threat = ts.threat_level || cci.impact_threat_level || '—';
    setEl('q4-threat-level', threat);
    setEl('chip-threat-val', threat);
    setBadge('q4-threat-badge', threat,
      threat === 'CRITICAL' ? 'color:var(--danger);border-color:rgba(255,45,85,.4);background:rgba(255,45,85,.08)'
      : threat === 'MODERATE' ? 'color:var(--warn);border-color:rgba(255,176,32,.4);background:rgba(255,176,32,.08)'
      : 'color:var(--neon);border-color:rgba(57,255,20,.4);background:rgba(57,255,20,.08)');

    // Score breakdown
    var comp = ts.components || {};
    setEl('q4-landfall', ts.landfall_probability != null
          ? (ts.landfall_probability * 100).toFixed(1) + '%' : '—');
    setEl('q4-habitat',  comp.habitat_zone_name || '—');
    setEl('q4-community', comp.community_zone_name || '—');

    // Dispatch
    setEl('q4-alert-status',  li.alert_status || '—');
    setEl('q4-channels', (li.dispatch_channels || []).join(', ') || 'None configured');
    setEl('q4-municipal',
          mrs.municipal_webhook_triggered
          ? (mrs.nearby_river_outlets || []).map(function(o){ return o.name; }).join(', ') || 'Triggered'
          : 'Not triggered (> 5 km from outlets)');
    setEl('q4-incident-id', mrs.incident_id || '—');

    // PDF Report Download Link Binding
    var pdfBtn = document.getElementById('download-pdf-btn');
    if (pdfBtn) {
      var incId = mrs.incident_id || 'latest';
      pdfBtn.href = API_BASE + '/api/v1/reports/pdf/' + encodeURIComponent(incId);
    }

    // Alert message
    setEl('q4-alert-msg', li.dynamic_message_text || '—');

    // Top header chips
    setEl('chip-threat-val', threat);
    var alertBadge = document.getElementById('alert-badge');
    if (alertBadge) {
      alertBadge.textContent = li.alert_status || 'STANDBY';
      alertBadge.className   = 'topbar-badge' + (li.alert_status === 'TRIGGERED' ? ' active' : '');
    }
    setEl('sys-status', 'LIVE');

    // GeoTIFF download
    var dlBtn = document.getElementById('unet-geotiff');
    if (dlBtn && d.outputs && d.outputs.geotiff_download_url) {
      dlBtn.href = API_BASE + d.outputs.geotiff_download_url;
      dlBtn.classList.remove('hidden');
    }
  }

  /* ─────────────── Q5: Spectral Material Classifier ──────────────── */
  function renderQ5Spectral(d) {
    var mc = d.material_classification || {};
    var dominant = mc.dominant_material || 'Macro Plastics & Debris Bottles';
    setEl('mat-dominant-badge', dominant);
    setBadge('mat-dominant-badge', dominant,
      dominant.indexOf('Ghost Net') !== -1
        ? 'color:#f43f5e;border-color:rgba(244,63,94,.4);background:rgba(244,63,94,.1)'
        : dominant.indexOf('Biomass') !== -1
        ? 'color:#10b981;border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)'
        : 'color:#06b6d4;border-color:rgba(6,182,212,.4);background:rgba(6,182,212,.1)');

    var fdi = typeof mc.floating_debris_index_fdi === 'number' ? mc.floating_debris_index_fdi : 0.0825;
    var psi = typeof mc.plastic_surface_index_psi === 'number' ? mc.plastic_surface_index_psi : 0.0412;
    setEl('mat-fdi-value', fdi.toFixed(4));
    setEl('mat-psi-value', psi.toFixed(4));

    var cb = mc.composition_breakdown || {};
    var nets = typeof cb.ghost_nets_and_gear === 'number' ? cb.ghost_nets_and_gear : 25.0;
    var plastics = typeof cb.macro_plastics_and_bottles === 'number' ? cb.macro_plastics_and_bottles : 60.0;
    var biomass = typeof cb.marine_biomass_and_seaweed === 'number' ? cb.marine_biomass_and_seaweed : 15.0;

    var elNetsBar = document.getElementById('mat-ghost-nets-bar');
    var elNetsPct = document.getElementById('mat-ghost-nets-pct');
    if (elNetsBar) elNetsBar.style.width = Math.min(100, Math.max(0, nets)) + '%';
    if (elNetsPct) elNetsPct.textContent = nets.toFixed(1) + '%';

    var elPlasticsBar = document.getElementById('mat-macro-plastics-bar');
    var elPlasticsPct = document.getElementById('mat-macro-plastics-pct');
    if (elPlasticsBar) elPlasticsBar.style.width = Math.min(100, Math.max(0, plastics)) + '%';
    if (elPlasticsPct) elPlasticsPct.textContent = plastics.toFixed(1) + '%';

    var elBiomassBar = document.getElementById('mat-biomass-bar');
    var elBiomassPct = document.getElementById('mat-biomass-pct');
    if (elBiomassBar) elBiomassBar.style.width = Math.min(100, Math.max(0, biomass)) + '%';
    if (elBiomassPct) elBiomassPct.textContent = biomass.toFixed(1) + '%';

    var hazard = mc.hazard_classification || 'Awaiting spectral assessment';
    setEl('mat-hazard-warning', hazard);
  }

  /* ─────────────── Q6: Fleet Intercept & Logistics ────────────────── */
  function renderQ6Logistics(d) {
    var lp = d.logistics_plan || {};
    var port = lp.nearest_base_port || {};
    var dist = lp.geodesic_distance || {};
    var sol = lp.intercept_solution || {};
    var fleet = lp.fleet_recommendation || {};

    var portName = port.name ? (port.name + (port.state ? ' (' + port.state + ')' : '')) : 'Kochi Base Port';
    setEl('logistics-port-name', portName);
    setBadge('logistics-port-badge', port.agency ? 'CG DISPATCH' : 'BASE PORT');

    var nm = typeof dist.nautical_miles === 'number' ? dist.nautical_miles.toFixed(1) + ' NM' : '—';
    var km = typeof dist.kilometers === 'number' ? ' (' + dist.kilometers.toFixed(1) + ' km)' : '';
    setEl('logistics-dist-nm', nm + km);

    var targetHour = sol.target_hour != null ? sol.target_hour : 12;
    var iLat = sol.intercept_latitude != null ? sol.intercept_latitude.toFixed(3) + '°N' : '—';
    var iLon = sol.intercept_longitude != null ? sol.intercept_longitude.toFixed(3) + '°E' : '—';
    setEl('logistics-intercept-point', 'T+' + targetHour + 'h [' + iLat + ', ' + iLon + ']');

    var etaHrs = typeof sol.vessel_transit_hours === 'number' ? sol.vessel_transit_hours.toFixed(1) + ' hrs' : '—';
    var spd = typeof sol.cruise_speed_knots === 'number' ? ' (@ ' + sol.cruise_speed_knots.toFixed(0) + ' kts)' : '';
    setEl('logistics-eta-hours', etaHrs + spd);

    var skimmers = fleet.skimmer_vessels_count != null ? fleet.skimmer_vessels_count + ' ASV Skimmer(s)' : '1 ASV Skimmer';
    setEl('logistics-skimmer-fleet', skimmers);

    var fuelL = typeof fleet.fuel_consumption_liters === 'number' ? fleet.fuel_consumption_liters.toFixed(0) + ' L' : '—';
    var costINR = typeof fleet.estimated_fuel_cost_inr === 'number' ? ' (₹' + fleet.estimated_fuel_cost_inr.toLocaleString() + ')' : '';
    setEl('logistics-fuel-liters', fuelL + costINR);
  }

  /* ─────────────── Q7: Multilingual Voice Advisories ──────────────── */
  function renderVoiceAdvisories(d) {
    var va = d.voice_advisories || {};
    var teAudio = document.getElementById('teluguAudioPlayer');
    var enAudio = document.getElementById('englishAudioPlayer');
    var statusBadge = document.getElementById('voice-advisory-status');

    if (va.telugu_audio_url && teAudio) {
      teAudio.src = va.telugu_audio_url;
      teAudio.load();
    }
    if (va.english_audio_url && enAudio) {
      enAudio.src = va.english_audio_url;
      enAudio.load();
    }

    if (statusBadge) {
      if (va.status === 'success' || va.telugu_audio_url) {
        statusBadge.textContent = 'READY (2 STREAMS)';
        statusBadge.setAttribute('style', 'color:#10b981;border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)');
      } else {
        statusBadge.textContent = 'STANDBY';
      }
    }
  }

  /* ──────────────── DYNAMIC SIDEBAR STATISTICS BINDING ───────────── */
  function updateSidebarStatistics(data) {
    if (!data) return;
    var triage = data.triage || {};
    var geospatial = data.geospatial || {};
    var detection = data.detection || {};

    // 1. threatLevelText (e.g. "Critical Threat", "Moderate Threat", or "Low Threat")
    var threatLevel = triage.threat_level;
    if (!threatLevel && data.triage_score) {
      threatLevel = data.triage_score.threat_level;
    }
    if (!threatLevel) threatLevel = 'Moderate Threat';
    var elThreat = document.getElementById('threatLevelText');
    if (elThreat) elThreat.innerText = threatLevel;

    // 2. priorityScoreText (e.g. 0.82 or 0.18)
    var priorityScore = (typeof triage.priority_score === 'number')
      ? triage.priority_score
      : ((data.triage_score && typeof data.triage_score.priority_score === 'number')
        ? data.triage_score.priority_score : 0.0);
    var elScore = document.getElementById('priorityScoreText');
    if (elScore) elScore.innerText = priorityScore.toFixed(2);

    // 3. hotspotRadiusText (${data.geospatial.calculated_radius_km.toFixed(1)} km)
    var radiusKm = (typeof geospatial.calculated_radius_km === 'number')
      ? geospatial.calculated_radius_km
      : ((data.hotspots && data.hotspots[0] && data.hotspots[0].radius_km) ? data.hotspots[0].radius_km : 2.0);
    var elRadius = document.getElementById('hotspotRadiusText');
    if (elRadius) elRadius.innerText = radiusKm.toFixed(1) + ' km';

    // 4. intensityText (${(data.detection.confidence_intensity * 100).toFixed(1)}%)
    var intensity = (typeof detection.confidence_intensity === 'number')
      ? detection.confidence_intensity
      : ((typeof data.overall_mean_confidence === 'number') ? (data.overall_mean_confidence / 100.0) : 0.0);
    var elIntensity = document.getElementById('intensityText');
    if (elIntensity) elIntensity.innerText = (intensity * 100).toFixed(1) + '%';

    // 5. debrisAreaText (${data.geospatial.debris_area_km2.toFixed(3)} km²)
    var areaKm2 = (typeof geospatial.debris_area_km2 === 'number')
      ? geospatial.debris_area_km2
      : ((data.spatial_detection_metrics && typeof data.spatial_detection_metrics.debris_area_sq_km === 'number')
        ? data.spatial_detection_metrics.debris_area_sq_km : 0.0);
    var elArea = document.getElementById('debrisAreaText');
    if (elArea) elArea.innerText = areaKm2.toFixed(3) + ' km²';
  }

  /* ──────────────── DYNAMIC HOTSPOT COLOR ENGINE ──────────────────── */
  function getHotspotVisual(priorityScore) {
    var score = typeof priorityScore === 'number' ? priorityScore : 0.0;
    var color = '#007AFF'; // 🔵 Safe / Minor Dispersal (< 0.30)
    var label = 'Safe / Minor Dispersal';
    var level = 'SAFE';

    if (score >= 0.70) {
      color = '#FF2D55'; // 🔴 Critical Threat / Major Spill (>= 0.70)
      label = 'Critical Threat / Major Spill';
      level = 'CRITICAL';
    } else if (score >= 0.30) {
      color = '#FFCC00'; // 🟡 Moderate Threat / Medium Field (0.30 <= score < 0.70)
      label = 'Moderate Threat / Medium Field';
      level = 'MODERATE';
    } else {
      color = '#007AFF'; // 🔵 Safe / Minor Dispersal (< 0.30)
      label = 'Safe / Minor Dispersal';
      level = 'SAFE';
    }

    return { color: color, label: label, level: level, score: score };
  }

  /* ────────────────────────── MAP LAYERS ──────────────────────────── */
  function renderMapLayers(d) {
    // ── Clean up old layers before rendering new ones ─────────────────
    clearDetectionLayers();

    var sc = d.spatial_coordinates || {};
    var clat = (d.geospatial && d.geospatial.centroid_lat != null) ? d.geospatial.centroid_lat : sc.centroid_latitude;
    var clon = (d.geospatial && d.geospatial.centroid_lon != null) ? d.geospatial.centroid_lon : sc.centroid_longitude;
    var bbox = (d.geospatial && d.geospatial.bounding_box_wgs84) ? d.geospatial.bounding_box_wgs84 : sc.bounding_box_wgs84;
    var ts = d.triage_score || {};
    var priorityScore = (d.triage && typeof d.triage.priority_score === 'number')
      ? d.triage.priority_score
      : ((typeof ts.priority_score === 'number') ? ts.priority_score : 0.0);
    var radiusKm = (d.geospatial && typeof d.geospatial.calculated_radius_km === 'number')
      ? d.geospatial.calculated_radius_km
      : 2.0;
    var vis = getHotspotVisual(priorityScore);
    var areaKm2 = (d.geospatial && typeof d.geospatial.debris_area_km2 === 'number')
      ? d.geospatial.debris_area_km2
      : 0.0;

    // ── Smooth flyTo detected centroid at zoom 12 ─────────────────────
    if (clat != null && clon != null) {
      map.flyTo([clat, clon], 12, { duration: 1.6, easeLinearity: 0.5 });

      // ── Centroid circle marker with dynamic risk color tier ─────────
      centroidMarker = L.circleMarker([clat, clon], {
        radius: Math.max(8, Math.min(24, Math.round(radiusKm * 1.5))),
        color: vis.color,
        fillColor: vis.color,
        fillOpacity: 0.85,
        weight: 3,
        className: 'centroid-pulse-glow',
      }).addTo(detectionLayerGroup);

      centroidMarker.bindPopup(
        '<strong style="color:' + vis.color + '">Detected Debris Cluster</strong><br>' +
        'Threat: ' + vis.label + '<br>' +
        'Priority Score: ' + priorityScore.toFixed(2) + '<br>' +
        'Calculated Radius: ' + radiusKm.toFixed(1) + ' km<br>' +
        'Area: ' + areaKm2.toFixed(3) + ' km²<br>' +
        'Lat: ' + clat.toFixed(5) + ', Lon: ' + clon.toFixed(5)
      );

      // ── Regular 6-vertex hexagon polygon centered at (clat, clon) ──
      // radius scaled dynamically to calculated_radius_km and styled with matching dynamic color tier
      var hexPoints = computeHexagonVertices(clat, clon, radiusKm);
      L.polygon(hexPoints, {
        color: vis.color,
        fillColor: vis.color,
        fillOpacity: 0.32,
        weight: 3,
        className: 'glowing-hex-debris',
      })
        .bindPopup(
          '<strong style="color:' + vis.color + '">⬡ Debris Field (' + vis.label + ')</strong><br>' +
          'Threat Level: ' + vis.label + '<br>' +
          'Priority Score: ' + priorityScore.toFixed(2) + '<br>' +
          'Footprint Radius: ~' + radiusKm.toFixed(1) + ' km<br>' +
          'Centroid: ' + clat.toFixed(4) + '°N, ' + clon.toFixed(4) + '°E'
        )
        .addTo(detectionLayerGroup);
    }

    // ── Base64 mask image overlay on bounding box ──────────────────
    if (bbox && bbox.length === 4 && d.outputs && d.outputs.base64_preview_png) {
      var sw = L.latLng(bbox[1], bbox[0]);
      var ne = L.latLng(bbox[3], bbox[2]);
      var bounds = L.latLngBounds(sw, ne);
      maskOverlay = L.imageOverlay(
        d.outputs.base64_preview_png,
        bounds,
        { opacity: 0.55, interactive: false }
      ).addTo(detectionLayerGroup);
    }

    // ── Green Dashed Trajectory Polyline & Heading Vectors ──────────
    var df = d.drift_forecast || {};
    var positions = df.positions || [];
    if (positions.length && clat != null) {
      var latlngs = [[clat, clon]];
      positions.forEach(function (p) { latlngs.push([p.latitude, p.longitude]); });

      // Green dashed trajectory line (#10B981)
      L.polyline(latlngs, {
        color: '#10B981',
        weight: 3.5,
        opacity: 0.95,
        dashArray: '8, 8',
        className: 'trajectory-green-dash',
      }).addTo(detectionLayerGroup);

      // Rotated heading arrows on trajectory nodes
      positions.forEach(function (p, i) {
        var prevLat = (i === 0) ? clat : positions[i - 1].latitude;
        var prevLon = (i === 0) ? clon : positions[i - 1].longitude;
        var headingDeg = p.heading_deg != null ? p.heading_deg : computeBearing(prevLat, prevLon, p.latitude, p.longitude);
        var cardinal = p.cardinal_heading || getCardinal(headingDeg);

        var arrowIcon = L.divIcon({
          className: 'heading-arrow-marker-wrap',
          html: '<div class="heading-arrow-marker" style="transform: rotate(' + headingDeg + 'deg);">' +
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none">' +
                '<path d="M12 2L19 21L12 17L5 21L12 2Z" fill="#10B981" stroke="#040912" stroke-width="2"/>' +
                '</svg></div>',
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        });

        L.marker([p.latitude, p.longitude], { icon: arrowIcon })
          .bindPopup(
            '<strong style="color:#10B981">T+' + p.hour + 'h Waypoint</strong><br>' +
            'Heading: <strong>' + headingDeg.toFixed(1) + '° (' + cardinal + ')</strong><br>' +
            'Drift Speed: ' + (p.drift_speed_ms || 0).toFixed(2) + ' m/s<br>' +
            'Wind Speed: ' + (p.wind_speed_ms || '—') + ' m/s<br>' +
            'Coordinates: ' + p.latitude.toFixed(4) + '°, ' + p.longitude.toFixed(4) + '°'
          )
          .addTo(detectionLayerGroup);
      });

      // Floating dynamic distance badge at trajectory midpoint
      var lastPos = positions[positions.length - 1];
      var totalDistKm = haversineKm(clat, clon, lastPos.latitude, lastPos.longitude);
      var midPos = positions[Math.floor(positions.length / 2)] || lastPos;
      var badgeIcon = L.divIcon({
        className: 'dynamic-distance-badge-wrap',
        html: '<div class="dynamic-distance-badge"><span class="dist-icon">⚡</span>' + totalDistKm.toFixed(1) + ' km DRIFT</div>',
        iconSize: [120, 24],
        iconAnchor: [60, 12],
      });
      L.marker([midPos.latitude, midPos.longitude], { icon: badgeIcon, interactive: false }).addTo(detectionLayerGroup);

      // Concentric landfall target rings at destination
      var landfallPoint = [lastPos.latitude, lastPos.longitude];
      var targetRings = [
        { radius: 12000, color: '#ff2d55', opacity: 0.35, weight: 1.5, dash: '8, 8' },
        { radius: 7000,  color: '#ff4757', opacity: 0.55, weight: 2.0, dash: '6, 6' },
        { radius: 3000,  color: '#ff6b81', opacity: 0.75, weight: 2.5, dash: '4, 4' },
      ];
      targetRings.forEach(function (ring) {
        L.circle(landfallPoint, {
          radius: ring.radius,
          color: ring.color,
          fillColor: ring.color,
          fillOpacity: 0.05,
          weight: ring.weight,
          dashArray: ring.dash,
          className: 'concentric-target-ring',
        }).addTo(detectionLayerGroup);
      });

      var bullseyeIcon = L.divIcon({
        className: 'target-bullseye-wrap',
        html: '<div class="target-bullseye-marker">🎯</div>',
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });
      L.marker(landfallPoint, { icon: bullseyeIcon })
        .bindPopup(
          '<strong style="color:#ff2d55">Predicted Landfall Target Zone</strong><br>' +
          'Forecast Horizon: T+72h<br>' +
          'Coordinates: ' + lastPos.latitude.toFixed(4) + '°N, ' + lastPos.longitude.toFixed(4) + '°E<br>' +
          'Total Trajectory Drift: ~' + totalDistKm.toFixed(1) + ' km'
        )
        .addTo(detectionLayerGroup);

      // Update interactive forecast range slider with live drift trajectory
      updateForecastTrajectoryData({ lat: clat, lon: clon }, positions, 72);
    }

    // ── Source attribution backtrack origin discharge point ─────────
    renderOriginAttributionOnMap(d.source_attribution, (clat != null ? { lat: clat, lon: clon } : null), detectionLayerGroup);

    // ── Q6 Autonomous Cleanup Intercept Vector on Map ───────────────
    renderLogisticsInterceptOnMap(d.logistics_plan);
  }

  /* ── Render Reverse Lagrangian Origin Discharge Point on Leaflet Map ── */
  function renderOriginAttributionOnMap(sa, centroid, targetLayer) {
    if (!sa) return;
    var layer = targetLayer || detectionLayerGroup;
    var ori = sa.primary_origin || sa.origin_coordinates;
    if (!ori || ori.latitude == null || ori.longitude == null) return;

    var oriLat = ori.latitude;
    var oriLon = ori.longitude;
    var srcType = sa.origin_type || sa.source_type || 'VESSEL_DISCHARGE';
    var srcName = (sa.nearest_source && sa.nearest_source.name) || sa.nearest_river_mouth || 'Maritime Shipping Corridor';
    var confScore = (typeof sa.confidence_score === 'number')
      ? (sa.confidence_score * 100).toFixed(1) + '%'
      : '94.2%';
    var srcDist = (sa.nearest_source && sa.nearest_source.distance_km != null)
      ? sa.nearest_source.distance_km.toFixed(1) + ' km'
      : (sa.river_mouth_dist_km != null ? sa.river_mouth_dist_km.toFixed(1) + ' km' : '—');
    var estTime = sa.discharge_timestamp_estimate
      ? new Date(sa.discharge_timestamp_estimate).toUTCString()
      : 'T-72h (Backtracked)';
    var totalBacktrackDist = sa.total_backtrack_distance_km != null
      ? sa.total_backtrack_distance_km.toFixed(1) + ' km'
      : (centroid ? haversineKm(centroid.lat, centroid.lon, oriLat, oriLon).toFixed(1) + ' km' : '—');

    // 1. Concentric purple radar pulse rings at origin discharge point
    var purpleRings = [
      { radius: 10000, opacity: 0.35, weight: 1.5, dash: '6, 6' },
      { radius: 5500,  opacity: 0.55, weight: 2.0, dash: '4, 4' },
      { radius: 2200,  opacity: 0.80, weight: 2.5, dash: '2, 3' }
    ];
    purpleRings.forEach(function (ring) {
      L.circle([oriLat, oriLon], {
        radius: ring.radius,
        color: '#a855f7',
        fillColor: '#9333ea',
        fillOpacity: 0.08,
        weight: ring.weight,
        dashArray: ring.dash,
        className: 'purple-origin-pulse-ring',
      }).addTo(layer);
    });

    // 2. Distinct Purple Target Marker (Bullseye with pulsing core)
    var purpleTargetIcon = L.divIcon({
      className: 'origin-discharge-marker-wrap',
      html: '<div class="origin-target-bullseye">' +
              '<div class="origin-ring-pulse"></div>' +
              '<div class="origin-center-pin">🎯</div>' +
            '</div>',
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });

    var originMarker = L.marker([oriLat, oriLon], { icon: purpleTargetIcon }).addTo(layer);

    var popupHtml =
      '<div class="origin-popup-card">' +
        '<div class="origin-popup-header">' +
          '<span class="origin-icon">🎯</span>' +
          '<strong>Origin Discharge Point</strong>' +
          '<span class="origin-badge-purple">' + srcType + '</span>' +
        '</div>' +
        '<div class="origin-popup-body">' +
          '<div class="origin-row"><span class="k">Attribution:</span> <span class="v highlight-purple">' + srcType + '</span></div>' +
          '<div class="origin-row"><span class="k">Nearest Entity:</span> <span class="v">' + srcName + ' (' + srcDist + ')</span></div>' +
          '<div class="origin-row"><span class="k">Confidence:</span> <span class="v conf-val">' + confScore + '</span></div>' +
          '<div class="origin-row"><span class="k">Est. Discharge:</span> <span class="v font-mono">' + estTime + '</span></div>' +
          '<div class="origin-row"><span class="k">Coordinates:</span> <span class="v font-mono">' + oriLat.toFixed(4) + '°N, ' + oriLon.toFixed(4) + '°E</span></div>' +
          '<div class="origin-row"><span class="k">Backtrack Distance:</span> <span class="v">' + totalBacktrackDist + '</span></div>' +
          '<div class="origin-desc">' + (sa.description || 'Reverse Lagrangian particle integration indicates origin discharge site.') + '</div>' +
        '</div>' +
      '</div>';

    originMarker.bindPopup(popupHtml);

    // 3. Dashed polyline connecting origin point backward to the current detected centroid
    var linePoints = [];
    if (centroid && centroid.lat != null && centroid.lon != null) {
      linePoints.push([centroid.lat, centroid.lon]);
    }
    if (sa.backtrack_path && sa.backtrack_path.length > 0) {
      sa.backtrack_path.forEach(function (pt) {
        if (pt.latitude != null && pt.longitude != null) {
          linePoints.push([pt.latitude, pt.longitude]);
        }
      });
    }
    if (linePoints.length === 0 || (linePoints[linePoints.length - 1][0] !== oriLat || linePoints[linePoints.length - 1][1] !== oriLon)) {
      linePoints.push([oriLat, oriLon]);
    }

    if (linePoints.length >= 2) {
      L.polyline(linePoints, {
        color: '#c084fc', // Distinct vivid purple
        weight: 3.5,
        opacity: 0.95,
        dashArray: '8, 8',
        className: 'reverse-backtrack-polyline',
      })
        .bindPopup(
          '<strong style="color:#c084fc">⏪ Reverse Lagrangian Backtrack Trajectory</strong><br>' +
          'Discharge Origin: ' + srcName + '<br>' +
          'Total Backtrack: ~' + totalBacktrackDist + '<br>' +
          'Confidence: ' + confScore + '<br>' +
          'Source: ' + srcType
        )
        .addTo(layer);
    }
  }

  function renderLogisticsInterceptOnMap(lp) {
    if (!lp) return;
    var port = lp.nearest_base_port;
    var sol = lp.intercept_solution;
    if (port && port.latitude != null && port.longitude != null && sol && sol.intercept_latitude != null) {
      var portLatLng = [port.latitude, port.longitude];
      var wpLatLng = [sol.intercept_latitude, sol.intercept_longitude];

      // Base port marker (⚓)
      var portIcon = L.divIcon({
        className: 'baseport-marker-wrap',
        html: '<div class="baseport-marker-pin">⚓</div>',
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });
      L.marker(portLatLng, { icon: portIcon })
        .bindPopup(
          '<strong style="color:#38bdf8">⚓ ' + (port.name || 'Coast Guard Base Port') + '</strong><br>' +
          'Agency: ' + (port.agency || 'Indian Coast Guard') + '<br>' +
          'State: ' + (port.state || 'India') + '<br>' +
          'Position: ' + port.latitude.toFixed(4) + '°N, ' + port.longitude.toFixed(4) + '°E'
        )
        .addTo(detectionLayerGroup);

      // Intercept waypoint marker (🚢)
      var wpIcon = L.divIcon({
        className: 'intercept-waypoint-wrap',
        html: '<div class="intercept-waypoint-pin">🚢</div>',
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });
      L.marker(wpLatLng, { icon: wpIcon })
        .bindPopup(
          '<strong style="color:#c084fc">🚢 Fleet Intercept Waypoint (T+' + (sol.target_hour || 12) + 'h)</strong><br>' +
          'Transit Hours: ' + (sol.vessel_transit_hours ? sol.vessel_transit_hours.toFixed(1) : '—') + ' hrs<br>' +
          'Intercept Coords: ' + sol.intercept_latitude.toFixed(4) + '°N, ' + sol.intercept_longitude.toFixed(4) + '°E<br>' +
          'Recommended Fleet: ' + (lp.fleet_recommendation ? lp.fleet_recommendation.skimmer_vessels_count + ' ASV Skimmer(s)' : '1 ASV')
        )
        .addTo(detectionLayerGroup);

      // Blue dashed intercept vector line
      L.polyline([portLatLng, wpLatLng], {
        color: '#38bdf8',
        weight: 3,
        opacity: 0.9,
        dashArray: '6, 6',
        className: 'logistics-intercept-line',
      })
        .bindPopup(
          '<strong style="color:#38bdf8">Clean-Up Fleet Intercept Trajectory</strong><br>' +
          'Origin: ' + (port.name || 'Base Port') + '<br>' +
          'Distance: ' + (lp.geodesic_distance ? lp.geodesic_distance.nautical_miles.toFixed(1) + ' NM' : '—') + '<br>' +
          'Speed: ' + (sol.cruise_speed_knots || 14) + ' knots'
        )
        .addTo(detectionLayerGroup);
    }
  }

  /* ── Clear all live-detection map overlays ──────────────────────── */
  function clearDetectionLayers() {
    if (detectionLayerGroup) detectionLayerGroup.clearLayers();
    centroidMarker = null;
    maskOverlay    = null;
  }

  /* ────────────────────────── SIMULATION ─────────────────────────── */
  function loadSimulation() {
    fetch(API_BASE + '/api/v1/simulation/msc-elsa-3')
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        renderSimulation(data);
        setEl('sys-status', 'ONLINE');
      })
      .catch(function (err) {
        console.warn('Backend API offline or unreachable, loading case study simulation baseline:', err);
        setEl('sys-status', 'SIMULATED');
        renderSimulation(getBaselineSimulationData());
      });
  }

  function getBaselineSimulationData() {
    return {
      case_name: 'MSC ELSA 3 Incident',
      incident_date: '2025-05-25',
      initial_centroid: { lat: 8.2177, lon: 76.752 },
      overall_mean_confidence: 89.4,
      triage: {
        threat_level: "Critical Threat",
        priority_score: 0.82,
        landfall_probability: 0.88,
      },
      geospatial: {
        calculated_radius_km: 12.0,
        debris_area_km2: 28.274,
        centroid_lat: 8.2177,
        centroid_lon: 76.752,
      },
      detection: {
        confidence_intensity: 0.894,
        debris_count: 5,
      },
      detections: [
        { centroid: { lat: 8.225, lon: 76.761 }, confidence: 0.94, confidence_percent: 94.2, label: 'Macro-Plastic', area_px: 38 },
        { centroid: { lat: 8.210, lon: 76.745 }, confidence: 0.91, confidence_percent: 91.0, label: 'Macro-Plastic', area_px: 26 },
        { centroid: { lat: 8.198, lon: 76.782 }, confidence: 0.88, confidence_percent: 88.5, label: 'Nurdles', area_px: 42 },
        { centroid: { lat: 8.232, lon: 76.738 }, confidence: 0.85, confidence_percent: 85.1, label: 'Macro-Plastic', area_px: 19 },
        { centroid: { lat: 8.185, lon: 76.810 }, confidence: 0.89, confidence_percent: 88.9, label: 'Nurdles', area_px: 33 }
      ],
      hotspots: [
        { id: 'HS-001', centroid: { lat: 8.215, lon: 76.758 }, radius_km: 12.0, intensity: 88.4, detection_count: 3 },
        { id: 'HS-002', centroid: { lat: 8.192, lon: 76.795 }, radius_km: 14.5, intensity: 78.2, detection_count: 2 }
      ],
      trajectories: [
        {
          origin_lat: 8.2177,
          origin_lon: 76.752,
          points: [
            { hour: 0, lat: 8.2177, lon: 76.752, risk_status: 'RELEASE' },
            { hour: 6, lat: 8.1810, lon: 76.8240, risk_status: 'OFFSHORE' },
            { hour: 12, lat: 8.1420, lon: 76.8980, risk_status: 'COASTAL_APPROACH' },
            { hour: 24, lat: 8.0850, lon: 77.0520, risk_status: 'HIGH_RISK_TRANSIT' },
            { hour: 36, lat: 8.0420, lon: 77.2150, risk_status: 'IMMINENT_LANDFALL' },
            { hour: 48, lat: 8.0120, lon: 77.3820, risk_status: 'SHORELINE_INTERCEPT' },
            { hour: 60, lat: 8.0350, lon: 77.5610, risk_status: 'GULF_OF_MANNAR_DRIFT' },
            { hour: 72, lat: 8.0820, lon: 77.7450, risk_status: 'ECO_RESERVE_IMPACT' }
          ]
        }
      ],
      priority_risk_zones: [
        {
          name: 'Kanyakumari Coastal Corridor',
          risk_level: 'CRITICAL',
          eta_hours: 38.5,
          recommendation: 'Pre-position offshore containment booms; mobilize municipal cleanup teams.',
          geometry: {
            coordinates: [[[77.10, 8.02], [77.35, 8.02], [77.35, 8.15], [77.10, 8.15], [77.10, 8.02]]]
          }
        },
        {
          name: 'Gulf of Mannar Biosphere Reserve',
          risk_level: 'HIGH',
          eta_hours: 58.0,
          recommendation: 'Deploy shallow-water skimmers to protect coral reef and seagrass beds.',
          geometry: {
            coordinates: [[[77.50, 8.00], [77.85, 8.00], [77.85, 8.20], [77.50, 8.20], [77.50, 8.00]]]
          }
        },
        {
          name: 'Kovalam Eco-Shoreline',
          risk_level: 'MODERATE',
          eta_hours: 14.0,
          recommendation: 'Continuous automated drone surveillance; notify coastal village council.',
          geometry: {
            coordinates: [[[76.80, 8.15], [77.05, 8.15], [77.05, 8.30], [76.80, 8.30], [76.80, 8.15]]]
          }
        }
      ],
      material_classification: {
        dominant_material: "Macro Plastics & Debris Bottles",
        floating_debris_index_fdi: 0.1245,
        plastic_surface_index_psi: 0.0832,
        composition_breakdown: {
          ghost_nets_and_gear: 24.5,
          macro_plastics_and_bottles: 62.0,
          marine_biomass_and_seaweed: 13.5
        },
        hazard_classification: "MEDIUM HAZARD: Floating Macro Plastics Ingestion Risk for Pelagic Fish & Turtles"
      },
      logistics_plan: {
        status: "success",
        nearest_base_port: {
          name: "Kochi Base Port",
          state: "Kerala",
          latitude: 9.9312,
          longitude: 76.2673,
          agency: "Indian Coast Guard District HQ 4"
        },
        geodesic_distance: {
          nautical_miles: 105.4,
          kilometers: 195.2
        },
        intercept_solution: {
          target_hour: 12,
          intercept_latitude: 8.142,
          intercept_longitude: 76.898,
          vessel_transit_hours: 7.5,
          cruise_speed_knots: 14.0
        },
        fleet_recommendation: {
          skimmer_vessels_count: 2,
          coverage_capacity_km2: 0.5,
          fuel_consumption_liters: 1575.0,
          estimated_fuel_cost_inr: 141750.0
        }
      },
      voice_advisories: {
        status: "success",
        incident_id: "SIM-ELSA-3",
        telugu_audio_url: "/static/audio/latest_te.mp3",
        english_audio_url: "/static/audio/latest_en.mp3"
      },
      source_attribution: {
        status: "success",
        origin_type: "VESSEL_DISCHARGE",
        source_type: "VESSEL_DISCHARGE",
        confidence_score: 0.942,
        origin_coordinates: { latitude: 8.2177, longitude: 76.752 },
        primary_origin: { latitude: 8.2177, longitude: 76.752 },
        nearest_source: {
          name: "MSC ELSA 3 Sinking Site / Kerala Coastal Shipping Corridor",
          type: "VESSEL_DISCHARGE",
          distance_km: 0.0,
          latitude: 8.2177,
          longitude: 76.752
        },
        nearest_river_mouth: "Kallada River Mouth",
        river_mouth_dist_km: 79.5,
        discharge_timestamp_estimate: "2025-05-25T14:30:00Z",
        description: "Reverse Lagrangian backtracking traces origin to confirmed sinking position of container carrier MSC ELSA 3 (~13 NM off Kerala Coast).",
        backtrack_path: [
          { hour: 0, latitude: 8.2177, longitude: 76.752 },
          { hour: -24, latitude: 8.162, longitude: 76.681 },
          { hour: -48, latitude: 8.115, longitude: 76.620 },
          { hour: -72, latitude: 8.080, longitude: 76.570 }
        ]
      }
    };
  }

  function renderSimulation(data) {
    var pScore = (data.triage && typeof data.triage.priority_score === 'number')
      ? data.triage.priority_score
      : 0.82;
    renderDetections(data.detections || []);
    renderHotspots(data.hotspots || [], pScore);
    renderSimTrajectory(data.trajectories || [], pScore);
    renderRiskZones(data.priority_risk_zones || []);
    renderSimTrajectoryNodes(data.trajectories || []);
    renderTelemetry(data);
    startCountdown(data);
    updateSidebarStatistics(data);
    renderQ3(data);
    renderQ5Spectral(data);
    renderQ6Logistics(data);
    renderVoiceAdvisories(data);
    renderLogisticsInterceptOnMap(data.logistics_plan);
    renderOriginAttributionOnMap(data.source_attribution, data.initial_centroid || { lat: 8.2177, lon: 76.752 }, map);

    // Initialise forecast slider with simulation drift trajectory
    if (data.trajectories && data.trajectories[0]) {
      var traj = data.trajectories[0];
      var origin = { lat: traj.origin_lat, lon: traj.origin_lon };
      var eta = (data.priority_risk_zones && data.priority_risk_zones[0] && data.priority_risk_zones[0].eta_hours != null)
        ? data.priority_risk_zones[0].eta_hours : 72;
      updateForecastTrajectoryData(origin, traj.points || [], eta);
    }
  }

  function renderDetections(detections) {
    detections.forEach(function (det, idx) {
      if (!det.centroid) return;
      var priorityScore = det.confidence != null ? det.confidence : ((det.confidence_percent || 50) / 100.0);
      var vis = getHotspotVisual(priorityScore);

      L.circleMarker([det.centroid.lat, det.centroid.lon], {
        radius: 8,
        color: vis.color,
        fillColor: vis.color,
        fillOpacity: 0.65,
        weight: 2,
      })
        .bindPopup(
          '<strong style="color:' + vis.color + '">Detection ' + (idx + 1) + '</strong><br>' +
          'Threat: ' + vis.label + '<br>' +
          'Class: ' + det.label + '<br>' +
          'Confidence: <span style="color:#39ff14">' + (det.confidence_percent || 0).toFixed(1) + '%</span><br>' +
          'Priority Score: ' + priorityScore.toFixed(2)
        )
        .addTo(map);
    });
  }

  function renderHotspots(hotspots, defaultScore) {
    hotspots.forEach(function (hs) {
      if (!hs.centroid) return;
      var priorityScore = (typeof defaultScore === 'number') ? defaultScore : ((hs.intensity || 50) / 100.0);
      var radiusKm = hs.radius_km || 12.0;
      var vis = getHotspotVisual(priorityScore);

      // 6-vertex hexagon polygon centered at (center_lat, center_lon) with radius scaled dynamically to calculated_radius_km
      var hexPoints = computeHexagonVertices(hs.centroid.lat, hs.centroid.lon, radiusKm);
      L.polygon(hexPoints, {
        color: vis.color,
        fillColor: vis.color,
        fillOpacity: 0.28,
        weight: 2.5,
        className: 'glowing-hex-debris',
      })
        .bindPopup(
          '<strong style="color:' + vis.color + '">⬡ Hotspot ' + hs.id + ' (' + vis.label + ')</strong><br>' +
          'Threat Level: ' + vis.label + '<br>' +
          'Priority Score: ' + priorityScore.toFixed(2) + '<br>' +
          'Calculated Radius: ' + radiusKm.toFixed(1) + ' km<br>' +
          'Intensity: ' + (hs.intensity || 0).toFixed(1) + '%'
        )
        .addTo(map);

      // Centroid marker with dynamic color tier
      L.circleMarker([hs.centroid.lat, hs.centroid.lon], {
        radius: Math.max(8, Math.min(22, Math.round(radiusKm * 1.3))),
        color: vis.color,
        fillColor: vis.color,
        fillOpacity: 0.85,
        weight: 2.5,
      })
        .bindPopup(
          '<strong style="color:' + vis.color + '">Hotspot ' + hs.id + '</strong><br>' +
          'Threat Level: ' + vis.label + '<br>' +
          'Priority Score: ' + priorityScore.toFixed(2) + '<br>' +
          'Radius: ' + radiusKm.toFixed(1) + ' km'
        )
        .addTo(map);
    });
  }

  function renderSimTrajectory(trajectories, defaultScore) {
    var visOrigin = getHotspotVisual(defaultScore || 0.82);
    trajectories.forEach(function (traj) {
      var latlngs = [];
      var origin = null;
      if (traj.origin_lat && traj.origin_lon) {
        origin = [traj.origin_lat, traj.origin_lon];
        latlngs.push(origin);

        // Glowing hexagon at release centroid styled with dynamic color tier
        var hexPoints = computeHexagonVertices(traj.origin_lat, traj.origin_lon, 2.8);
        L.polygon(hexPoints, {
          color: visOrigin.color,
          fillColor: visOrigin.color,
          fillOpacity: 0.32,
          weight: 3,
          className: 'glowing-hex-debris',
        })
          .bindPopup(
            '<strong style="color:' + visOrigin.color + '">⬡ Release Origin (' + visOrigin.label + ')</strong><br>' +
            'Coordinates: ' + traj.origin_lat.toFixed(4) + '°N, ' + traj.origin_lon.toFixed(4) + '°E<br>' +
            'Spill Source: MSC ELSA 3 Sinking Corridor'
          )
          .addTo(map);
      }

      var points = traj.points || [];
      points.forEach(function (p) { latlngs.push([p.lat, p.lon]); });
      if (latlngs.length < 2) return;

      // Green dashed trajectory line (#10B981)
      L.polyline(latlngs, {
        color: '#10B981',
        weight: 3.5,
        opacity: 0.95,
        dashArray: '8, 8',
        className: 'trajectory-green-dash',
      }).addTo(map);

      // Rotated heading arrows on nodes
      points.forEach(function (p, i) {
        var prevLat = (i === 0 && origin) ? origin[0] : points[i - 1].lat;
        var prevLon = (i === 0 && origin) ? origin[1] : points[i - 1].lon;
        var headingDeg = computeBearing(prevLat, prevLon, p.lat, p.lon);
        var cardinal = getCardinal(headingDeg);

        var arrowIcon = L.divIcon({
          className: 'heading-arrow-marker-wrap',
          html: '<div class="heading-arrow-marker" style="transform: rotate(' + headingDeg + 'deg);">' +
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none">' +
                '<path d="M12 2L19 21L12 17L5 21L12 2Z" fill="#10B981" stroke="#040912" stroke-width="2"/>' +
                '</svg></div>',
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        });

        L.marker([p.lat, p.lon], { icon: arrowIcon })
          .bindPopup(
            '<strong style="color:#10B981">T+' + p.hour + 'h Waypoint</strong><br>' +
            'Risk: ' + p.risk_status + '<br>' +
            'Heading: <strong>' + headingDeg.toFixed(1) + '° (' + cardinal + ')</strong><br>' +
            p.lat.toFixed(4) + '°N, ' + p.lon.toFixed(4) + '°E'
          )
          .addTo(map);
      });

      // Floating dynamic distance badge
      var lastP = points[points.length - 1];
      var totalDist = origin ? haversineKm(origin[0], origin[1], lastP.lat, lastP.lon) : 68.4;
      var midP = points[Math.floor(points.length / 2)] || lastP;
      var badgeIcon = L.divIcon({
        className: 'dynamic-distance-badge-wrap',
        html: '<div class="dynamic-distance-badge"><span class="dist-icon">⚡</span>' + totalDist.toFixed(1) + ' km DRIFT</div>',
        iconSize: [120, 24],
        iconAnchor: [60, 12],
      });
      L.marker([midP.lat, midP.lon], { icon: badgeIcon, interactive: false }).addTo(map);

      // Concentric landfall target rings at destination
      var landfallPoint = [lastP.lat, lastP.lon];
      var targetRings = [
        { radius: 12000, color: '#ff2d55', opacity: 0.35, weight: 1.5, dash: '8, 8' },
        { radius: 7000,  color: '#ff4757', opacity: 0.55, weight: 2.0, dash: '6, 6' },
        { radius: 3000,  color: '#ff6b81', opacity: 0.75, weight: 2.5, dash: '4, 4' },
      ];
      targetRings.forEach(function (ring) {
        L.circle(landfallPoint, {
          radius: ring.radius,
          color: ring.color,
          fillColor: ring.color,
          fillOpacity: 0.05,
          weight: ring.weight,
          dashArray: ring.dash,
          className: 'concentric-target-ring',
        }).addTo(map);
      });

      var bullseyeIcon = L.divIcon({
        className: 'target-bullseye-wrap',
        html: '<div class="target-bullseye-marker">🎯</div>',
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });
      L.marker(landfallPoint, { icon: bullseyeIcon })
        .bindPopup(
          '<strong style="color:#ff2d55">Predicted Landfall Target Zone</strong><br>' +
          'Forecast Horizon: T+72h<br>' +
          'Coordinates: ' + lastP.lat.toFixed(4) + '°N, ' + lastP.lon.toFixed(4) + '°E<br>' +
          'Total Trajectory Drift: ~' + totalDist.toFixed(1) + ' km'
        )
        .addTo(map);
    });
  }

  /* ──────────────── FORECAST RANGE SLIDER CONTROLLER ──────────────── */
  function haversineKm(lat1, lon1, lat2, lon2) {
    var R = 6371.0;
    var dLat = (lat2 - lat1) * Math.PI / 180.0;
    var dLon = (lon2 - lon1) * Math.PI / 180.0;
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  function wireForecastSlider() {
    var slider = document.getElementById('forecast-range-slider');
    var playBtn = document.getElementById('slider-play-btn');
    var stepBtns = document.querySelectorAll('.step-btn');

    if (slider) {
      slider.addEventListener('input', function (e) {
        var hour = parseInt(e.target.value, 10);
        setForecastHour(hour);
      });
    }

    if (playBtn) {
      playBtn.addEventListener('click', function () {
        toggleForecastAnimation();
      });
    }

    if (stepBtns) {
      stepBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
          var hour = parseInt(btn.getAttribute('data-hour'), 10);
          if (slider) slider.value = hour;
          setForecastHour(hour);
        });
      });
    }
  }

  function updateForecastTrajectoryData(origin, positions, etaHours) {
    currentForecastOrigin = origin;
    currentForecastTrajectory = positions || [];
    currentEtaHours = etaHours != null ? etaHours : 72;
    var slider = document.getElementById('forecast-range-slider');
    if (slider) slider.value = 0;
    setForecastHour(0);
  }

  function getProjectedPositionAtHour(hour) {
    if (!currentForecastOrigin) return null;
    if (hour <= 0 || !currentForecastTrajectory.length) {
      var nextP = currentForecastTrajectory[0];
      var nextLat = nextP ? (nextP.lat != null ? nextP.lat : nextP.latitude) : currentForecastOrigin.lat;
      var nextLon = nextP ? (nextP.lon != null ? nextP.lon : nextP.longitude) : currentForecastOrigin.lon;
      var h0Bearing = (nextP && (nextLat !== currentForecastOrigin.lat || nextLon !== currentForecastOrigin.lon))
        ? computeBearing(currentForecastOrigin.lat, currentForecastOrigin.lon, nextLat, nextLon)
        : 45.0;
      return {
        lat: currentForecastOrigin.lat,
        lon: currentForecastOrigin.lon,
        hour: 0,
        distKm: 0.0,
        heading_deg: h0Bearing,
      };
    }

    // Sort trajectory points by hour
    var sorted = currentForecastTrajectory.slice().sort(function (a, b) {
      var ha = a.hour != null ? a.hour : (a.forecast_hour || 0);
      var hb = b.hour != null ? b.hour : (b.forecast_hour || 0);
      return ha - hb;
    });

    var p0 = {
      lat: currentForecastOrigin.lat,
      lon: currentForecastOrigin.lon,
      hour: 0,
    };

    for (var i = 0; i < sorted.length; i++) {
      var cur = sorted[i];
      var chour = cur.hour != null ? cur.hour : (cur.forecast_hour || 0);
      var clat = cur.lat != null ? cur.lat : cur.latitude;
      var clon = cur.lon != null ? cur.lon : cur.longitude;

      if (hour === chour) {
        var d = haversineKm(currentForecastOrigin.lat, currentForecastOrigin.lon, clat, clon);
        var prevNode = (i === 0) ? p0 : {
          lat: sorted[i - 1].lat != null ? sorted[i - 1].lat : sorted[i - 1].latitude,
          lon: sorted[i - 1].lon != null ? sorted[i - 1].lon : sorted[i - 1].longitude,
        };
        var nodeHeading = (cur.heading_deg != null)
          ? cur.heading_deg
          : computeBearing(prevNode.lat, prevNode.lon, clat, clon);
        return { lat: clat, lon: clon, hour: hour, distKm: d, heading_deg: nodeHeading };
      }

      if (hour < chour) {
        var prev = (i === 0) ? p0 : {
          lat: sorted[i - 1].lat != null ? sorted[i - 1].lat : sorted[i - 1].latitude,
          lon: sorted[i - 1].lon != null ? sorted[i - 1].lon : sorted[i - 1].longitude,
          hour: sorted[i - 1].hour != null ? sorted[i - 1].hour : (sorted[i - 1].forecast_hour || 0),
        };
        var prevHour = prev.hour;
        var ratio = (hour - prevHour) / Math.max(1, (chour - prevHour));
        var ilat = prev.lat + ratio * (clat - prev.lat);
        var ilon = prev.lon + ratio * (clon - prev.lon);
        var idist = haversineKm(currentForecastOrigin.lat, currentForecastOrigin.lon, ilat, ilon);
        var segHeading = computeBearing(prev.lat, prev.lon, clat, clon);
        return { lat: ilat, lon: ilon, hour: hour, distKm: idist, heading_deg: segHeading };
      }
    }

    // Beyond last point
    var last = sorted[sorted.length - 1];
    var llat = last.lat != null ? last.lat : last.latitude;
    var llon = last.lon != null ? last.lon : last.longitude;
    var ldist = haversineKm(currentForecastOrigin.lat, currentForecastOrigin.lon, llat, llon);
    var pSecond = sorted.length > 1 ? sorted[sorted.length - 2] : p0;
    var pSecondLat = pSecond.lat != null ? pSecond.lat : pSecond.latitude;
    var pSecondLon = pSecond.lon != null ? pSecond.lon : pSecond.longitude;
    var lastHeading = computeBearing(pSecondLat, pSecondLon, llat, llon);
    return { lat: llat, lon: llon, hour: hour, distKm: ldist, heading_deg: lastHeading };
  }

  function setForecastHour(hour) {
    setEl('slider-hour-val', 'T+' + hour + 'h Forecast');
    updateTimerHUD(hour);

    // Update active step button
    document.querySelectorAll('.step-btn').forEach(function (btn) {
      if (parseInt(btn.getAttribute('data-hour'), 10) === hour) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    var pos = getProjectedPositionAtHour(hour);
    if (!pos) return;

    setEl('slider-coords', pos.lat.toFixed(4) + '°, ' + pos.lon.toFixed(4) + '°');
    setEl('slider-dist', pos.distKm.toFixed(2) + ' km');

    // Calculate landfall countdown
    var remainingHours = Math.max(0, currentEtaHours - hour);
    var countdownStr = '';
    if (remainingHours <= 0) {
      countdownStr = 'LANDFALL REACHED';
      setEl('slider-countdown', countdownStr);
    } else {
      var remSecs = Math.round(remainingHours * 3600);
      var rh = Math.floor(remSecs / 3600);
      var rm = Math.floor((remSecs % 3600) / 60);
      var rs = remSecs % 60;
      countdownStr = pad(rh) + ':' + pad(rm) + ':' + pad(rs);
      setEl('slider-countdown', countdownStr);
    }

    var heading = (pos.heading_deg != null) ? pos.heading_deg : 0;
    var cardinal = getCardinal(heading);

    // Rotated arrow icon at active marker position pointing in direction of motion (heading_deg)
    var arrowHtml =
      '<div class="forecast-active-arrow-wrap" style="transform: rotate(' + heading.toFixed(1) + 'deg);">' +
      '<svg width="28" height="28" viewBox="0 0 24 24" fill="none">' +
      '<path d="M12 2L20 20L12 16L4 20L12 2Z" fill="#10B981" stroke="#ffffff" stroke-width="2"/>' +
      '</svg></div>';

    var activeIcon = L.divIcon({
      className: 'forecast-active-marker',
      html: arrowHtml,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });

    // Move or create projected drift marker along trajectory line
    if (forecastMarker) {
      forecastMarker.setLatLng([pos.lat, pos.lon]);
      forecastMarker.setIcon(activeIcon);
    } else {
      forecastMarker = L.marker([pos.lat, pos.lon], {
        icon: activeIcon,
        zIndexOffset: 2000,
      }).addTo(map);
    }

    // Display a tooltip showing projected hours, coordinates, distance traveled (km), and landfall countdown
    var tooltipHtml =
      '<div class="forecast-hud-tooltip">' +
      '<div style="font-weight:700;color:#39ff14;margin-bottom:2px">Forecast T+' + hour + 'h</div>' +
      '<div>Coords: ' + pos.lat.toFixed(4) + '°N, ' + pos.lon.toFixed(4) + '°E</div>' +
      '<div>Distance: ' + pos.distKm.toFixed(1) + ' km</div>' +
      '<div>Heading: ' + heading.toFixed(1) + '° (' + cardinal + ')</div>' +
      '<div style="color:#22d3ee;font-weight:600">Landfall Countdown: ' + countdownStr + '</div>' +
      '</div>';

    if (forecastMarker.getTooltip()) {
      forecastMarker.setTooltipContent(tooltipHtml);
    } else {
      forecastMarker.bindTooltip(tooltipHtml, {
        permanent: true,
        direction: 'top',
        offset: [0, -14],
        className: 'forecast-active-tooltip',
      }).openTooltip();
    }

    forecastMarker.unbindPopup();
    forecastMarker.bindPopup(tooltipHtml);
  }

  function toggleForecastAnimation() {
    var playBtn = document.getElementById('slider-play-btn');
    var slider = document.getElementById('forecast-range-slider');

    if (isPlayingForecast) {
      clearInterval(forecastAnimTimer);
      isPlayingForecast = false;
      if (playBtn) {
        playBtn.classList.remove('playing');
        playBtn.innerHTML = '<span id="slider-play-icon">▶ Play Animation</span>';
      }
    } else {
      isPlayingForecast = true;
      if (playBtn) {
        playBtn.classList.add('playing');
        playBtn.innerHTML = '<span id="slider-play-icon">❚❚ Pause</span>';
      }

      forecastAnimTimer = setInterval(function () {
        var current = slider ? parseInt(slider.value, 10) : 0;
        var next = current + 6;
        if (next > 72) {
          next = 0;
        }
        if (slider) slider.value = next;
        setForecastHour(next);
      }, 850);
    }
  }

  function renderRiskZones(zones) {
    var list = document.getElementById('risk-zones');
    if (!list) return;
    list.innerHTML = '';

    if (!zones.length) {
      list.innerHTML = '<li class="traj-placeholder">No priority zones.</li>';
      return;
    }
    zones.forEach(function (zone) {
      var color = riskColor(zone.risk_level);

      // Map polygon
      if (zone.geometry && zone.geometry.coordinates) {
        var ring = zone.geometry.coordinates[0].map(function (c) { return [c[1], c[0]]; });
        L.polygon(ring, {
          color: color, weight: 2, fillColor: color, fillOpacity: 0.13,
        })
          .bindPopup(
            '<strong style="color:' + color + '">' + zone.name + ' Priority Zone</strong><br>' +
            'Risk: ' + zone.risk_level + '<br>' +
            'ETA: ' + (zone.eta_hours != null ? 'T+' + zone.eta_hours + 'h' : 'n/a') + '<br>' +
            '<span style="color:#94a3b8">' + zone.recommendation + '</span>'
          )
          .addTo(map);
      }

      // Sidebar list item
      var li = document.createElement('li');
      li.className = 'risk-item';
      li.innerHTML =
        '<div><p class="risk-item-name">' + zone.name + '</p>' +
        '<p class="risk-item-eta">ETA: ' + (zone.eta_hours != null ? 'T+' + zone.eta_hours + 'h' : 'n/a') + '</p></div>' +
        '<span class="risk-badge-sm" style="color:' + color + ';border:1px solid ' + color + '44;background:' + color + '18">' +
        zone.risk_level + '</span>';
      list.appendChild(li);
    });
  }

  function renderSimTrajectoryNodes(trajectories) {
    // Note: Q3 traj-list is used for live U-Net drift; simulation nodes go in risk-zones.
    // This function can be repurposed or left empty for simulation-only nodes.
  }

  function renderTelemetry(data) {
    setEl('stat-detections', (data.detections || []).length);
    setEl('stat-confidence',  (data.overall_mean_confidence || 0).toFixed(1) + '%');
  }

  function startCountdown(data) {
    if (countdownInterval) clearInterval(countdownInterval);
    var zones  = data.priority_risk_zones || [];
    var target = zones[0];
    zones.forEach(function (z) {
      if (z.eta_hours != null && (!target || target.eta_hours == null || z.eta_hours < target.eta_hours)) {
        target = z;
      }
    });
    var targetName  = target ? target.name : 'Kanyakumari';
    var etaHours    = target && target.eta_hours != null ? target.eta_hours : 72;
    setEl('countdown-target', targetName);
    var secs = Math.max(0, Math.round(etaHours * 3600));

    function tick() {
      if (secs <= 0) {
        setEl('countdown', 'IMPACT IMMINENT');
        clearInterval(countdownInterval);
        return;
      }
      var h = Math.floor(secs / 3600);
      var m = Math.floor((secs % 3600) / 60);
      var s = secs % 60;
      setEl('countdown', pad(h) + ':' + pad(m) + ':' + pad(s));
      secs -= 1;
    }
    tick();
    countdownInterval = setInterval(tick, 1000);
  }

  /* ────────────────────────── CLEAR STATE ─────────────────────────── */
  function clearAllQuadrantMetrics() {
    // Q1
    var q1 = document.getElementById('q1-results');
    if (q1) q1.classList.add('hidden');
    setMeter('meter-glint', 'meter-glint-pct', 0, 20);
    setMeter('meter-sat',   'meter-sat-pct',   0, 20);

    // Q2
    ['q2-area-m2','q2-area-km2','q2-pixels','q2-coverage',
     'q2-res-x','q2-res-y','q2-px-area','q2-centroid',
     'debrisAreaText','hotspotRadiusText','intensityText'].forEach(function(id){ setEl(id,'—'); });
    setEl('q2-cluster', 'AWAITING DETECTION');

    // Q3
    setEl('q3-source-type', '—');
    setEl('q3-source-desc', 'Awaiting detection…');
    var tl = document.getElementById('q3-traj-list');
    if (tl) tl.innerHTML = '<li class="traj-placeholder">Awaiting drift forecast…</li>';

    // Q4
    setEl('score-num', '—');
    var arc = document.getElementById('score-arc');
    if (arc) arc.style.strokeDashoffset = CIRCUMFERENCE;
    ['q4-threat-level','threatLevelText','priorityScoreText','q4-landfall','q4-habitat','q4-community',
     'q4-alert-status','q4-channels','q4-municipal','q4-incident-id','q4-alert-msg'].forEach(function(id){ setEl(id,'—'); });
    var dlBtn = document.getElementById('unet-geotiff');
    if (dlBtn) dlBtn.classList.add('hidden');

    // Q5
    setEl('mat-dominant-badge', '—');
    setEl('mat-fdi-value', '—');
    setEl('mat-psi-value', '—');
    ['mat-ghost-nets-bar','mat-macro-plastics-bar','mat-biomass-bar'].forEach(function(id){
      var bar = document.getElementById(id);
      if (bar) bar.style.width = '0%';
    });
    ['mat-ghost-nets-pct','mat-macro-plastics-pct','mat-biomass-pct'].forEach(function(id){ setEl(id, '—%'); });
    setEl('mat-hazard-warning', 'Awaiting spectral unmixing analysis…');

    // Q6
    ['logistics-port-name','logistics-dist-nm','logistics-intercept-point',
     'logistics-eta-hours','logistics-skimmer-fleet','logistics-fuel-liters'].forEach(function(id){ setEl(id,'—'); });
    setEl('logistics-port-badge', '—');

    // Q7
    setEl('voice-advisory-status', 'STANDBY');
    var te = document.getElementById('teluguAudioPlayer');
    if (te) { te.pause(); te.removeAttribute('src'); }
    var en = document.getElementById('englishAudioPlayer');
    if (en) { en.pause(); en.removeAttribute('src'); }
  }

  /* ────────────────────────── UI HELPERS ─────────────────────────── */
  function setEl(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setBadge(id, text, style) {
    var el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (style) el.setAttribute('style', style);
  }

  function setStatus(text, cls) {
    var el = document.getElementById('unet-status');
    if (!el) return;
    el.textContent = text;
    el.className   = 'status-line' + (cls ? ' ' + cls : '');
  }

  function showProcessingOverlay(show) {
    var el = document.getElementById('processing-overlay');
    if (!el) return;
    if (show) el.classList.remove('hidden');
    else      el.classList.add('hidden');
  }

  function fmtNum(n, decimals) {
    if (n == null) return '—';
    var d = decimals != null ? decimals : 2;
    return parseFloat(n).toFixed(d);
  }

  function pad(n) { return String(n).padStart(2, '0'); }

  function riskColor(level) {
    if (!level) return '#64748b';
    var l = level.toLowerCase();
    if (l === 'critical') return '#ff2d55';
    if (l === 'high')     return '#ffb020';
    if (l === 'moderate') return '#ffb020';
    return '#22d3ee';
  }

  function clusterAbbr(scale) {
    if (scale === 'MAJOR_PLASTIC_SPILL_WINDROW')    return 'MAJOR';
    if (scale === 'MODERATE_DEBRIS_FIELD')           return 'MOD';
    if (scale === 'DISPERSED_MICRO_ACCUMULATION')    return 'MICRO';
    return scale ? scale.substring(0, 6) : '—';
  }

  /* ──────────────── EMERGENCY ALERTS & PDF REPORT CONTROLS ───────── */
  function wireAlertAndReportControls() {
    var sendBtn = document.getElementById('send-advisory-btn');
    if (sendBtn) {
      sendBtn.addEventListener('click', function () {
        dispatchAdvisory();
      });
    }
  }

  function dispatchAdvisory() {
    var sendBtn = document.getElementById('send-advisory-btn');
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.textContent = '⚡ Dispatching Advisory…';
    }

    var alertMsgEl = document.getElementById('q4-alert-msg');
    var alertMsg = alertMsgEl ? alertMsgEl.textContent : '';
    var payload = {};
    if (alertMsg && alertMsg !== '—') {
      payload.message = alertMsg;
    }

    fetch(API_BASE + '/api/v1/alerts/dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.textContent = '⚡ Send Emergency Advisory';
        }
        var status = data.alert_status || 'TRIGGERED';
        setEl('q4-alert-status', status);
        var alertBadge = document.getElementById('alert-badge');
        if (alertBadge) {
          alertBadge.textContent = status;
          alertBadge.className = 'topbar-badge active';
        }
        showToast('⚠️ Emergency Maritime Advisory successfully dispatched via Twilio SMS & WhatsApp fleet channels!', 'success');
      })
      .catch(function (err) {
        console.warn('Dispatch failed', err);
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.textContent = '⚡ Send Emergency Advisory';
        }
        setEl('q4-alert-status', 'TRIGGERED');
        showToast('Advisory dispatched (simulated offline broadcast active).', 'info');
      });
  }

  function showToast(message, type) {
    var container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    var toast = document.createElement('div');
    toast.className = 'toast-notification' + (type ? ' ' + type : '');
    toast.innerHTML = '<span class="toast-dot"></span><span>' + message + '</span>';
    container.appendChild(toast);
    setTimeout(function () {
      toast.classList.add('visible');
    }, 10);
    setTimeout(function () {
      toast.classList.remove('visible');
      setTimeout(function () {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    }, 4500);
  }

  /* ────────────────── Q8: COORDINATE INSPECTOR ────────────────────── */
  var coordInspectLayerGroup = null;

  function wireCoordinateInspector() {
    var btn = document.getElementById('coord-predict-btn');
    if (btn) {
      btn.addEventListener('click', function () {
        runCoordinatePredict();
      });
    }
  }

  function runCoordinatePredict() {
    var latEl = document.getElementById('coord-lat-input');
    var lonEl = document.getElementById('coord-lon-input');
    var statusEl = document.getElementById('coord-status-line');
    var badgeEl = document.getElementById('coord-status-badge');
    var resultsEl = document.getElementById('coord-results');
    var btn = document.getElementById('coord-predict-btn');

    var lat = latEl ? parseFloat(latEl.value) : NaN;
    var lon = lonEl ? parseFloat(lonEl.value) : NaN;

    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      if (statusEl) { statusEl.textContent = 'Invalid coordinates. Enter valid Lat (-90..90) / Lon (-180..180).';
                      statusEl.className = 'status-line error'; }
      return;
    }

    if (btn) { btn.disabled = true; btn.textContent = '🌊 Fetching live ocean vectors…'; }
    if (statusEl) { statusEl.textContent = 'Querying Open-Meteo Marine API…'; statusEl.className = 'status-line'; }
    if (badgeEl) { badgeEl.textContent = 'LOADING'; badgeEl.setAttribute('style', 'color:#fbbf24;border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.1)'); }
    if (resultsEl) resultsEl.classList.add('hidden');

    fetch(API_BASE + '/api/v1/detect/coordinates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latitude: lat, longitude: lon })
    })
      .then(function (res) {
        return res.json().then(function (d) { return { ok: res.ok, data: d }; });
      })
      .then(function (r) {
        if (btn) { btn.disabled = false; btn.innerHTML = '<span class="run-icon">🌊</span> Predict 72h Drift'; }
        if (!r.ok) {
          if (statusEl) { statusEl.textContent = 'API error: ' + (r.data.detail || 'Unknown'); statusEl.className = 'status-line error'; }
          if (badgeEl) { badgeEl.textContent = 'ERROR'; badgeEl.setAttribute('style', 'color:#ff2d55;border-color:rgba(255,45,85,.4);background:rgba(255,45,85,.08)'); }
          return;
        }
        renderCoordinateResult(lat, lon, r.data);
      })
      .catch(function (err) {
        if (btn) { btn.disabled = false; btn.innerHTML = '<span class="run-icon">🌊</span> Predict 72h Drift'; }
        if (statusEl) { statusEl.textContent = 'Request failed: ' + err.message; statusEl.className = 'status-line error'; }
        if (badgeEl) { badgeEl.textContent = 'OFFLINE'; badgeEl.setAttribute('style', 'color:#ff2d55;border-color:rgba(255,45,85,.4);background:rgba(255,45,85,.08)'); }
      });
  }

  function renderCoordinateResult(origLat, origLon, d) {
    var statusEl = document.getElementById('coord-status-line');
    var badgeEl = document.getElementById('coord-status-badge');
    var resultsEl = document.getElementById('coord-results');

    var cp = d.coastal_proximity || {};
    var ra = d.risk_assessment || {};
    var df = d.drift_forecast || {};
    var sc = d.surface_conditions || {};
    var va = d.voice_advisories || {};
    var positions = df.positions || [];
    var firstPos = positions[0] || {};

    // Populate result fields
    setEl('coord-harbor', cp.nearest_harbor || '—');
    setEl('coord-dist', cp.distance_km != null ? cp.distance_km.toFixed(1) + ' km' : '—');

    var threat = (ra.threat_level || '—').toUpperCase();
    var threatEl = document.getElementById('coord-threat');
    if (threatEl) {
      threatEl.textContent = threat;
      threatEl.style.color = threat === 'CRITICAL' ? 'var(--danger)'
        : threat === 'HIGH' ? 'var(--warn)' : 'var(--neon)';
    }
    setEl('coord-score', ra.priority_score != null ? (ra.priority_score * 100).toFixed(1) + '%' : '—');
    setEl('coord-speed', firstPos.drift_speed_knots != null ? firstPos.drift_speed_knots.toFixed(2) + ' kts' : '—');
    setEl('coord-heading', firstPos.heading_deg != null ? firstPos.heading_deg.toFixed(1) + '° (' + (firstPos.cardinal_heading || '') + ')' : '—');
    setEl('coord-drift-km', df.total_drift_km != null ? df.total_drift_km.toFixed(1) + ' km' : '—');
    var oceanVel = sc.ocean_current_velocity_ms != null ? sc.ocean_current_velocity_ms.toFixed(3) + ' m/s @ ' + (sc.ocean_current_direction_deg || 0).toFixed(0) + '°' : 'fallback';
    setEl('coord-current', oceanVel);

    // Audio advisories
    if (va.telugu_audio_url) {
      var teAudio = document.getElementById('coordTeluguAudio');
      if (teAudio) { teAudio.src = API_BASE + va.telugu_audio_url; teAudio.load(); }
    }
    if (va.english_audio_url) {
      var enAudio = document.getElementById('coordEnglishAudio');
      if (enAudio) { enAudio.src = API_BASE + va.english_audio_url; enAudio.load(); }
    }

    // Show results
    if (resultsEl) resultsEl.classList.remove('hidden');
    if (statusEl) { statusEl.textContent = 'Prediction complete — 72h drift trajectory rendered on map.'; statusEl.className = 'status-line ok'; }
    if (badgeEl) { badgeEl.textContent = 'READY'; badgeEl.setAttribute('style', 'color:#10b981;border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)'); }

    // Render trajectory on map
    renderCoordinateTrajectory(origLat, origLon, positions, ra, cp);

    // Render Reverse Lagrangian Attribution origin and backtrack on map + update Q3
    if (d.source_attribution) {
      renderOriginAttributionOnMap(d.source_attribution, { lat: origLat, lon: origLon }, coordInspectLayerGroup);
      renderQ3(d);
    }

    showToast('🌊 72h drift prediction ready — ' + threat + ' threat at ' + (cp.nearest_harbor || 'coastal zone'), 'success');
  }

  function renderCoordinateTrajectory(origLat, origLon, positions, ra, cp) {
    // Clear previous coordinate inspect layers
    if (coordInspectLayerGroup) {
      coordInspectLayerGroup.clearLayers();
    } else {
      coordInspectLayerGroup = L.layerGroup().addTo(map);
    }

    var threatColor = (ra.threat_level === 'CRITICAL') ? '#ff2d55'
      : (ra.threat_level === 'HIGH') ? '#ffb020' : '#39ff14';

    // Origin pin
    var originIcon = L.divIcon({
      className: 'coord-origin-wrap',
      html: '<div class="coord-origin-pin" style="border-color:' + threatColor + ';box-shadow:0 0 14px ' + threatColor + '80">🎯</div>',
      iconSize: [28, 28], iconAnchor: [14, 14]
    });
    L.marker([origLat, origLon], { icon: originIcon })
      .bindPopup('<strong style="color:' + threatColor + '">Inspection Origin</strong><br>' +
        'Lat: ' + origLat.toFixed(5) + '°, Lon: ' + origLon.toFixed(5) + '°<br>' +
        'Nearest Harbor: ' + (cp.nearest_harbor || '—') + '<br>' +
        'Threat: ' + (ra.threat_level || '—') + ' (Score: ' + ((ra.priority_score || 0) * 100).toFixed(0) + '%)')
      .addTo(coordInspectLayerGroup);

    // Build trajectory polyline
    if (positions.length) {
      var latlngs = [[origLat, origLon]];
      positions.forEach(function (p) { latlngs.push([p.latitude, p.longitude]); });

      L.polyline(latlngs, {
        color: threatColor, weight: 3, opacity: 0.9, dashArray: '6, 5',
        className: 'coord-inspect-trajectory'
      }).addTo(coordInspectLayerGroup);

      // Waypoint markers
      positions.forEach(function (p, i) {
        if (i % 2 !== 0 && i !== positions.length - 1) return; // show every other + last
        L.circleMarker([p.latitude, p.longitude], {
          radius: i === positions.length - 1 ? 8 : 5,
          color: threatColor, fillColor: threatColor,
          fillOpacity: 0.75, weight: 2
        })
          .bindPopup('<strong style="color:' + threatColor + '">T+' + p.hour + 'h</strong><br>' +
            p.latitude.toFixed(4) + '°N, ' + p.longitude.toFixed(4) + '°E<br>' +
            'Speed: ' + (p.drift_speed_knots || 0).toFixed(2) + ' kts<br>' +
            'Heading: ' + (p.heading_deg || 0).toFixed(1) + '° (' + (p.cardinal_heading || '') + ')')
          .addTo(coordInspectLayerGroup);
      });

      // Fly to trajectory midpoint
      var midIdx = Math.floor(positions.length / 2);
      var midP = positions[midIdx] || positions[0];
      map.flyTo([midP.latitude, midP.longitude], 9, { duration: 1.4, easeLinearity: 0.5 });
    }
  }

})();
