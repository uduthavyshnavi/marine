/* AquaFlex AI - GIS dashboard controller.
   Initialises the Leaflet map, fetches the MSC ELSA 3 simulation payload,
   renders detection hotspots, drift trajectory, risk zones, and drives the
   72-hour forecast impact countdown timer. */

(function () {
  'use strict';

  var API_BASE =
    window.AQUAFLEX_API_BASE ||
    (function () {
      if (window.location.protocol === 'file:') return 'http://127.0.0.1:8000';
      var host = window.location.hostname || '127.0.0.1';
      var port = window.location.port;
      if (port === '8000') return '';
      return 'http://' + host + ':8000';
    })();

  var SINKING_SITE = { lat: 8.2177, lon: 76.752 };
  var MAP_CENTER = { lat: 8.6, lon: 77.6 };
  var MAP_ZOOM = 7;

  var map;
  var countdownInterval = null;

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    try {
      initMap();
    } catch (err) {
      console.error('Map init failed:', err);
      showError('Map failed to load: ' + err.message);
    }
    loadSimulation();
  }

  function initMap() {
    map = L.map('map', {
      center: [MAP_CENTER.lat, MAP_CENTER.lon],
      zoom: MAP_ZOOM,
      zoomControl: true,
      worldCopyJump: true,
    });

    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        attribution:
          'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQI | AquaFlex AI',
        maxZoom: 16,
      }
    ).addTo(map);

    addSinkingSiteMarker();
  }

  function addSinkingSiteMarker() {
    var site = L.circleMarker([SINKING_SITE.lat, SINKING_SITE.lon], {
      radius: 7,
      color: '#22d3ee',
      fillColor: '#22d3ee',
      fillOpacity: 0.9,
      weight: 2,
    }).addTo(map);

    site
      .bindPopup(
        '<strong style="color:#22d3ee">MSC ELSA 3 Sinking Site</strong><br>' +
          '~13 NM off Kerala<br>Sank 25 May 2025'
      )
      .openPopup();
  }

  function loadSimulation() {
    var url = API_BASE + '/api/v1/simulation/msc-elsa-3';
    fetch(url)
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        renderSimulation(data);
      })
      .catch(function (err) {
        console.error('Failed to load simulation:', err);
        showError('Unable to reach backend at ' + url + ' - ' + err.message);
      });
  }

  function renderSimulation(data) {
    renderDetections(data.detections || []);
    renderHotspots(data.hotspots || []);
    renderTrajectory(data.trajectories || []);
    renderRiskZones(data.priority_risk_zones || []);
    renderTrajectoryNodes(data.trajectories || []);
    renderTelemetry(data);
    startCountdown(data);
  }

  function renderDetections(detections) {
    detections.forEach(function (det, idx) {
      if (!det.centroid) return;
      var lat = det.centroid.lat;
      var lon = det.centroid.lon;
      var pct = (det.confidence_percent || 0).toFixed(1);

      var marker = L.circleMarker([lat, lon], {
        radius: 5 + (det.confidence_percent || 0) / 12,
        color: '#ff2d55',
        fillColor: '#ff2d55',
        fillOpacity: 0.55,
        weight: 1.5,
      }).addTo(map);

      marker.bindPopup(
        '<strong style="color:#ff2d55">Detection ' +
          (idx + 1) +
          '</strong><br>Class: ' +
          det.label +
          '<br>Confidence: <span style="color:#39ff14">' +
          pct +
          '%</span><br>Lat/Lon: ' +
          lat.toFixed(4) +
          ', ' +
          lon.toFixed(4)
      );
    });
  }

  function renderHotspots(hotspots) {
    hotspots.forEach(function (hs) {
      if (!hs.centroid) return;
      L.circle([hs.centroid.lat, hs.centroid.lon], {
        radius: (hs.radius_km || 10) * 1000,
        color: '#ff2d55',
        weight: 1,
        fillColor: '#ff2d55',
        fillOpacity: 0.12,
      })
        .bindPopup(
          '<strong style="color:#ff2d55">Hotspot ' +
            hs.id +
            '</strong><br>Intensity: ' +
            (hs.intensity || 0).toFixed(1) +
            '%<br>Detections: ' +
            hs.detection_count
        )
        .addTo(map);
    });
  }

  function renderTrajectory(trajectories) {
    trajectories.forEach(function (traj) {
      var latlngs = [];
      if (traj.origin_lat && traj.origin_lon) {
        latlngs.push([traj.origin_lat, traj.origin_lon]);
      }
      (traj.points || []).forEach(function (p) {
        latlngs.push([p.lat, p.lon]);
      });
      if (latlngs.length < 2) return;

      L.polyline(latlngs, {
        color: '#39ff14',
        weight: 3,
        opacity: 0.9,
        dashArray: '8, 10',
      }).addTo(map);

      (traj.points || []).forEach(function (p) {
        L.circleMarker([p.lat, p.lon], {
          radius: 4,
          color: '#39ff14',
          fillColor: '#05070d',
          fillOpacity: 1,
          weight: 2,
        })
          .bindPopup(
            '<strong style="color:#39ff14">T+' +
              p.hour +
              'h</strong><br>Risk: ' +
              p.risk_status +
              '<br>Lat/Lon: ' +
              p.lat.toFixed(4) +
              ', ' +
              p.lon.toFixed(4)
          )
          .addTo(map);
      });
    });
  }

  function renderRiskZones(zones) {
    var list = document.getElementById('risk-zones');
    if (!list) return;
    list.innerHTML = '';
    if (!zones.length) {
      list.innerHTML =
        '<li class="text-slate-500 text-xs font-mono">No priority zones.</li>';
      return;
    }

    zones.forEach(function (zone) {
      var color =
        zone.risk_level === 'Critical'
          ? '#ff2d55'
          : zone.risk_level === 'High'
          ? '#ffb020'
          : '#22d3ee';

      if (zone.geometry && zone.geometry.coordinates) {
        var ring = zone.geometry.coordinates[0].map(function (c) {
          return [c[1], c[0]];
        });
        L.polygon(ring, {
          color: color,
          weight: 2,
          fillColor: color,
          fillOpacity: 0.15,
        })
          .bindPopup(
            '<strong style="color:' +
              color +
              '">' +
              zone.name +
              ' Priority Zone</strong><br>Risk: ' +
              zone.risk_level +
              '<br>ETA: ' +
              (zone.eta_hours != null ? 'T+' + zone.eta_hours + 'h' : 'n/a') +
              '<br><span style="color:#94a3b8">' +
              zone.recommendation +
              '</span>'
          )
          .addTo(map);
      }

      var li = document.createElement('li');
      li.className =
        'flex items-start justify-between gap-2 rounded-lg bg-black/30 px-3 py-2 border border-white/5';
      li.innerHTML =
        '<div><p class="font-semibold text-slate-100">' +
        zone.name +
        '</p><p class="text-[11px] text-slate-400">ETA: ' +
        (zone.eta_hours != null ? 'T+' + zone.eta_hours + 'h' : 'n/a') +
        '</p></div><span class="text-[10px] font-mono px-2 py-0.5 rounded" style="color:' +
        color +
        ';border:1px solid ' +
        color +
        '55;background:' +
        color +
        '1a">' +
        zone.risk_level +
        '</span>';
      list.appendChild(li);
    });
  }

  function renderTrajectoryNodes(trajectories) {
    var list = document.getElementById('trajectory-nodes');
    if (!list) return;
    list.innerHTML = '';
    var traj = trajectories[0];
    if (!traj || !traj.points || !traj.points.length) {
      list.innerHTML = '<li class="text-slate-500 text-xs">No trajectory data.</li>';
      return;
    }
    traj.points.forEach(function (p) {
      var li = document.createElement('li');
      li.className = 'flex justify-between text-slate-300';
      li.innerHTML =
        '<span class="text-cyber">T+' +
        p.hour +
        'h</span><span>' +
        p.lat.toFixed(3) +
        ', ' +
        p.lon.toFixed(3) +
        '</span><span class="text-slate-500">' +
        p.risk_status +
        '</span>';
      list.appendChild(li);
    });
  }

  function renderTelemetry(data) {
    var det = document.getElementById('stat-detections');
    var conf = document.getElementById('stat-confidence');
    if (det) det.textContent = (data.detections || []).length;
    if (conf) conf.textContent = (data.overall_mean_confidence || 0).toFixed(1) + '%';
  }

  function startCountdown(data) {
    if (countdownInterval) clearInterval(countdownInterval);

    var zones = data.priority_risk_zones || [];
    var target = zones[0];
    zones.forEach(function (z) {
      if (z.eta_hours != null && (!target || target.eta_hours == null || z.eta_hours < target.eta_hours)) {
        target = z;
      }
    });

    var targetName = target ? target.name : 'Kanyakumari';
    var etaHours = target && target.eta_hours != null ? target.eta_hours : 72;

    var targetEl = document.getElementById('countdown-target');
    if (targetEl) targetEl.textContent = targetName;
    var badge = document.getElementById('alert-badge');
    var countdownEl = document.getElementById('countdown');

    var secondsRemaining = Math.max(0, Math.round(etaHours * 3600));

    function tick() {
      if (secondsRemaining <= 0) {
        if (countdownEl) countdownEl.textContent = 'IMPACT IMMINENT';
        if (badge) badge.textContent = 'IMPACT';
        if (badge) badge.classList.add('alert-active');
        if (countdownEl) countdownEl.classList.add('alert-active');
        clearInterval(countdownInterval);
        return;
      }
      var h = Math.floor(secondsRemaining / 3600);
      var m = Math.floor((secondsRemaining % 3600) / 60);
      var s = secondsRemaining % 60;
      if (countdownEl) countdownEl.textContent = pad(h) + ':' + pad(m) + ':' + pad(s);
      if (badge) badge.textContent = 'ACTIVE';
      if (badge) badge.classList.add('alert-active');
      secondsRemaining -= 1;
    }

    tick();
    countdownInterval = setInterval(tick, 1000);
  }

  function pad(n) {
    return String(n).padStart(2, '0');
  }

  function showError(message) {
    var list = document.getElementById('risk-zones');
    if (list) {
      list.innerHTML = '<li class="text-danger text-xs font-mono">' + message + '</li>';
    }
    var nodes = document.getElementById('trajectory-nodes');
    if (nodes) {
      nodes.innerHTML = '<li class="text-danger text-xs">Connection error.</li>';
    }
  }
})();

