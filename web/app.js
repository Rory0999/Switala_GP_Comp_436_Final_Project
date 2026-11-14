// User interface only: geometry editing, parameter collection, plotting and playback.
// All physics runs in Python (microseismic/ package) inside worker.js.
const $ = id => document.getElementById(id);

// ---- Model window (fixed) and defaults -------------------------------------------
const WIDTH = 600, Z_TOP = 2600, HEIGHT = 600, Z_BOT = Z_TOP + HEIGHT;
const DEFAULT_LAYERS = [
  ['Overburden siltstone', 190, 3900],
  ['Upper frac barrier (limestone)', 60, 4800],
  ['Target shale', 140, 3300],
  ['Lower shale', 90, 3700],
  ['Basal limestone', 120, 5000],
];
const SNR_STEPS = [null, 20, 10, 5, 3, 2, 1];
const MAX_RECEIVERS = 64;
const PRESETS = {
  well: () => line(560, 2700, 560, 3140, 16),
  lateral: () => line(80, 2760, 520, 2760, 12),
  L: () => [...line(560, 2700, 560, 3140, 12), ...line(80, 2760, 520, 2760, 12)],
  two: () => [...line(40, 2700, 40, 3140, 12), ...line(560, 2700, 560, 3140, 12)],
  clear: () => [],
};
function line(x0, z0, x1, z1, n) {
  return Array.from({ length: n }, (_, i) => [x0 + (x1 - x0) * i / (n - 1), z0 + (z1 - z0) * i / (n - 1)]);
}

// ---- State ------------------------------------------------------------------------
let layers = DEFAULT_LAYERS.map(l => [...l]);
let receivers = PRESETS.L();
let previewEvents = null, result = null, runInputs = null;
let worker = null, ready = false, busy = false, timer = null, requestId = 0, previewTimer = null;
let drag = null;

// ---- Worker -----------------------------------------------------------------------
function status(msg) { $('status').textContent = msg; }
function startWorker() {
  if (worker) worker.terminate();
  ready = false; busy = false; lock();
  const w = new Worker('./worker.js', { type: 'module' }); worker = w;
  w.onmessage = ({ data }) => {
    if (worker !== w) return;
    if (data.type === 'ready') { ready = true; status('Python is ready. Edit the model and geometry, then run the simulation.'); lock(); requestPreview(); }
    else if (data.type === 'progress') { $('progress').value = data.percent; status(`${data.phase} · ${data.percent}%`); }
    else if (data.type === 'error') {
      busy = false; lock();
      status(`Could not run: ${data.message}`);
    } else if (data.type === 'result') {
      if (data.result.kind === 'events') {
        if (data.id === requestId) { previewEvents = data.result.events; draw(); }
        return;
      }
      busy = false; result = typed(data.result); $('progress').value = 100;
      const nf = result.frame_times.length;
      $('frame').max = nf - 1; $('frame').value = Math.round((nf - 1) / 2);
      $('view').value = 'image';
      status(`Complete in ${result.elapsed.toFixed(1)} s · Python ran in this browser.`);
      lock(); showMetrics(); draw();
    }
  };
  w.onerror = e => { ready = false; busy = false; status(`Python could not load: ${e.message}. Check your connection and reload.`); lock(); };
}

// Flat arrays arrive as {shape, data}; make sure data is a Float32Array whatever the conversion gave.
function typed(o) {
  if (Array.isArray(o)) return o.map(typed);
  if (o && typeof o === 'object' && !ArrayBuffer.isView(o)) {
    if ('shape' in o && 'data' in o) return { shape: Array.from(o.shape), data: ArrayBuffer.isView(o.data) ? o.data : Float32Array.from(o.data) };
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typed(v)]));
  }
  return o;
}

function lock() {
  $('run').disabled = !ready || busy;
  $('run').textContent = !ready ? 'Loading Python…' : busy ? 'Running…' : 'Run forward model + time-reverse imaging';
  $('cancel').disabled = !busy;
  document.querySelectorAll('#controls input, #controls select, #controls button:not(#run):not(#cancel)')
    .forEach(el => { if (!el.dataset.fixed) el.disabled = busy; });
  $('play').disabled = !result || busy; $('frame').disabled = !result || busy;
}

// ---- Inputs -----------------------------------------------------------------------
const FRAC_FIELDS = ['x', 'z', 'half_length', 'height', 'upward_percent', 'n_events', 'time_spread_ms', 'magnitude_range', 'seed'];
function fracture() { const f = {}; for (const k of FRAC_FIELDS) f[k] = Number($(`frac-${k}`).value); return f; }
function numerics() { return { dx: Number($('num-dx').value), frequency: Number($('num-frequency').value), duration: Number($('num-duration').value) }; }
function verr() { return Number($('verr').value); }
function errorLayer() { const v = $('error-layer').value; return v === 'all' ? 'all' : Number(v); }
function snr() { return SNR_STEPS[Number($('snr').value)]; }

function renderLayers() {
  const body = $('layer-rows'); body.replaceChildren();
  layers.forEach((l, i) => {
    const tr = document.createElement('tr');
    const cell = (value, type, idx) => {
      const td = document.createElement('td'), inp = document.createElement('input');
      inp.type = type; inp.value = value;
      if (type === 'number') { inp.min = idx === 1 ? 5 : 1000; inp.max = idx === 1 ? 600 : 7000; inp.step = idx === 1 ? 5 : 50; }
      inp.addEventListener('change', () => { layers[i][idx] = type === 'number' ? Number(inp.value) : inp.value; renderErrorLayers(); invalidate(); });
      td.append(inp); return td;
    };
    const del = document.createElement('td'), b = document.createElement('button');
    b.type = 'button'; b.className = 'del'; b.textContent = '×'; b.title = 'Remove layer';
    b.disabled = layers.length <= 1;
    b.onclick = () => { layers.splice(i, 1); renderLayers(); invalidate(); };
    del.append(b);
    tr.append(cell(l[0], 'text', 0), cell(l[1], 'number', 1), cell(l[2], 'number', 2), del);
    body.append(tr);
  });
  renderErrorLayers();
}
function renderErrorLayers() {
  const sel = $('error-layer'), old = sel.value;
  sel.replaceChildren(new Option('All layers (uniform scaling)', 'all'), ...layers.map((l, i) => new Option(`Only: ${l[0]}`, String(i))));
  sel.value = [...sel.options].some(o => o.value === old) ? old : 'all';
}

function layerTops() { const tops = [Z_TOP]; for (const l of layers.slice(0, -1)) tops.push(tops.at(-1) + l[1]); return tops; }
function velocityAt(z) { const tops = layerTops(); let v = layers[0][2]; tops.forEach((t, i) => { if (z >= t) v = layers[i][2]; }); return v; }

function details() {
  const n = numerics(), vmax = Math.max(...layers.map(l => l[2])), vmin = Math.min(...layers.map(l => l[2]));
  const nx = Math.round(WIDTH / n.dx) + 1, dt = 0.45 * n.dx / vmax, nt = Math.ceil(n.duration / dt) + 1;
  const ppw = vmin / (2.5 * n.frequency) / n.dx, passes = verr() === 0 ? 2 : 3;
  $('details').textContent = `${nx} × ${nx} grid (+45-node sponge) · Δt = ${(1000 * dt).toFixed(3)} ms · ${nt} steps × ${passes} runs · ${ppw.toFixed(1)} points per minimum wavelength · ${receivers.length} receivers`;
  const warn = [];
  if (ppw < 6) warn.push('Coarse grid: fewer than 6 points per wavelength; expect numerical dispersion.');
  if (receivers.length < 4) warn.push('Place at least a few receivers.');
  const f = fracture();
  if (f.x - f.half_length < 0 || f.x + f.half_length > WIDTH || f.z - f.height * f.upward_percent / 100 < Z_TOP || f.z + f.height * (1 - f.upward_percent / 100) > Z_BOT)
    warn.push('The fracture extends outside the model.');
  const lastTop = layerTops().at(-1);
  if (lastTop >= Z_BOT) warn.push('Layers above the last one are thicker than the model; the deepest layers are not visible.');
  const maxDist = Math.hypot(WIDTH, HEIGHT) / vmin + f.time_spread_ms / 1000 + 1.5 / n.frequency;
  if (n.duration < Math.min(maxDist, 0.7) * 0.75) warn.push('Recording may be too short for far events to reach all receivers.');
  $('warning').textContent = warn.join(' ');
  $('receiver-count').textContent = `${receivers.length} / ${MAX_RECEIVERS} receivers. Drag receivers on the left model panel.`;
}

function invalidate(preview = true) {
  stop(); result = null; $('frame').value = 0; $('frame').max = 0; $('progress').value = 0;
  if ($('view').value !== 'model') $('view').value = 'model';
  $('metrics').innerHTML = '<p class="muted">Inputs changed. Run the simulation to compare locations.</p>';
  $('verr-label').textContent = `${verr() > 0 ? '+' : ''}${verr()} %`;
  $('snr-label').textContent = snr() === null ? 'No noise' : `SNR ${snr()}`;
  lock(); details(); draw();
  if (preview) requestPreview();
  if (ready && !busy) status('Inputs changed. Run the simulation to calculate new results.');
}
function requestPreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    if (!ready || busy) return;
    requestId += 1;
    worker.postMessage({ kind: 'events', id: requestId, fracture: fracture() });
  }, 150);
}

// ---- Plot helpers -----------------------------------------------------------------
function modelAxes(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const box = { x: 66, y: 12, w: canvas.width - 82, h: canvas.height - 62 };
  const X = x => box.x + x / WIDTH * box.w, Z = z => box.y + (z - Z_TOP) / HEIGHT * box.h;
  return { ctx, box, X, Z, canvas };
}
function frameAxes(a) {
  const { ctx, box } = a;
  ctx.strokeStyle = '#7d93a1'; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = '#425c6b'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
  for (let x = 0; x <= WIDTH; x += 100) { ctx.fillText(x, a.X(x), box.y + box.h + 16); }
  ctx.textAlign = 'right';
  for (let z = Z_TOP; z <= Z_BOT; z += 100) { ctx.fillText(z, box.x - 6, a.Z(z) + 4); }
  ctx.textAlign = 'center'; ctx.font = '13px system-ui';
  ctx.fillText('Distance along fracture plane, x (m)', box.x + box.w / 2, box.y + box.h + 40);
  ctx.save(); ctx.translate(14, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('True vertical depth (m)', 0, 0); ctx.restore();
}
function drawLayers(a, labels = true, fill = true) {
  const { ctx, box } = a, tops = layerTops(), vmin = 1500, vmax = 6500;
  layers.forEach((l, i) => {
    const z0 = Math.max(tops[i], Z_TOP), z1 = i + 1 < layers.length ? Math.min(tops[i + 1], Z_BOT) : Z_BOT;
    if (z1 <= z0) return;
    if (fill) {
      const g = Math.round(205 + 45 * (l[2] - vmin) / (vmax - vmin));
      ctx.fillStyle = `rgb(${g - 8},${g - 2},${g})`; ctx.fillRect(box.x, a.Z(z0), box.w, a.Z(z1) - a.Z(z0));
    }
    if (i > 0) { ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = fill ? '#8aa0ad' : 'rgba(255,255,255,.55)'; ctx.beginPath(); ctx.moveTo(box.x, a.Z(z0)); ctx.lineTo(box.x + box.w, a.Z(z0)); ctx.stroke(); ctx.restore(); }
    if (labels && a.Z(z1) - a.Z(z0) > 13) {
      ctx.font = '11.5px system-ui'; ctx.textAlign = 'left'; ctx.fillStyle = fill ? '#38515f' : 'rgba(255,255,255,.8)';
      ctx.fillText(`${l[0]} · ${l[2]} m/s`, box.x + 6, a.Z(z0) + 13);
    }
  });
}
function blit(a, field, nz, nx, colormap) {
  const img = new ImageData(nx, nz);
  for (let k = 0, n = nx * nz; k < n; k++) {
    const [r, g, b, al] = colormap(field[k]);
    img.data[4 * k] = r; img.data[4 * k + 1] = g; img.data[4 * k + 2] = b; img.data[4 * k + 3] = al;
  }
  const off = document.createElement('canvas'); off.width = nx; off.height = nz; off.getContext('2d').putImageData(img, 0, 0);
  a.ctx.imageSmoothingEnabled = true; a.ctx.drawImage(off, a.box.x, a.box.y, a.box.w, a.box.h);
}
const seismic = gain => v => {
  const s = Math.max(-1, Math.min(1, v * gain));
  return s >= 0 ? [255, Math.round(255 * (1 - s)), Math.round(255 * (1 - s)), Math.round(255 * Math.min(1, 1.6 * s))]
                : [Math.round(255 * (1 + s)), Math.round(255 * (1 + s)), 255, Math.round(255 * Math.min(1, -1.6 * s))];
};
const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const inferno = v => {
  const s = Math.max(0, Math.min(1, v)) * (INFERNO.length - 1), i = Math.min(INFERNO.length - 2, Math.floor(s)), f = s - i;
  const c = INFERNO[i].map((c0, k) => Math.round(c0 + f * (INFERNO[i + 1][k] - c0)));
  return [...c, 255];
};
function marker(a, x, z, text, fill, size = 16, stroke = '#10293a') {
  const { ctx } = a; ctx.font = `${size}px system-ui`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.lineWidth = 2.5; ctx.strokeStyle = stroke; ctx.strokeText(text, a.X(x), a.Z(z)); ctx.fillStyle = fill; ctx.fillText(text, a.X(x), a.Z(z));
  ctx.textBaseline = 'alphabetic';
}
function drawGeometry(a, events, picks, pairs, editable) {
  const { ctx } = a, f = fracture();
  // Fracture rectangle (where events are drawn from).
  const up = f.height * f.upward_percent / 100;
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = '#e8417a'; ctx.lineWidth = 1.5;
  ctx.strokeRect(a.X(f.x - f.half_length), a.Z(f.z - up), a.X(f.x + f.half_length) - a.X(f.x - f.half_length), a.Z(f.z - up + f.height) - a.Z(f.z - up));
  ctx.restore();
  if (picks && pairs) {
    ctx.strokeStyle = '#2dd05a'; ctx.lineWidth = 1.5;
    for (const [i, j] of pairs) { ctx.beginPath(); ctx.moveTo(a.X(events.x[i]), a.Z(events.z[i])); ctx.lineTo(a.X(picks.x[j]), a.Z(picks.z[j])); ctx.stroke(); }
  }
  if (events) events.x.forEach((x, i) => marker(a, x, events.z[i], '★', '#f4b229', 17));
  if (picks) picks.x.forEach((x, i) => {
    ctx.beginPath(); ctx.arc(a.X(x), a.Z(picks.z[i]), 9, 0, 2 * Math.PI);
    ctx.lineWidth = 4; ctx.strokeStyle = '#10293a'; ctx.stroke(); ctx.lineWidth = 2.2; ctx.strokeStyle = '#2dd05a'; ctx.stroke();
  });
  receivers.forEach(([x, z]) => marker(a, x, z, '▼', '#00c2d6', 13));
  marker(a, f.x, f.z, '✚', '#e8417a', 16);
  if (editable && drag && drag.kind === 'receiver') marker(a, ...receivers[drag.index], '▼', '#ffe14d', 15);
}

// ---- Main drawing -----------------------------------------------------------------
function draw() {
  const view = $('view').value, f = Number($('frame').value), gain = Number($('gain').value);
  const shown = result && runInputs ? runInputs.processing : { velocity_error_percent: verr(), error_layer: errorLayer() };
  const err = shown.velocity_error_percent;
  const errLabel = e => `${e > 0 ? '+' : ''}${e} % velocity${shown.error_layer !== 'all' ? ` in ${layers[shown.error_layer]?.[0] ?? 'one layer'}` : ' (all layers)'}`;
  $('frame-number').textContent = result ? `${f + 1} / ${result.frame_times.length}` : '0 / 0';
  $('gain-label').style.display = view === 'forward' || view === 'reverse' ? '' : 'none';
  for (const side of ['a', 'b']) {
    const a = modelAxes($(`panel-${side}`));
    const run = result ? result.runs[side === 'a' ? 'true' : 'wrong'] : null;
    const title = side === 'a' ? 'Correct velocity model' : `Back-propagated with ${errLabel(err)}`;
    $(`title-${side}`).textContent = side === 'b' && result && !run ? 'Wrong velocity model (velocity error was 0 %)' : title;
    let caption = '';
    if (!result || view === 'model' || (side === 'b' && !run && view !== 'forward')) {
      drawLayers(a, true, true);
      drawGeometry(a, result ? result.events : previewEvents, null, null, side === 'a');
      caption = side === 'a' ? 'Stars show the events that will be simulated (seeded). Edit the geometry here.' :
        (result && !run ? 'Set a nonzero velocity error and run again to compare.' : 'After a run, this panel shows imaging with the wrong velocity.');
    } else if (view === 'forward') {
      drawLayers(a, false, true);
      const fr = result.forward_frames, [nf, nz, nx] = fr.shape, n = nz * nx;
      blit(a, fr.data.subarray(f * n, (f + 1) * n), nz, nx, seismic(gain / maxAbs(fr.data)));
      drawLayers(a, true, false); drawGeometry(a, result.events, null, null, false);
      caption = `Forward time t = ${(1000 * result.frame_times[f]).toFixed(1)} ms (same recording for both panels). Blue = negative, red = positive pressure.`;
    } else if (view === 'reverse') {
      drawLayers(a, false, true);
      const fr = run.frames, [nf, nz, nx] = fr.shape, n = nz * nx;
      blit(a, fr.data.subarray(f * n, (f + 1) * n), nz, nx, seismic(gain / maxAbs(fr.data)));
      drawLayers(a, true, false); drawGeometry(a, result.events, null, null, false);
      const tau = run.frame_times[f], T = result.t.at(-1);
      caption = `Reverse time τ = ${(1000 * tau).toFixed(1)} ms ↔ forward time T − τ = ${(1000 * (T - tau)).toFixed(1)} ms. Energy refocuses at events.`;
    } else {
      const im = run.image; blit(a, im.data, im.shape[0], im.shape[1], inferno);
      drawLayers(a, true, false);
      drawGeometry(a, result.events, run.picks, run.pairs, false);
      const m = run.metrics;
      caption = `Max-over-time focus amplitude (receiver zones muted). Located ${m.n_matched}/${m.n_true} events, mean error ${fmt(m.mean_error)} m.`;
    }
    frameAxes(a);
    $(`caption-${side}`).textContent = caption;
  }
  drawGather();
}
const maxCache = new WeakMap();
function maxAbs(arr) { if (maxCache.has(arr)) return maxCache.get(arr); let m = 0; for (const v of arr) { const a = Math.abs(v); if (a > m) m = a; } m = m || 1; maxCache.set(arr, m); return m; }
function fmt(v, d = 1) { return v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d); }

function drawGather() {
  const canvas = $('gather'), ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result) { ctx.font = '15px system-ui'; ctx.fillStyle = '#68818f'; ctx.textAlign = 'center'; ctx.fillText('Recorded traces appear after a run.', canvas.width / 2, canvas.height / 2); return; }
  const box = { x: 66, y: 12, w: canvas.width - 82, h: canvas.height - 56 };
  const [nt, nr] = result.data.shape, d = result.data.data, T = result.t.at(-1);
  const X = t => box.x + t / T * box.w, Y = j => box.y + (j + 0.5) / nr * box.h;
  ctx.strokeStyle = '#7d93a1'; ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = '#425c6b'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
  const tick = [0.01, 0.02, 0.05, 0.1, 0.2].find(s => T / s <= 10) || 0.5;
  for (let t = 0; t <= T + 1e-9; t += tick) ctx.fillText((1000 * t).toFixed(0), X(t), box.y + box.h + 16);
  ctx.fillText('Recording time (ms)', box.x + box.w / 2, box.y + box.h + 36);
  ctx.save(); ctx.translate(14, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Receiver number', 0, 0); ctx.restore();
  ctx.textAlign = 'right';
  const spacing = box.h / nr, step = Math.max(1, Math.ceil(12 / spacing));
  for (let j = 0; j < nr; j += step) ctx.fillText(j + 1, box.x - 6, Y(j) + 4);
  const stride = Math.max(1, Math.floor(nt / 1500));
  for (let j = 0; j < nr; j++) {
    let m = 0; for (let i = 0; i < nt; i++) m = Math.max(m, Math.abs(d[i * nr + j]));
    m = m || 1; ctx.strokeStyle = j % 2 ? '#0b6e80' : '#16303f'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i < nt; i += stride) { const px = X(result.t[i]), py = Y(j) - 0.48 * spacing * d[i * nr + j] / m; if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
  }
  const view = $('view').value, f = Number($('frame').value);
  if (view === 'forward' || view === 'reverse') {
    const t = view === 'forward' ? result.frame_times[f] : T - result.runs.true.frame_times[f];
    ctx.save(); ctx.strokeStyle = '#df8f19'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(t), box.y); ctx.lineTo(X(t), box.y + box.h); ctx.stroke(); ctx.restore();
  }
}

function showMetrics() {
  const runs = [['true', 'Correct velocity'], ['wrong', 'Wrong velocity']].filter(([k]) => result.runs[k]);
  const ev = result.events;
  const rows = [
    ['Events located (matched / true)', m => `${m.n_matched} / ${m.n_true}`],
    ['False picks (no true event within 1.5 λ)', m => m.n_picked - m.n_matched, (m) => m.n_picked - m.n_matched > 0],
    ['Mean location error of matched events (m)', m => fmt(m.mean_error), m => m.mean_error > 15],
    ['Mean horizontal bias Δx (m, + = toward larger x)', m => fmt(m.mean_dx)],
    ['Mean depth bias Δz (m, + = deeper)', m => fmt(m.mean_dz)],
    ['Mean origin-time bias (ms)', m => fmt(m.mean_dt0_ms)],
    [`Apparent fracture length (m) · true ${fmt(result.runs.true.metrics.true_length, 0)}`, m => fmt(m.picked_length, 0)],
    [`Apparent fracture height (m) · true ${fmt(result.runs.true.metrics.true_height, 0)}`, m => fmt(m.picked_height, 0)],
    ['Shift of event-cloud centre (Δx, Δz) (m)', m => `${fmt(m.centroid_shift_x)}, ${fmt(m.centroid_shift_z)}`],
  ];
  let html = '<table><thead><tr><th></th>' + runs.map(([, l]) => `<th>${l}</th>`).join('') + '</tr></thead><tbody>';
  for (const [label, get, bad] of rows) html += `<tr><td>${label}</td>` + runs.map(([k]) => { const m = result.runs[k].metrics; return `<td class="${bad && bad(m) ? 'bad' : ''}">${get(m)}</td>`; }).join('') + '</tr>';
  html += '</tbody></table>';
  html += `<p class="note">${ev.x.length} events simulated; dominant wavelength in the slowest layer ≈ ${fmt(Math.min(...layers.map(l => l[2])) / runInputs.model.frequency, 0)} m, so events closer than about half of that at similar origin times cannot be separated. Picks are matched to the nearest true event (greedy, within 1.5 wavelengths). Apparent dimensions use all picks, including false ones.</p>`;
  $('metrics').innerHTML = html;
}

// ---- Geometry editing on the left panel -------------------------------------------
function eventToModel(e) {
  const canvas = $('panel-a'), rect = canvas.getBoundingClientRect(), a = modelAxesNoClear(canvas);
  const px = (e.clientX - rect.left) * canvas.width / rect.width, py = (e.clientY - rect.top) * canvas.height / rect.height;
  return { x: (px - a.box.x) / a.box.w * WIDTH, z: Z_TOP + (py - a.box.y) / a.box.h * HEIGHT, px, py, a };
}
function modelAxesNoClear(canvas) { const box = { x: 66, y: 12, w: canvas.width - 82, h: canvas.height - 62 }; return { box }; }
const clampX = x => Math.min(WIDTH, Math.max(0, x)), clampZ = z => Math.min(Z_BOT, Math.max(Z_TOP, z));
function nearestReceiver(p, maxPx = 12) {
  let best = -1, dist = maxPx;
  receivers.forEach(([x, z], i) => {
    const dx = (x - p.x) / WIDTH * p.a.box.w, dz = (z - p.z) / HEIGHT * p.a.box.h, d = Math.hypot(dx, dz);
    if (d < dist) { dist = d; best = i; }
  });
  return best;
}
$('panel-a').addEventListener('pointerdown', e => {
  if (busy) return;
  const p = eventToModel(e);
  if (p.x < 0 || p.x > WIDTH || p.z < Z_TOP || p.z > Z_BOT) return;
  const mode = $('mode').value;
  if (mode === 'receiver') {
    let i = nearestReceiver(p);
    if (i < 0) {
      if (receivers.length >= MAX_RECEIVERS) { status(`Maximum ${MAX_RECEIVERS} receivers. Delete one first.`); return; }
      receivers.push([p.x, p.z]); i = receivers.length - 1;
    }
    drag = { kind: 'receiver', index: i };
  } else if (mode === 'array') {
    if (!receivers.length) return;
    drag = { kind: 'array', start: p, original: receivers.map(r => [...r]) };
  } else if (mode === 'delete') {
    const i = nearestReceiver(p); if (i >= 0) { receivers.splice(i, 1); invalidate(false); }
    return;
  } else if (mode === 'perforation') {
    drag = { kind: 'perforation' }; setPerforation(p);
  }
  $('panel-a').setPointerCapture(e.pointerId);
  if ($('view').value !== 'model') $('view').value = 'model';
  draw();
});
function setPerforation(p) { $('frac-x').value = Math.round(clampX(p.x)); $('frac-z').value = Math.round(clampZ(p.z)); }
$('panel-a').addEventListener('pointermove', e => {
  if (!drag) return;
  const p = eventToModel(e);
  if (drag.kind === 'receiver') receivers[drag.index] = [clampX(p.x), clampZ(p.z)];
  else if (drag.kind === 'array') {
    let dx = p.x - drag.start.x, dz = p.z - drag.start.z;
    const xs = drag.original.map(r => r[0]), zs = drag.original.map(r => r[1]);
    dx = Math.min(WIDTH - Math.max(...xs), Math.max(-Math.min(...xs), dx));
    dz = Math.min(Z_BOT - Math.max(...zs), Math.max(Z_TOP - Math.min(...zs), dz));
    receivers = drag.original.map(([x, z]) => [x + dx, z + dz]);
  } else if (drag.kind === 'perforation') { setPerforation(p); previewEvents = null; }
  result = null; draw(); details();
});
const endDrag = () => { if (!drag) return; const kind = drag.kind; drag = null; invalidate(kind === 'perforation'); };
$('panel-a').addEventListener('pointerup', endDrag);
$('panel-a').addEventListener('pointercancel', endDrag);
$('panel-a').addEventListener('contextmenu', e => {
  e.preventDefault(); if (busy) return;
  const i = nearestReceiver(eventToModel(e)); if (i >= 0) { receivers.splice(i, 1); invalidate(false); }
});

// ---- Controls ---------------------------------------------------------------------
document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => { receivers = PRESETS[b.dataset.preset](); invalidate(false); }));
// New layers are inserted above the deepest layer, which always fills the model base.
$('add-layer').onclick = () => { layers.splice(layers.length - 1, 0, ['New layer', 40, 4000]); renderLayers(); invalidate(false); };
$('reset-layers').onclick = () => { layers = DEFAULT_LAYERS.map(l => [...l]); renderLayers(); invalidate(false); };
for (const k of FRAC_FIELDS) $(`frac-${k}`).addEventListener('change', () => { previewEvents = null; invalidate(); });
for (const id of ['num-frequency', 'num-dx', 'num-duration', 'error-layer']) $(id).addEventListener('change', () => invalidate(false));
for (const id of ['verr', 'snr']) $(id).addEventListener('input', () => invalidate(false));
$('controls').addEventListener('submit', e => {
  e.preventDefault(); if (!ready || busy) return;
  if (!receivers.length) { status('Place at least one receiver or choose a preset.'); return; }
  invalidate(false);
  runInputs = {
    kind: 'experiment', id: ++requestId,
    model: { ...numerics(), layers: layers.map(l => [String(l[0]), Number(l[1]), Number(l[2])]) },
    fracture: fracture(), receivers: receivers.map(r => [...r]),
    processing: { velocity_error_percent: verr(), error_layer: errorLayer(), snr: snr() },
  };
  busy = true; lock(); status('Starting simulation…');
  worker.postMessage(runInputs);
});
$('cancel').onclick = () => { stop(); status('Calculation cancelled. Restarting Python…'); startWorker(); };
$('view').onchange = () => { stop(); draw(); };
$('gain').oninput = draw;
$('frame').oninput = () => { stop(); draw(); };
function stop() { clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
$('play').onclick = () => {
  if (timer) { stop(); return; } if (!result) return;
  if ($('view').value === 'model' || $('view').value === 'image') $('view').value = 'forward';
  if (Number($('frame').value) >= Number($('frame').max)) $('frame').value = 0;
  $('play').textContent = 'Pause';
  timer = setInterval(() => { const next = Number($('frame').value) + 1; if (next > Number($('frame').max)) { stop(); return; } $('frame').value = next; draw(); }, 90);
};

renderLayers();
$('verr-label').textContent = `${verr() > 0 ? '+' : ''}${verr()} %`;
details(); draw(); startWorker();
