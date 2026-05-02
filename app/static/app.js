/* === app.js — Generative Fashion Designer SPA === */
'use strict';

const API = '';  // same-origin
let currentModel = 'vae';
let lastB64 = null;
let lastMeta = {};
let galleryData = [];
let activeFilter = 'all';

const MODEL_INFO = {
  vae: { name:'β-VAE', desc:'Learns a structured latent space for smooth interpolation and reconstruction. Uses ResNet encoder + transposed-conv decoder.' },
  dcgan: { name:'DCGAN', desc:'Classic adversarial training with spectral normalization and self-attention for stable, high-frequency texture synthesis.' },
  wgan_gp: { name:'WGAN-GP', desc:'Wasserstein distance training with gradient penalty. More stable convergence and meaningful loss curves.' },
  cgan: { name:'cGAN', desc:'Class-conditional generation targeting specific texture categories using projection discriminator.' },
  latent_dit: { name:'Latent DiT', desc:'State-of-the-art Latent Diffusion model using a Transformer backbone (DiT) on 8x8 latent space.' },
};

/* ── Init ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initHeroCanvas();
  checkHealth();
  loadClasses();
  loadMetrics();
  loadGallery();
  initModelTabs();
  initGalleryFilter();
  setInterval(checkHealth, 30000);
});

/* ── Nav scroll highlight ───────────────────────── */
function initNavScroll() {
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
    const sections = ['generate','compare','metrics','gallery'];
    let active = 'generate';
    for (const id of sections) {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top < 120) active = id;
    }
    document.querySelectorAll('.nav-link').forEach(l => {
      l.classList.toggle('active', l.dataset.section === active);
    });
  }, { passive: true });
}

function scrollToSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
}

/* ── Hero animated canvas ────────────────────────── */
function initHeroCanvas() {
  const grid = document.getElementById('canvasGrid');
  if (!grid) return;
  const colors = [
    ['#1a0a2e','#3b1f6b'],['#0a1f3b','#1a4a6b'],
    ['#0f2a1a','#1a5a3b'],['#2a1a0f','#5a3b1a'],
    ['#1f0a1f','#4a1f4a'],['#0a2a2a','#1a5a5a'],
  ];
  for (let i = 0; i < 16; i++) {
    const tile = document.createElement('div');
    tile.className = 'canvas-tile';
    const [c1,c2] = colors[i % colors.length];
    tile.style.background = `linear-gradient(${Math.random()*360}deg, ${c1}, ${c2})`;
    tile.style.animationDelay = `${Math.random()*3}s`;
    tile.style.animationDuration = `${2+Math.random()*2}s`;
    grid.appendChild(tile);
  }
}

/* ── Health check ────────────────────────────────── */
async function checkHealth() {
  const dot = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    dot.className = 'status-dot online';
    text.textContent = d.cuda ? `Online · ${d.gpu.replace('NVIDIA ','').substring(0,20)}` : 'Online · CPU';
    document.getElementById('heroGpu').textContent = d.cuda ? d.gpu.split(' ').slice(-2).join(' ') : 'CPU';
  } catch {
    dot.className = 'status-dot error';
    text.textContent = 'Server offline';
  }
}

/* ── Load classes ────────────────────────────────── */
async function loadClasses() {
  try {
    const r = await fetch(`${API}/api/classes`);
    const d = await r.json();
    const selects = ['classSelect','compareClass'];
    for (const sid of selects) {
      const sel = document.getElementById(sid);
      if (!sel) continue;
      if (sid === 'classSelect') sel.innerHTML = '<option value="">— Random —</option>';
      else sel.innerHTML = '';
      d.classes.forEach((c,i) => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c.charAt(0).toUpperCase()+c.slice(1);
        if (sid === 'compareClass' && i === 0) opt.selected = true;
        sel.appendChild(opt);
      });
    }
  } catch(e) { console.warn('Could not load classes', e); }
}

/* ── Model tabs ──────────────────────────────────── */
function initModelTabs() {
  document.querySelectorAll('.model-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentModel = tab.dataset.model;
      updateModelInfo();
      // Show class selector only for cGAN
      const classGroup = document.getElementById('classGroup');
      if (classGroup) classGroup.style.opacity = currentModel === 'cgan' ? '1' : '0.4';
    });
  });
  updateModelInfo();
}

function updateModelInfo() {
  const info = MODEL_INFO[currentModel] || {};
  const el = document.getElementById('modelInfo');
  if (el) {
    el.querySelector('.model-info-title').textContent = info.name || currentModel;
    el.querySelector('.model-info-desc').textContent = info.desc || '';
  }
  document.getElementById('outputTitle').textContent = `Output · ${(info.name || currentModel)}`;
}

/* ── Generate ────────────────────────────────────── */
async function handleGenerate() {
  const btn = document.getElementById('generateBtn');
  const btnText = document.getElementById('generateBtnText');
  const num = parseInt(document.getElementById('sampleCount').value);
  const classLabel = currentModel === 'cgan' ? document.getElementById('classSelect').value || null : null;

  setOutputLoading(true);
  btn.disabled = true;
  btnText.textContent = 'Generating…';

  try {
    const r = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: currentModel, num_samples: num, class_label: classLabel }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    showOutput(d.b64, d);
    showToast(`Generated ${num} samples with ${currentModel.toUpperCase()}`, 'success');
    loadGallery();
  } catch(e) {
    showToast('Generation failed: ' + e.message, 'error');
    setOutputLoading(false, true);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Generate';
  }
}

async function handleInterpolate() {
  setOutputLoading(true);
  try {
    const r = await fetch(`${API}/api/interpolate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: currentModel, steps: 10 }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    showOutput(d.b64, { ...d, num_samples: d.steps, class_label: null });
    showToast(`Interpolation (${d.steps} steps) complete`, 'success');
  } catch(e) {
    showToast('Interpolation failed: ' + e.message, 'error');
    setOutputLoading(false, true);
  }
}

function setOutputLoading(isLoading, isError = false) {
  const placeholder = document.getElementById('outputPlaceholder');
  const loading = document.getElementById('outputLoading');
  const img = document.getElementById('outputImage');
  const loadingModel = document.getElementById('loadingModel');

  if (isLoading) {
    placeholder.style.display = 'none';
    loading.style.display = 'block';
    img.style.display = 'none';
    if (loadingModel) loadingModel.textContent = `model: ${currentModel}`;
  } else if (isError) {
    placeholder.style.display = 'block';
    loading.style.display = 'none';
    img.style.display = 'none';
  }
}

function showOutput(b64, meta) {
  const placeholder = document.getElementById('outputPlaceholder');
  const loading = document.getElementById('outputLoading');
  const img = document.getElementById('outputImage');

  placeholder.style.display = 'none';
  loading.style.display = 'none';

  img.src = `data:image/png;base64,${b64}`;
  img.style.display = 'block';
  img.onclick = () => openLightbox(`data:image/png;base64,${b64}`, `${meta.model?.toUpperCase()} · ${meta.num_samples} samples`);

  document.getElementById('outputMeta').textContent =
    `${meta.num_samples} samples · ${meta.model}${meta.class_label ? ` · ${meta.class_label}` : ''}`;
  document.getElementById('outputTimestamp').textContent =
    meta.timestamp ? `generated at ${new Date(meta.timestamp*1000).toLocaleTimeString()}` : '';

  const dlBtn = document.getElementById('downloadBtn');
  const fsBtn = document.getElementById('fullscreenBtn');
  if (dlBtn) dlBtn.disabled = false;
  if (fsBtn) fsBtn.disabled = false;

  lastB64 = b64;
  lastMeta = meta;
}

function downloadOutput() {
  if (!lastB64) return;
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${lastB64}`;
  a.download = `fashionai_${lastMeta.model || 'output'}_${Date.now()}.png`;
  a.click();
}

function toggleFullscreen() {
  if (!lastB64) return;
  openLightbox(`data:image/png;base64,${lastB64}`,
    `${lastMeta.model?.toUpperCase()} · ${lastMeta.num_samples} samples`);
}

/* ── Compare ─────────────────────────────────────── */
async function handleCompare() {
  const cls = document.getElementById('compareClass').value;
  const mdl = document.getElementById('compareModel').value;

  setCompareLoading(true);

  try {
    // Run both in parallel
    const [genRes, gemRes] = await Promise.all([
      fetch(`${API}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: mdl, num_samples: 9, class_label: mdl === 'cgan' ? cls : null }),
      }).then(r => r.json()),
      fetch(`${API}/api/gemini-compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ class_name: cls, size: 256 }),
      }).then(r => r.json()),
    ]);

    // Our model output
    const ourWrap = document.getElementById('compareOurWrap');
    const ourLabel = document.getElementById('compareOurLabel');
    const ourCap = document.getElementById('compareOurCaption');
    if (!genRes.error) {
      ourWrap.innerHTML = '';
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${genRes.b64}`;
      img.alt = `${mdl} generated ${cls}`;
      img.onclick = () => openLightbox(img.src, `${mdl.toUpperCase()} output — ${cls}`);
      ourWrap.appendChild(img);
      ourLabel.textContent = mdl.toUpperCase();
      ourCap.textContent = `${mdl} · 9 samples${mdl === 'cgan' ? ` · ${cls}` : ''}`;
    } else {
      ourWrap.innerHTML = `<div class="compare-placeholder"><span>Error: ${genRes.error}</span></div>`;
    }

    // Gemini reference
    const gemWrap = document.getElementById('compareGeminiWrap');
    const gemCap = document.getElementById('compareGeminiCaption');
    if (!gemRes.error) {
      gemWrap.innerHTML = '';
      const img = document.createElement('img');
      img.src = `data:image/png;base64,${gemRes.b64}`;
      img.alt = `Gemini reference for ${cls}`;
      img.onclick = () => openLightbox(img.src, `Gemini Vision reference — ${cls}`);
      gemWrap.appendChild(img);
      const srcLabel = gemRes.source === 'gemini' ? 'Gemini Vision API' :
                       gemRes.source === 'gemini_cached' ? 'Gemini (cached)' :
                       'Procedural fallback';
      gemCap.textContent = `${srcLabel} · ${cls}`;
    } else {
      gemWrap.innerHTML = `<div class="compare-placeholder"><span>Error: ${gemRes.error}</span></div>`;
    }

    showToast('Comparison ready', 'success');
  } catch(e) {
    showToast('Comparison failed: ' + e.message, 'error');
  }
}

function setCompareLoading(isLoading) {
  const ids = ['compareOurWrap', 'compareGeminiWrap'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el && isLoading) el.innerHTML = '<div class="compare-placeholder shimmer" style="min-height:200px;width:100%"></div>';
  }
}

/* ── Metrics ─────────────────────────────────────── */
async function loadMetrics() {
  try {
    const r = await fetch(`${API}/api/metrics`);
    const d = await r.json();
    renderMetrics(d.metrics);
  } catch(e) {
    document.getElementById('metricsGrid').innerHTML = '<p style="color:var(--text-3);padding:32px">Could not load metrics</p>';
  }
}

function renderMetrics(metrics) {
  const grid = document.getElementById('metricsGrid');
  const modelNames = { vae:'β-VAE', dcgan:'DCGAN', wgan_gp:'WGAN-GP', cgan:'cGAN', latent_dit:'Latent DiT' };
  let html = '';
  for (const [modelId, vals] of Object.entries(metrics)) {
    const rows = Object.entries(vals).map(([k,v]) =>
      `<div class="metric-row"><span class="metric-key">${k.toUpperCase().replace('_',' ')}</span><span class="metric-val">${typeof v === 'number' ? v.toFixed(2) : v}</span></div>`
    ).join('');
    html += `<div class="metric-card">
      <div class="metric-model">${modelId}</div>
      <div class="metric-name">${modelNames[modelId] || modelId}</div>
      ${rows}
    </div>`;
  }
  grid.innerHTML = html || '<p style="color:var(--text-3)">No metrics available</p>';
}

/* ── Gallery ─────────────────────────────────────── */
async function loadGallery() {
  try {
    const r = await fetch(`${API}/api/gallery`);
    const d = await r.json();
    galleryData = d.gallery || [];
    renderGallery();
  } catch(e) {
    document.getElementById('galleryGrid').innerHTML = '<p style="color:var(--text-3);padding:32px;grid-column:1/-1">Could not load gallery</p>';
  }
}

function initGalleryFilter() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeFilter = btn.dataset.filter;
      renderGallery();
    });
  });
}

function renderGallery() {
  const grid = document.getElementById('galleryGrid');
  const items = activeFilter === 'all' ? galleryData : galleryData.filter(i => i.model === activeFilter);

  if (!items.length) {
    grid.innerHTML = '<div class="gallery-loading">No items found. Generate some textures first!</div>';
    return;
  }

  const ts = i => i.timestamp ? new Date(i.timestamp*1000).toLocaleTimeString() : 'Saved';

  grid.innerHTML = items.map(item => `
    <div class="gallery-item" onclick="openGalleryItem('${item.filename}','${item.model}')">
      <div style="height:160px;background:linear-gradient(135deg,var(--bg2),var(--bg3));display:flex;align-items:center;justify-content:center">
        <div class="loading-spinner" style="width:28px;height:28px;border-width:2px"></div>
      </div>
      <div class="gallery-item-meta">
        <span class="gallery-model-badge badge-${item.model}">${item.model}</span>
        <span class="gallery-time">${ts(item)}</span>
      </div>
    </div>
  `).join('');

  // Lazy load images
  items.forEach((item, idx) => {
    if (!item.filename) return;
    fetch(`${API}/api/gallery/image/${item.filename}`)
      .then(r => r.json())
      .then(d => {
        const el = grid.children[idx];
        if (!el || !d.b64) return;
        const imgWrap = el.querySelector('div');
        imgWrap.innerHTML = '';
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${d.b64}`;
        img.style.cssText = 'width:100%;height:160px;object-fit:cover;display:block;transition:transform 0.3s';
        imgWrap.appendChild(img);
      })
      .catch(() => {});
  });
}

async function openGalleryItem(filename, model) {
  try {
    const r = await fetch(`${API}/api/gallery/image/${filename}`);
    const d = await r.json();
    if (d.b64) openLightbox(`data:image/png;base64,${d.b64}`, `${model.toUpperCase()} · ${filename}`);
  } catch(e) {}
}

/* ── Lightbox ────────────────────────────────────── */
function openLightbox(src, caption) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightboxCaption').textContent = caption || '';
  document.getElementById('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

/* ── Toast ───────────────────────────────────────── */
function showToast(msg, type = '') {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, 3500);
}
