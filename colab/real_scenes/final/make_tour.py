"""v3 tour: the trained room as a web tour like splattour.com, built on the official SuperSplat Viewer (PlayCanvas, MIT).
  scene.sog      the full model levelled (floor at y = 0), cropped to the room box (no blurry shell from the window view, no
                 floaters outside), SOG-compressed (~10x smaller than the .ply) by @playcanvas/splat-transform
  settings.json  start view inside the room at the height the phone was held, a guided walk along the real camera path (where the
                 model is at its best: it plays first), hotspots at evenly spaced spots of the walk, all in room coordinates
  collision.glb  room box + the room mesh's furniture (mesh_tsdf.py, simplified): walk mode stays inside, on the floor
Coordinates: the viewer turns every splat file 180 deg about z (3DGS convention), so the file gets F (R p + t) with F = diag(-1,-1,1)
while cameras/hotspots are written in room coordinates R p + t (measured with a test scene, see HANDOFF_V3.md section 9).
Two steps (the camera reader needs gsplat's examples + torch, the converter needs Node):
  python make_tour.py plan  <examples dir> <data dir (images/ + sparse/0)> <run out dir (full/, mesh/)> <tour dir>
  python make_tour.py build <run out dir> <tour dir> <viewer public dir> [splat-transform command]"""
import json, os, re, shutil, subprocess, sys, numpy as np
F = np.diag([-1., -1, 1])

# Floor plan minimap (live position dot, clickable spots) + strip of real-photo thumbnails; clicking goes to that spot.
OVERLAY = """<style>
  /* SuperSplat viewer's design system (its index.css tokens): Arial, panels of 50 % black + 20 px blur, 16 px corners,
     1 px white-8 % borders, 34 px icon buttons (filled icon + 40 % black outline), keycap chips; our accent stays #e8a96b */
  :root{--nm-text:#e0dcdd;--nm-light:#fff;--nm-dark:#aaa;--nm-acc:#e8a96b;--nm-wall:#f1e9d8;--nm-panel:rgba(0,0,0,.5);--nm-line:rgba(255,255,255,.08)}
  .nm-ui,.nm-ui *{box-sizing:border-box;font-family:Arial,sans-serif;-webkit-user-select:none;user-select:none}
  .nm-panel{background:var(--nm-panel);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid var(--nm-line);color:var(--nm-text)}
  .nm-btn{width:34px;height:34px;padding:0;margin:0;border:0;cursor:pointer;color:var(--nm-text);background:transparent;display:flex;align-items:center;justify-content:center}
  .nm-btn:hover{color:var(--nm-light)} .nm-btn svg{display:block}
  .nm-btn svg>g.s{fill:none;stroke:#000;stroke-width:2;stroke-linejoin:round;opacity:.4} .nm-btn svg>g.f{fill:currentColor;stroke:none}
  .nm-group{display:flex;background:rgba(0,0,0,.3);border-radius:6px}
  .nm-key{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;padding:0 5px;border-radius:4px;border:1px solid rgba(255,255,255,.14);border-bottom-color:rgba(255,255,255,.26);background:rgba(255,255,255,.08);color:var(--nm-light);font-size:11px}
  .nm-pill{height:34px;padding:0 20px;border:1px solid rgba(255,255,255,.1);border-radius:8px;font-size:13px;cursor:pointer;color:var(--nm-dark);background:rgba(255,255,255,.04);transition:background-color 250ms ease,color 250ms ease}
  .nm-pill:hover,.nm-pill.on{color:var(--nm-light);background:rgba(255,255,255,.1)}
  #nm-plan{position:fixed;left:max(16px,env(safe-area-inset-left));top:max(16px,env(safe-area-inset-top));width:min(30vw,260px);border-radius:16px;padding:8px;z-index:10}
  #nm-plan .wrap{position:relative;cursor:pointer;border-radius:10px;overflow:hidden}
  #nm-plan .bar{display:flex;justify-content:space-between;align-items:center;padding:4px 2px 0 6px;font-size:12px}
  .nm-svg{display:block;width:100%;height:auto;background:#141414}
  .nm-svg .wall{stroke:var(--nm-wall);stroke-linecap:square;fill:none}
  .nm-svg .spot{fill:var(--nm-acc);stroke:#fff;stroke-width:2;cursor:pointer} .nm-svg .spot:hover{fill:#ffc58a}
  .nm-svg .lbl{fill:#fff;font:bold 13px Arial,sans-serif;paint-order:stroke;stroke:rgba(0,0,0,.7);stroke-width:3px}
  .nm-svg .me{fill:var(--nm-acc);stroke:#fff;stroke-width:2} .nm-svg .cone{fill:var(--nm-acc);opacity:.5}
  #nm-strip{position:fixed;left:50%;transform:translateX(-50%);bottom:calc(max(16px,env(safe-area-inset-bottom)) + 72px);display:flex;gap:6px;z-index:10;max-width:calc(100vw - 32px);overflow-x:auto;padding:6px;border-radius:16px;scrollbar-width:none}
  #nm-strip button{border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:0;background:#141414;cursor:pointer;position:relative;flex:none;overflow:hidden;transition:border-color 250ms ease}
  #nm-strip button:hover{border-color:var(--nm-acc)}
  #nm-strip img{width:96px;height:58px;object-fit:cover;display:block}
  #nm-strip span{position:absolute;left:0;right:0;bottom:0;padding:10px 6px 3px;color:#fff;font-size:11px;text-align:left;background:linear-gradient(transparent,rgba(0,0,0,.7))}
  #nm-jump{position:fixed;right:max(16px,env(safe-area-inset-right));bottom:calc(max(16px,env(safe-area-inset-bottom)) + 112px);z-index:10;display:none;width:52px;height:52px;border-radius:50%}
  #nm-jump svg{width:26px;height:26px}
  body:has(.sse-settingsPanel:not(.sse-hidden)) #nm-jump{display:none!important}
  #nm-fp,#nm-help{position:fixed;inset:0;z-index:30;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.25);backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px)}
  #nm-fp .box{position:relative;width:min(92vw,980px);height:min(78vh,700px);border-radius:16px;overflow:hidden}
  #nm-fp .tog{position:absolute;left:12px;top:12px;z-index:2;display:flex;gap:4px}
  #nm-fp .tog .nm-pill{height:30px;padding:0 14px}
  #nm-fp .box.nm-panel,#nm-help .card.nm-panel{background:rgba(20,20,20,.86)}
  #nm-fp .x,#nm-help .x{position:absolute;right:10px;top:10px;z-index:2}
  #nm-fp .v2d,#nm-fp canvas{position:absolute;left:0;top:52px;right:0;bottom:36px;width:100%;height:calc(100% - 88px)} #nm-fp .v2d svg{width:100%;height:100%;background:none}
  #nm-fp .cap{position:absolute;left:0;right:0;bottom:12px;text-align:center;color:var(--nm-dark);font-size:12px}
  #nm-help .card{position:relative;width:340px;max-width:calc(100% - 32px);max-height:calc(100% - 32px);overflow-y:auto;padding:24px 20px 20px;border-radius:16px;display:flex;flex-direction:column;gap:14px;font-size:13px}
  #nm-help .title{font-size:16px;color:var(--nm-light);font-weight:bold} #nm-help .sub{color:var(--nm-dark);font-size:12px;margin-top:-8px}
  #nm-help h1{font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:1px;color:var(--nm-acc);padding-bottom:6px;margin:0}
  #nm-help .sec{padding-top:12px;border-top:1px solid rgba(255,255,255,.06)}
  #nm-help .row{display:flex;justify-content:space-between;align-items:center;gap:16px;min-height:28px;font-size:12px}
  #nm-help .keys{display:flex;gap:3px;align-items:center;color:var(--nm-dark);font-size:11px}
  #nm-help .go{align-self:stretch}
  #nm-find{position:fixed;left:50%;bottom:calc(max(16px,env(safe-area-inset-bottom)) + 156px);transform:translateX(-50%);width:min(80vw,340px);z-index:11;font:13px Arial,sans-serif;color:var(--nm-text);display:none}
  #nm-q{width:100%;padding:9px 12px;border-radius:8px;border:1px solid rgba(255,255,255,.1);background:var(--nm-panel);backdrop-filter:blur(20px);color:#fff;font:inherit;outline:none}
  #nm-hits{display:flex;flex-direction:column-reverse;gap:4px;margin-bottom:6px;max-height:50vh;overflow-y:auto}
  #nm-hits button,#nm-card{display:flex;align-items:center;gap:8px;background:rgba(0,0,0,.6);border:1px solid rgba(255,255,255,.1);border-radius:8px;color:var(--nm-text);font:inherit;padding:4px;cursor:pointer;text-align:left}
  #nm-hits img,#nm-card img{width:44px;height:44px;object-fit:cover;border-radius:5px;flex:none}
  #nm-card{position:fixed;z-index:11;display:none;max-width:260px;padding:6px 8px}
  #nm-card b{display:block} #nm-card small{color:var(--nm-dark)}
  @media (max-width:640px),(pointer:coarse){#nm-strip{display:none!important}#nm-plan{width:42vw;padding:6px}body.nm-walk #nm-jump{display:flex}
    #nm-find{bottom:calc(max(16px,env(safe-area-inset-bottom)) + 70px)}}
</style>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <symbol id="nmPlanIcon" viewBox="0 0 24 24"><path fill-rule="evenodd" d="M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zm1 2v14h6v-5h2v5h6v-6h-4v-2h4V5h-8v4h-2V5H5z"/></symbol>
  <symbol id="nmExpandIcon" viewBox="0 0 24 24"><path d="M14 3h7v7h-2V6.41l-4.29 4.3-1.42-1.42L17.59 5H14V3zM3 14h2v3.59l4.29-4.3 1.42 1.42L6.41 19H10v2H3v-7z"/></symbol>
  <symbol id="nmKeysIcon" viewBox="0 0 24 24"><path d="M20 5H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm-9 3h2v2h-2V8zm0 3h2v2h-2v-2zM8 8h2v2H8V8zm0 3h2v2H8v-2zm-1 2H5v-2h2v2zm0-3H5V8h2v2zm9 7H8v-2h8v2zm0-4h-2v-2h2v2zm0-3h-2V8h2v2zm3 3h-2v-2h2v2zm0-3h-2V8h2v2z"/></symbol>
  <symbol id="nmUpIcon" viewBox="0 0 24 24"><path d="M12 7.6l7 7-1.6 1.6-5.4-5.4-5.4 5.4L5 14.6z"/></symbol>
  <symbol id="nmCloseIcon" viewBox="0 0 24 24"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></symbol>
  <symbol id="nmJumpIcon" viewBox="0 0 24 24"><path d="M12 3l6.5 6.5h-4.25V15h-4.5V9.5H5.5L12 3zM5 18h14v2.5H5z"/></symbol>
</svg>
<div class="nm-ui">
<div id="nm-find"><div id="nm-hits"></div><input id="nm-q" type="search" placeholder="Search the room: sofa, lamp, bag ..." autocomplete="off"></div>
<div id="nm-card" class="nm-panel"></div>
<div id="nm-plan" class="nm-panel"><div class="wrap" id="nm-mini"></div><div class="bar"><span>Floor plan</span><span class="nm-group">
  <button class="nm-btn" id="nm-helpbtn" title="Controls (?)"><svg width="24" height="24"><g class="s"><use href="#nmKeysIcon"/></g><g class="f"><use href="#nmKeysIcon"/></g></svg></button>
  <button class="nm-btn" id="nm-exp" title="Open floor plan"><svg width="20" height="20" viewBox="0 0 24 24"><g class="s"><use href="#nmExpandIcon"/></g><g class="f"><use href="#nmExpandIcon"/></g></svg></button></span></div></div>
<div id="nm-strip" class="nm-panel"></div>
<button class="nm-btn nm-panel" id="nm-jump" title="Jump" aria-label="Jump"><svg viewBox="0 0 24 24"><g class="s"><use href="#nmJumpIcon"/></g><g class="f"><use href="#nmJumpIcon"/></g></svg></button>
<div id="nm-fp"><div class="box nm-panel"><button class="nm-btn x" id="nm-fpx" title="Close"><svg width="24" height="24"><g class="s"><use href="#nmCloseIcon"/></g><g class="f"><use href="#nmCloseIcon"/></g></svg></button>
  <div class="tog"><button class="nm-pill on" data-m="2d">2D</button><button class="nm-pill" data-m="3d">3D</button></div>
  <div class="v2d" id="nm-fp2d"></div><canvas id="nm-fp3d" style="display:none"></canvas><div class="cap" id="nm-fpcap">Click a point to go there</div></div></div>
<div id="nm-help"><div class="card nm-panel"><button class="nm-btn x" id="nm-hx" title="Close"><svg width="24" height="24"><g class="s"><use href="#nmCloseIcon"/></g><g class="f"><use href="#nmCloseIcon"/></g></svg></button>
  <div class="title">How to explore</div><div class="sub" id="nm-hsub"></div><div id="nm-rows"></div><button class="nm-pill go on" id="nm-hgo">Start exploring</button></div></div>
</div>
<script type="module">
  const o = await (await fetch('./overlay.json')).json(); const f = o.floor;
  let wj = null; try { wj = await (await fetch('./walls.json')).json(); } catch (e) {}
  const walls = (wj && wj.walls) || [], H = (wj && wj.height) || 2.5;
  const box = (wj && wj.box) || [f.x0, 0, f.z0, f.x0 + f.w / f.px, H, f.z0 + f.h / f.px];
  const go = (i) => { try { window.viewer.selectAnnotation(i); } catch (e) {} };
  const P = (x, z) => [(x - f.x0) * f.px, (z - f.z0) * f.px];                              // room metres -> floor-plan pixels
  const NS = 'http://www.w3.org/2000/svg', el = (t, a, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  // 2D plan: the top-down picture (dimmed), solid cream walls, spots; `big` adds labels
  function plan2d(big) {
    const sv = el('svg', { viewBox: `0 0 ${f.w} ${f.h}`, class: 'nm-svg', preserveAspectRatio: 'xMidYMid meet' });
    el('image', { href: './floorplan.png', x: 0, y: 0, width: f.w, height: f.h, opacity: 0.42 }, sv);
    walls.forEach(w => { const [a, b] = P(w[0], w[1]), [c, d] = P(w[2], w[3]); el('line', { x1: a, y1: b, x2: c, y2: d, class: 'wall', 'stroke-width': Math.max(4, 0.13 * f.px) }, sv); });
    const r = Math.max(7, 0.16 * f.px) * (big ? 1 : 1.25);
    o.spots.forEach((s, i) => { const [x, y] = P(s.x, s.z); const c = el('circle', { cx: x, cy: y, r, class: 'spot' }, sv); c.onclick = (e) => { e.stopPropagation(); go(i); closeFp(); };
      if (big) { const t = el('text', { x, y: y - r - 6, 'text-anchor': 'middle', class: 'lbl' }, sv); t.textContent = s.title; } });
    const cone = el('path', { class: 'cone' }, sv), me = el('circle', { r: r * 0.85, class: 'me' }, sv);
    return { sv, cone, me };
  }
  const mini = plan2d(false); document.getElementById('nm-mini').appendChild(mini.sv);
  const camState = () => { const app = window.app; if (!app) return null; const c = app.root.findComponents('camera')[0]; if (!c) return null;
    const p = c.entity.getPosition(), fw = c.entity.forward; return { x: p.x, y: p.y, z: p.z, fx: fw.x, fz: fw.z }; };
  const putMe = (m) => { const c = camState(); if (!c) return; const [x, y] = P(c.x, c.z), L = 0.9 * f.px, a = Math.atan2(c.fz, c.fx), s = 0.55;
    m.me.setAttribute('cx', x); m.me.setAttribute('cy', y);
    m.cone.setAttribute('d', `M${x},${y} L${x + L * Math.cos(a - s)},${y + L * Math.sin(a - s)} A${L},${L} 0 0 1 ${x + L * Math.cos(a + s)},${y + L * Math.sin(a + s)} Z`); };
  // spots bar at the bottom (laptops; hidden on phones, where the floor plan is the way to jump)
  o.spots.forEach((s, i) => { const b = document.createElement('button'); b.title = s.title; b.innerHTML = `<img src="./${s.thumb}" alt=""><span>${s.title}</span>`; b.onclick = () => go(i); document.getElementById('nm-strip').appendChild(b); });
  // full floor plan: 2D / 3D (solid walls, drag to rotate), click a point to enter
  const fp = document.getElementById('nm-fp'), v2 = document.getElementById('nm-fp2d'), cv = document.getElementById('nm-fp3d'), cap = document.getElementById('nm-fpcap');
  const big = plan2d(true); v2.appendChild(big.sv); let mode = '2d', yaw = -0.6, pitch = 0.95, img = new Image(); img.src = './floorplan.png';
  const openFp = () => { fp.style.display = 'flex'; setMode(mode); }; const closeFp = () => { fp.style.display = 'none'; };
  document.getElementById('nm-exp').onclick = (e) => { e.stopPropagation(); openFp(); }; document.getElementById('nm-mini').onclick = openFp; document.getElementById('nm-fpx').onclick = closeFp;
  fp.onclick = (e) => { if (e.target === fp) closeFp(); };
  const setMode = (m) => { mode = m; fp.querySelectorAll('.tog button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
    v2.style.display = m === '2d' ? 'block' : 'none'; cv.style.display = m === '3d' ? 'block' : 'none';
    cap.textContent = m === '2d' ? 'Click a point to go there' : 'Drag to rotate · click a point to go there'; if (m === '3d') draw3d(); };
  fp.querySelectorAll('.tog button').forEach(b => b.onclick = () => setMode(b.dataset.m));
  const cx = (box[0] + box[3]) / 2, cz = (box[2] + box[5]) / 2, span = Math.max(box[3] - box[0], box[5] - box[2]), WH = Math.min(H * 0.82, 2.4), T = 0.12;
  let dots3d = [];
  function draw3d() {
    const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1; cv.width = r.width * dpr; cv.height = r.height * dpr;
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    const S = Math.min(r.width, r.height) * 0.78 / span, cy_ = Math.cos(yaw), sy = Math.sin(yaw), sp = Math.sin(pitch), cp = Math.cos(pitch);
    const pr = (x, y, z) => { const u = (x - cx) * cy_ - (z - cz) * sy, v = (x - cx) * sy + (z - cz) * cy_;
      return [r.width / 2 + u * S, r.height / 2 + 30 + (v * sp - y * cp) * S, v * cp + y * sp]; };
    // floor: the top-down picture, an affine image of the floor plane
    const o0 = pr(f.x0, 0, f.z0), ox = pr(f.x0 + 1 / f.px, 0, f.z0), oz = pr(f.x0, 0, f.z0 + 1 / f.px);
    g.save(); g.globalAlpha = 0.5; g.setTransform(dpr * (ox[0] - o0[0]), dpr * (ox[1] - o0[1]), dpr * (oz[0] - o0[0]), dpr * (oz[1] - o0[1]), dpr * o0[0], dpr * o0[1]);
    if (img.complete) g.drawImage(img, 0, 0); g.restore(); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    // solid walls: prisms, faces drawn back to front
    const faces = [];
    walls.forEach(w => { const dx = w[2] - w[0], dz = w[3] - w[1], L = Math.hypot(dx, dz) || 1, nx = -dz / L * T / 2, nz = dx / L * T / 2, ex = dx / L * T / 2, ez = dz / L * T / 2;
      const c = [[w[0] - ex + nx, w[1] - ez + nz], [w[2] + ex + nx, w[3] + ez + nz], [w[2] + ex - nx, w[3] + ez - nz], [w[0] - ex - nx, w[1] - ez - nz]];
      for (let k = 0; k < 4; k++) { const a = c[k], b = c[(k + 1) % 4]; const q = [pr(a[0], 0, a[1]), pr(b[0], 0, b[1]), pr(b[0], WH, b[1]), pr(a[0], WH, a[1])];
        const mx = (a[0] + b[0]) / 2 - (w[0] + w[2]) / 2, mz = (a[1] + b[1]) / 2 - (w[1] + w[3]) / 2, lit = 0.72 + 0.28 * Math.abs((mx * cy_ - mz * sy) / (Math.hypot(mx, mz) || 1));
        faces.push({ q, d: q.reduce((s, p) => s + p[2], 0) / 4, col: `rgb(${Math.round(225 * lit)},${Math.round(217 * lit)},${Math.round(200 * lit)})` }); }
      const top = c.map(p => pr(p[0], WH, p[1])); faces.push({ q: top, d: top.reduce((s, p) => s + p[2], 0) / 4 + 0.01, col: '#f4ede0' }); });
    faces.sort((a, b) => a.d - b.d).forEach(F => { g.beginPath(); F.q.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); g.fillStyle = F.col; g.fill(); g.strokeStyle = '#00000022'; g.stroke(); });
    dots3d = o.spots.map((s, i) => { const p = pr(s.x, 1.0, s.z); g.beginPath(); g.arc(p[0], p[1], 9, 0, 7); g.fillStyle = '#e8a96b'; g.fill(); g.lineWidth = 2; g.strokeStyle = '#fff'; g.stroke();
      g.font = 'bold 12px Arial, sans-serif'; g.textAlign = 'center'; g.lineWidth = 3; g.strokeStyle = '#000b'; g.strokeText(s.title, p[0], p[1] - 14); g.fillStyle = '#fff'; g.fillText(s.title, p[0], p[1] - 14);
      return [p[0], p[1], i]; });
  }
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, yaw, pitch, 0]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!drag) return; const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag[4] = Math.max(drag[4], Math.hypot(dx, dy));
    yaw = drag[2] + dx * 0.008; pitch = Math.max(0.45, Math.min(1.45, drag[3] - dy * 0.006)); draw3d(); });
  cv.addEventListener('pointerup', e => { const moved = drag && drag[4] > 5; drag = null; if (moved) return; const r = cv.getBoundingClientRect();
    const hit = dots3d.find(d => Math.hypot(d[0] - (e.clientX - r.left), d[1] - (e.clientY - r.top)) < 16); if (hit) { go(hit[2]); closeFp(); } });
  addEventListener('resize', () => { if (fp.style.display === 'flex' && mode === '3d') draw3d(); });
  addEventListener('keydown', e => { if (e.key === 'Escape') { closeFp(); closeHelp(); } }, true);
  setInterval(() => { putMe(mini); if (fp.style.display === 'flex') putMe(big); }, 150);
  // walk mode stands on the floor inside the room: entered from outside (orbit / fly), it first goes to the nearest spot
  const inRoom = (c) => c && c.x > box[0] + 0.15 && c.x < box[3] - 0.15 && c.z > box[2] + 0.15 && c.z < box[5] - 0.15 && c.y > 0.4 && c.y < Math.min(box[4] - 0.2, 2.4);
  let fixing = false;
  const hookWalk = () => { const v = window.viewer; if (!v || !v.events) return setTimeout(hookWalk, 300);
    v.events.on('cameraMode:changed', (m) => { if (m !== 'walk' || fixing) return; const c = camState(); if (inRoom(c)) return;
      fixing = true; v.state.cameraMode = 'orbit'; let best = 0, bd = 1e9; o.spots.forEach((s, i) => { const d = c ? Math.hypot(s.x - c.x, s.z - c.z) : i; if (d < bd) { bd = d; best = i; } });
      go(best); setTimeout(() => { try { v.toggleWalk(); } catch (e) {} fixing = false; }, 1600); }); };
  hookWalk();
  // walk mode. The viewer's walk controller is exposed as window.nammaWalk by make_tour.patch_js. Eye height (owner, 31 Oct:
  // inside the gear settings, not on screen): a group at the top of the viewer's own Settings panel, in its row/button style,
  // and Q / E; eye = camera above the floor (controller eyeHeight + its 0.2 m hover). Jump button on touch screens, walk only.
  let eye = 1.5, hv = null; try { eye = parseFloat(localStorage.getItem('nammaEye')) || 1.5; } catch (e) {}
  const setEye = (h) => { eye = Math.round(Math.max(0.9, Math.min(2.2, h)) * 10) / 10; if (hv) hv.textContent = eye.toFixed(1) + ' m';
    const w = window.nammaWalk; if (w) w.eyeHeight = eye - 0.2; try { localStorage.setItem('nammaEye', String(eye)); } catch (e) {} };
  const eyeRow = () => { const sp = document.querySelector('.sse-settingsPanel'); if (!sp || !window.nammaWalk) return setTimeout(eyeRow, 400);
    if (document.getElementById('nm-eyegrp')) return;
    const g = document.createElement('div'); g.className = 'sse-settingsGroup'; g.id = 'nm-eyegrp';
    g.innerHTML = '<div class="sse-settingsRow"><div style="cursor:default">Eye height (walk)</div><div id="nm-hv" style="cursor:default;font-size:12px"></div></div>'
      + '<div class="sse-settingsRow"><button id="nm-down" title="Lower (Q)">▼ Lower</button><button id="nm-up" title="Higher (E)">▲ Higher</button></div>';
    sp.insertBefore(g, sp.firstChild); hv = document.getElementById('nm-hv'); setEye(eye);
    document.getElementById('nm-up').onclick = (e) => { e.stopPropagation(); setEye(eye + 0.1); };
    document.getElementById('nm-down').onclick = (e) => { e.stopPropagation(); setEye(eye - 0.1); }; };
  eyeRow();
  // Jump height (owner, 31 Oct: "on phone it sometimes jumps little"): while moving, the capsule brushes walls and furniture, and the
  // viewer reads any downward push-out as a ceiling and cancels the jump. Same check, but a ceiling only when the push is mostly
  // downward: a wall brushed while moving keeps the full jump (the jump itself, gravity and speed are the viewer's own).
  const fixJump = (w) => { if (!w || w.__nmJump || typeof w._checkCollision !== 'function' || !w._velocity) return; w.__nmJump = 1;
    const o = { x: 0, y: 0, z: 0 };
    w._checkCollision = function (pos, disp) { const c = this.collision; if (!c) return;
      const center = pos.y - this.eyeHeight + this.capsuleHeight * 0.5, half = this.capsuleHeight * 0.5 - this.capsuleRadius;
      if (!c.queryCapsule(pos.x, center, pos.z, half, this.capsuleRadius, o)) return;
      disp.set(o.x, o.y, o.z); pos.add(disp);
      if (o.y < 0 && this._velocity.y > 0 && -o.y > Math.hypot(o.x, o.z)) this._velocity.y = 0;
      if (!this._grounded && o.y > 0 && this._velocity.y < 0) { this._velocity.y = 0; this._grounded = true; } }; };
  document.getElementById('nm-jump').onpointerdown = (e) => { e.preventDefault(); e.stopPropagation(); const w = window.nammaWalk; if (w && w._pendingMove) w._pendingMove[1] = 1; };
  const walkUi = (m) => { document.body.classList.toggle('nm-walk', m === 'walk' && !!window.nammaWalk); if (m === 'walk') { fixJump(window.nammaWalk); setEye(eye); } };
  const hookUi = () => { const v = window.viewer; if (!v || !v.events) return setTimeout(hookUi, 300); v.events.on('cameraMode:changed', walkUi); walkUi(v.state.cameraMode); };
  hookUi();
  addEventListener('keydown', (e) => { const v = window.viewer; if (!v || v.state.cameraMode !== 'walk' || /INPUT|TEXTAREA/.test((e.target || {}).tagName || '')) return;
    if (e.code === 'KeyE') setEye(eye + 0.1); else if (e.code === 'KeyQ') setEye(eye - 0.1); });
  // every visit (owner, 31 Oct), once the room is in: how to move (the viewer's own controls panel is closed for new visitors by the head script, so one card shows)
  const touch = matchMedia('(pointer:coarse)').matches || /Mobi|Android|iPhone|iPad/.test(navigator.userAgent);
  const K = (...k) => k.map(x => `<span class="nm-key">${x}</span>`).join('');
  const sec = (t, rows) => `<div class="sec"><h1>${t}</h1>${rows.map(r => `<div class="row"><span>${r[0]}</span><span class="keys">${r[1]}</span></div>`).join('')}</div>`;
  document.getElementById('nm-hsub').textContent = touch ? 'Walk through the real place with your fingers.' : 'Walk through the real place like in a game.';
  document.getElementById('nm-rows').innerHTML = touch
    ? sec('Walk', [['Move', 'left joystick'], ['Look around', 'drag with one finger'], ['Jump', 'jump button (right), or tap'], ['Eye height', 'gear (Settings)']])
      + sec('Go to a place', [['Rooms and spots', 'tap the floor plan'], ['Walk mode', 'person button (bottom right)']])
    : sec('Walk', [['Move', K('W', 'A', 'S', 'D')], ['Look around', 'mouse'], ['Jump', K('Space')], ['Eye height', K('Q') + ' / ' + K('E') + ' or gear'], ['Run / slow', K('Shift') + ' / ' + K('Ctrl')], ['Walk to a point', 'click']])
      + sec('Go to a place', [['Rooms and spots', 'photo bar or floor plan'], ['Walk mode', 'person button (bottom right)'], ['All shortcuts', K('Shift', 'H')]]);
  const help = document.getElementById('nm-help'), openHelp = () => { help.style.display = 'flex'; },
    closeHelp = () => { help.style.display = 'none'; };
  document.getElementById('nm-hx').onclick = closeHelp; document.getElementById('nm-hgo').onclick = closeHelp; help.onclick = (e) => { if (e.target === help) closeHelp(); };
  document.getElementById('nm-helpbtn').onclick = (e) => { e.stopPropagation(); openHelp(); };
  { const t0 = Date.now(), k = setInterval(() => { const v = window.viewer; if ((v && v.state && v.state.loaded) || Date.now() - t0 > 120000) { clearInterval(k); openHelp(); } }, 300); }
  // objects (objects3d.py): search by name / description, and click an object in the 3D view to see what it is
  const obs = o.objects || [], find = document.getElementById('nm-find'), q = document.getElementById('nm-q'), hits = document.getElementById('nm-hits'), card = document.getElementById('nm-card');
  const row = (x) => `${x.thumb ? `<img src="./${x.thumb}" alt="">` : ''}<span><b>${x.label}</b><br><small>${x.size}</small></span>`;
  const match = (s) => { const w = s.toLowerCase().split(' ').filter(Boolean); return obs.filter(x => w.every(t => x.words.includes(t) || x.words.split(' ').some(v => v.startsWith(t.slice(0, Math.max(3, t.length - 1)))))); };
  const show = () => { hits.innerHTML = ''; (q.value.trim() ? match(q.value) : obs).slice(0, 8).forEach(x => {
    const b = document.createElement('button'); b.innerHTML = row(x); b.onclick = () => { go(x.i); hits.innerHTML = ''; q.blur(); }; hits.appendChild(b); }); };
  if (obs.length) {
    find.style.display = 'block'; q.oninput = show; q.onfocus = show; q.onblur = () => setTimeout(() => { if (document.activeElement !== q) hits.innerHTML = ''; }, 200);
    ['keydown', 'keyup', 'keypress'].forEach(t => q.addEventListener(t, e => { e.stopPropagation(); if (t === 'keydown' && e.key === 'Enter') { const m = match(q.value); if (m.length) { go(m[0].i); hits.innerHTML = ''; q.blur(); } } }));
    let down = null;
    window.addEventListener('pointerdown', e => { down = e.target.tagName === 'CANVAS' ? [e.clientX, e.clientY, Date.now()] : null; });
    window.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6 || Date.now() - down[2] > 400) return;
      const cam = window.app && window.app.root.findComponents('camera')[0]; if (!cam) return;
      const r = e.target.getBoundingClientRect(), a = cam.entity.getPosition(), p = cam.screenToWorld(e.clientX - r.left, e.clientY - r.top, 10);
      const d = [p.x - a.x, p.y - a.y, p.z - a.z], o_ = [a.x, a.y, a.z]; let best = null, bt = 1e9;
      obs.forEach(x => { let t0 = 0, t1 = 1e9;                                                  // ray vs the object's box (+2 cm)
        for (let k = 0; k < 3; k++) { const inv = 1 / (d[k] || 1e-9); let u = (x.lo[k] - 0.02 - o_[k]) * inv, v = (x.hi[k] + 0.02 - o_[k]) * inv;
          if (u > v) [u, v] = [v, u]; t0 = Math.max(t0, u); t1 = Math.min(t1, v); }
        if (t0 <= t1 && t0 < bt) { bt = t0; best = x; } });
      if (!best) { card.style.display = 'none'; return; }
      card.innerHTML = row(best); card.style.display = 'flex'; card.onclick = () => { go(best.i); card.style.display = 'none'; };
      card.style.left = Math.min(e.clientX + 12, innerWidth - 270) + 'px'; card.style.top = Math.min(e.clientY + 12, innerHeight - 70) + 'px';
    });
  }
</script>"""

def room_collision(pts, h, path):
    """Walk-mode collision when there is no mesh yet: a floor slab at y = 0 and four walls along the room's main horizontal axes
    (2nd-98th percentile of the SfM points), 3 m high. Room coordinates in metres."""
    import trimesh
    flat = pts[:, [0, 2]]; m0 = flat.mean(0); ax = np.linalg.svd(flat - m0, full_matrices=False)[2]
    if np.linalg.det(ax) < 0: ax[1] *= -1
    q = (flat - m0) @ ax.T; lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0); pad = 0.02 * (hi - lo); lo, hi = lo - pad, hi + pad
    def box(cu, cv, cy, eu, ev, ey):
        T = np.eye(4); T[[0, 2], 0] = ax[0]; T[[0, 2], 2] = ax[1]; T[0, 3], T[2, 3] = m0 + cu * ax[0] + cv * ax[1]; T[1, 3] = cy
        return trimesh.creation.box(extents=(eu, ey, ev), transform=T)
    (u0, v0), (u1, v1) = lo, hi; cu, cv, wu, wv, th, hh = (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0, 0.1, 3.0
    parts = [box(cu, cv, -0.05, wu + 1, wv + 1, 0.1),                                      # floor (top at y = 0)
             box(u0 - th / 2, cv, hh / 2, th, wv + 2 * th, hh), box(u1 + th / 2, cv, hh / 2, th, wv + 2 * th, hh),
             box(cu, v0 - th / 2, hh / 2, wu + 2 * th, th, hh), box(cu, v1 + th / 2, hh / 2, wu + 2 * th, th, hh)]
    trimesh.util.concatenate(parts).export(path)

def smooth(x, r):
    """Gaussian moving average along the walk (handheld jitter out), edges kept."""
    k = np.exp(-0.5 * (np.arange(-2 * r, 2 * r + 1) / max(r, 1e-6)) ** 2); k /= k.sum()
    xp = np.pad(x, ((2 * r, 2 * r), (0, 0)), mode='edge'); return np.stack([np.convolve(xp[:, j], k, 'valid') for j in range(x.shape[1])], 1)

def euler_xyz(M):
    """Extrinsic x-y-z Euler angles in degrees of a rotation matrix (splat-transform -r)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(M).as_euler('xyz', degrees=True)

def walls2d(Q, a, lo, hi, h, res=0.04):
    """Straight walls for the floor plan (2D lines, 3D solid walls in the tour): where splats stack up above furniture height
    (0.55-0.92 x room height, so beds, sofas and tables drop out), Hough lines along the room's two main directions, merged per
    wall. Metres in room coordinates, [[x0, z0, x1, z1], ...]; falls back to the room box when fewer than 3 walls are found."""
    import cv2
    from scipy import ndimage
    box = [[lo[0], lo[2], hi[0], lo[2]], [hi[0], lo[2], hi[0], hi[2]], [hi[0], hi[2], lo[0], hi[2]], [lo[0], hi[2], lo[0], lo[2]]]
    m = (Q[:, 1] > 0.55 * h) & (Q[:, 1] < 0.92 * h) & (a > 0.2) & np.all((Q[:, [0, 2]] >= lo[[0, 2]]) & (Q[:, [0, 2]] <= hi[[0, 2]]), 1)
    W, H = int((hi[0] - lo[0]) / res) + 1, int((hi[2] - lo[2]) / res) + 1
    if m.sum() < 1000: return [[round(float(v), 3) for v in w] for w in box], 'box'
    g, _, _ = np.histogram2d(((Q[m, 2] - lo[2]) / res).astype(int), ((Q[m, 0] - lo[0]) / res).astype(int), bins=(H, W), range=((0, H), (0, W)), weights=a[m])
    g = ndimage.gaussian_filter(g, 1.0); nz = g[g > 0]; mask = g > max(0.1 * np.percentile(nz, 99), 4 * np.median(nz))   # above floaters
    mask = ndimage.binary_closing(mask, iterations=2); lab, n = ndimage.label(mask)
    if n: mask &= np.isin(lab, 1 + np.flatnonzero(ndimage.sum(mask, lab, range(1, n + 1)) >= 0.5 / res))
    L = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 360, int(0.4 / res), minLineLength=int(0.6 / res), maxLineGap=int(0.35 / res))
    if L is None: return [[round(float(v), 3) for v in w] for w in box], 'box'
    L = L.reshape(-1, 4).astype(float); d = L[:, 2:] - L[:, :2]; ln = np.hypot(d[:, 0], d[:, 1]); ang = np.arctan2(d[:, 1], d[:, 0]) % (np.pi / 2)
    th = np.angle(np.sum(ln * np.exp(4j * ang))) / 4                                         # the room's main direction (mod 90 deg)
    walls = []
    for k in range(2):
        u = np.array([np.cos(th + k * np.pi / 2), np.sin(th + k * np.pi / 2)]); v = np.array([-u[1], u[0]])
        off = (np.arctan2(d[:, 1], d[:, 0]) - (th + k * np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2
        sel = np.abs(off) < np.radians(6)
        if not sel.any(): continue
        c = (L[sel, :2] + L[sel, 2:]) / 2; vv = c @ v; a0, a1 = L[sel, :2] @ u, L[sel, 2:] @ u; lens = ln[sel]
        order = np.argsort(vv); groups, cur = [], [order[0]]
        for i in order[1:]:                                                                   # lines within 12 cm across = one wall
            if vv[i] - vv[cur[-1]] < 0.12 / res: cur.append(i)
            else: groups.append(cur); cur = [i]
        groups.append(cur)
        for gi in groups:
            off_v = float(np.average(vv[gi], weights=lens[gi])); iv = sorted((min(a0[i], a1[i]), max(a0[i], a1[i])) for i in gi)
            s0, s1 = iv[0]
            for x0, x1 in iv[1:] + [(1e18, 1e18)]:                                            # merge pieces with gaps < 40 cm
                if x0 - s1 < 0.4 / res: s1 = max(s1, x1); continue
                p0, p1 = s0 * u + off_v * v, s1 * u + off_v * v
                t = np.linspace(0, 1, max(2, int(s1 - s0))); pr = (p0[None] + t[:, None] * (p1 - p0)[None]).round().astype(int)
                near = ndimage.binary_dilation(mask, iterations=2)[np.clip(pr[:, 1], 0, H - 1), np.clip(pr[:, 0], 0, W - 1)]
                if (s1 - s0) * res >= 0.5 and near.mean() >= 0.6:                             # a real wall: mostly on wall pixels
                    walls.append([round(float(lo[0] + p0[0] * res), 3), round(float(lo[2] + p0[1] * res), 3),
                                  round(float(lo[0] + p1[0] * res), 3), round(float(lo[2] + p1[1] * res), 3)])
                s0, s1 = x0, x1
    return (walls, 'hough') if len(walls) >= 3 else ([[round(float(v), 3) for v in w] for w in box], 'box')


def floorplan(run, tour, M=None):
    """Floor plan like splattour's: the model seen from straight above with the ceiling and wall tops cut off (each pixel = the
    highest splat below the cut, in its own colour, holes filled from the nearest splat), walls drawn dark on top (splat density
    between knee and head height). Needs only <run>/full/model.ply, the level matrix and <tour>/plan.json (scale, room box), so
    it can be redone without a GPU: make_tour.py floorplan <run> <tour>. Returns (width, height, pixels per metre)."""
    from plyfile import PlyData
    from PIL import Image
    from scipy import ndimage
    pl = json.load(open(f'{tour}/plan.json')); s = pl['scale']; lo, hi = np.array(pl['box'][:3]), np.array(pl['box'][3:]); h = hi[1] / 1.05
    v = PlyData.read(f'{run}/full/model.ply')['vertex'].data
    if M is None and os.path.exists(f'{run}/full/level.json'): M = np.array(json.load(open(f'{run}/full/level.json'))['matrix'])
    if M is None:                                                                         # older runs: same matrix plan() used
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import to_splat
        M = to_splat.level_matrix(f'{run}/full/nav.json', np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64))
    Q = (np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64) @ M[:3, :3].T + M[:3, 3]) * s
    a = 1 / (1 + np.exp(-v['opacity'].astype(np.float64))); size = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).max(1)) * s
    rgb = np.clip(0.5 + 0.28209479 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1), 0, 1)
    inside = np.all((Q[:, [0, 2]] >= lo[[0, 2]]) & (Q[:, [0, 2]] <= hi[[0, 2]]), 1)
    span = hi[[0, 2]] - lo[[0, 2]]; px = 900 / span.max(); Wp, Hp = (span * px).astype(int) + 1
    out_px = px; px = min(px, 1 / max(float(np.median(size)), 1e-3)); Wp, Hp = (span * px).astype(int) + 1   # ~1 splat size per pixel
    col = np.clip(((Q[:, 0] - lo[0]) * px).astype(int), 0, Wp - 1); row = np.clip(((Q[:, 2] - lo[2]) * px).astype(int), 0, Hp - 1)
    # photo: every splat under the cut (0.75 x room height: wall tops show as lines, no ceiling) blended from the top down by its
    # opacity, like a camera looking straight down (MCMC models are mostly semi-transparent splats: "solid only" left gaps)
    below = inside & (Q[:, 1] < 0.75 * h) & (Q[:, 1] > -0.05 * h); m = below & (a > 0.02) & (size < 0.3)
    print(f'floor plan: {len(a)} splats, {inside.sum()} inside the room, {below.sum()} under the cut, {m.sum()} drawn; '
          f'opacity median {np.median(a):.2f}, size median {np.median(size) * 100:.1f} cm', flush=True)
    sig = size[m] * px; pid, hh, aa, cc = [], [], [], []                                    # each splat over its own footprint
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            d2 = dy * dy + dx * dx; k = (2.25 * sig ** 2 >= d2) if d2 else np.ones(len(sig), bool)
            r, c = row[m][k] + dy, col[m][k] + dx; ok = (r >= 0) & (r < Hp) & (c >= 0) & (c < Wp)
            pid.append((r * Wp + c)[ok]); hh.append(Q[m, 1][k][ok]); cc.append(rgb[m][k][ok])
            aa.append((a[m][k] * np.exp(-0.5 * d2 / np.maximum(sig[k], 0.5) ** 2))[ok])
    pid, hh, cc = np.concatenate(pid), np.concatenate(hh), np.concatenate(cc); aa = np.clip(np.concatenate(aa), 0, 0.99)
    o = np.lexsort((-hh, pid)); pid, aa, cc = pid[o], aa[o], cc[o]                          # per pixel, top first
    lt = np.log1p(-aa); ex = np.cumsum(lt) - lt                                              # log transmittance before each splat
    st = np.r_[0, np.flatnonzero(np.diff(pid)) + 1]; ex -= np.repeat(ex[st], np.diff(np.r_[st, len(pid)]))
    wgt = aa * np.exp(ex); n = Hp * Wp
    alpha = np.bincount(pid, wgt, n); img = np.stack([np.bincount(pid, wgt * cc[:, k], n) for k in range(3)], -1)
    img = (img / np.maximum(alpha, 1e-6)[:, None]).reshape(Hp, Wp, 3); alpha = alpha.reshape(Hp, Wp)
    has = alpha > 0.2
    dist, (ii, jj) = ndimage.distance_transform_edt(~has, return_indices=True)
    img = img[ii, jj] * (dist <= 4)[..., None]                                              # fill gaps up to ~3 cm, rest stays black
    # walls: opacity-weighted density between knee and head height (vertical surfaces stack up there)
    w = inside & (Q[:, 1] > 0.15 * h) & (Q[:, 1] < 0.85 * h) & (a > 0.3)
    g, _, _ = np.histogram2d(row[w], col[w], bins=(Hp, Wp), range=((0, Hp), (0, Wp)), weights=a[w])
    g = np.log1p(ndimage.gaussian_filter(g, 1.0)); g = np.clip(g / max(np.percentile(g[g > 0], 99), 1e-6), 0, 1) if (g > 0).any() else g
    img = img * (1 - 0.75 * np.clip((g - 0.35) / 0.4, 0, 1))[..., None]
    W2, H2 = (span * out_px).astype(int) + 1
    Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).resize((int(W2), int(H2)), Image.LANCZOS).save(f'{tour}/floorplan.png')
    walls, how = walls2d(Q, a, lo, hi, h)                                                  # 2D lines / 3D solid walls in the tour
    json.dump(dict(walls=walls, height=round(float(h), 2), box=[round(float(x), 3) for x in list(lo) + list(hi)], source=how),
              open(f'{tour}/walls.json', 'w'))
    print(f'floor plan walls: {len(walls)} ({how})', flush=True)
    return int(W2), int(H2), float(out_px)


def colmap_cameras(sparse, examples):
    """gsplat Parser(normalize=True)'s camtoworlds (frames in name order) and points from a COLMAP model alone: no frames, no
    pycolmap. Uses gsplat's own datasets/normalize.py from <examples>, the same steps as datasets/colmap.py (v1.5.3)."""
    import importlib.util
    from thin_model import read_model
    sp = importlib.util.spec_from_file_location('gs_normalize', f'{examples}/datasets/normalize.py'); N = importlib.util.module_from_spec(sp); sp.loader.exec_module(N)
    _, ims, pts = read_model(sparse); C = []
    for i in sorted(ims, key=lambda i: ims[i]['name']):
        w, x, y, z, *tv = ims[i]['qt']; M = np.eye(4); M[:3, 3] = tv
        M[:3, :3] = [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y], [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]]
        C.append(M)
    C = np.linalg.inv(np.stack(C)); P = np.array([p['xyz'] for p in pts.values()], np.float32)
    T1 = N.similarity_from_cameras(C); C = N.transform_cameras(T1, C); P = N.transform_points(T1, P)
    T2 = N.align_principal_axes(P); C = N.transform_cameras(T2, C); P = N.transform_points(T2, P)
    if np.median(P[:, 2]) > np.mean(P[:, 2]):
        T3 = np.diag([1.0, -1.0, -1.0, 1.0]); C = N.transform_cameras(T3, C); P = N.transform_points(T3, P)
    return C, P


def respot(run, tour, examples):
    """Spots of a built tour moved to the exact filmed camera poses (their thumbnails), and the guided walk smoothed over at
    most 6 frames (owner, 1 Nov, house: spots 3-5 sat ~1 m off the walk, inside furniture). plan() smoothed the path over
    len/60 frames: fine for 411 frames (r 6), but 16 for the house's 982, which cut corners at doors. Needs <run>/sparse/0,
    <run>/full/level.json and the tour's plan.json / settings.json / overlay.json; checks first that the cameras rebuilt here
    reproduce the tour's own walk track (same normalisation as the trainer) before changing anything."""
    C, _ = colmap_cameras(f'{run}/sparse/0', examples)
    M = np.array(json.load(open(f'{run}/full/level.json'))['matrix']); R, t = M[:3, :3], M[:3, 3]
    pl = json.load(open(f'{tour}/plan.json')); sc = pl['scale']; lo, hi = np.array(pl['box'][:3]), np.array(pl['box'][3:])
    st = json.load(open(f'{tour}/settings.json')); ov = json.load(open(f'{tour}/overlay.json'))
    cam = (C[:, :3, 3] @ R.T + t) * sc; fwd = C[:, :3, 2] @ R.T; fn = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    tr = st['animTracks'][0]; fov = tr['keyframes']['values']['fov'][0]; n = len(cam)
    keys = np.unique(np.linspace(0, n - 1, min(n, 120)).astype(int)); old = np.array(tr['keyframes']['values']['position']).reshape(-1, 3)
    r0 = max(2, n // 60); err = float(np.abs(smooth(cam, r0)[keys] - old).max())                 # the track as plan() made it
    assert len(keys) == len(old) and err < 0.01, f'cameras do not match the tour (max {err:.3f} m): not changed'
    look = 0.25 * float(np.linalg.norm(hi[[0, 2]] - lo[[0, 2]])); r = max(2, min(6, n // 60))
    cs = smooth(cam, r); fs = smooth(fwd, r); fs /= np.linalg.norm(fs, axis=1, keepdims=True)
    P = lambda a, b, i: dict(position=[round(float(x), 4) for x in a[i]], target=[round(float(x), 4) for x in a[i] + look * b[i]], fov=round(fov, 1))
    tr['keyframes']['values']['position'] = [float(x) for k in keys for x in P(cs, fs, k)['position']]
    tr['keyframes']['values']['target'] = [float(x) for k in keys for x in P(cs, fs, k)['target']]
    spots = np.linspace(0, n - 1, 8).astype(int)[1:-1]; moved = []
    for k, i in enumerate(spots):
        a = st['annotations'][k]; e = P(cam, fn, i); moved.append(round(float(np.linalg.norm(np.array(a['camera']['initial']['position']) - cam[i])), 2))
        a['camera']['initial'] = e; a['position'] = e['target']; ov['spots'][k].update(x=float(cam[i][0]), z=float(cam[i][2]))
    json.dump(st, open(f'{tour}/settings.json', 'w')); json.dump(ov, open(f'{tour}/overlay.json', 'w'))
    print('respot: track check', round(err, 4), 'm; spots moved (m):', moved, '; walk smoothing', r0, '->', r, flush=True)
    return dict(check_m=err, moved_m=moved, smoothing=[r0, r])


def plan(examples, data, run, tour):
    sys.path.insert(0, examples); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets.colmap import Parser
    from plyfile import PlyData
    import to_splat
    os.makedirs(tour, exist_ok=True)
    p = Parser(data, factor=1, normalize=True, test_every=8); C = p.camtoworlds; nav = f'{run}/full/nav.json'
    if os.path.exists(f'{run}/full/level.json'): M = np.array(json.load(open(f'{run}/full/level.json'))['matrix'])
    else:
        v = PlyData.read(f'{run}/full/model.ply')['vertex'].data; M = to_splat.level_matrix(nav, np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64))
    R, t = M[:3, :3], M[:3, 3]
    # metres: SfM has no scale; the phone was held ~1.35 m above the floor, so scale the median camera height to that. Then the
    # viewer's walk mode (eyes at 1.3 m) walks exactly where the video was filmed, where the model is at its best
    s = float(1.35 / max(np.median(C[:, :3, 3] @ R[1] + t[1]), 1e-6))
    L = lambda x: (x @ R.T + t) * s                                                        # model world -> room coordinates, metres
    cam = L(C[:, :3, 3]); fwd = C[:, :3, 2] @ R.T; pts = L(p.points)
    # room box: SfM points, 3rd-97th percentile horizontally (+12 %), floor - 5 % to the 99th percentile height (+5 %)
    lo, hi = np.percentile(pts, 3, 0), np.percentile(pts, 97, 0); pad = 0.12 * (hi - lo); lo, hi = lo - pad, hi + pad
    h = np.percentile(pts[:, 1], 99); lo[1], hi[1] = -0.05 * h, 1.05 * h
    E = euler_xyz(R @ F)                                                                   # file' = F E F p + F t  ->  E = R F
    json.dump(dict(rotate=E.tolist(), translate=t.tolist(), scale=s, box=lo.tolist() + hi.tolist()), open(f'{tour}/plan.json', 'w'))
    room_collision(pts, h, f'{tour}/room_collision.glb')
    # cameras in room coordinates, in walk order (Parser sorts the frame names = video order)
    K = p.Ks_dict[p.camera_ids[0]]; Wd, Hd = p.imsize_dict[p.camera_ids[0]]; fov = float(np.clip(np.degrees(2 * np.arctan(Hd / 2 / K[1, 1])), 10, 120))
    r = max(2, min(6, len(cam) // 60)); cs, fs = smooth(cam, r), smooth(fwd, r); fs /= np.linalg.norm(fs, axis=1, keepdims=True)   # <= 6: see respot
    look = 0.25 * float(np.linalg.norm(hi[[0, 2]] - lo[[0, 2]]))                            # look-at distance ~ quarter of the room
    pose = lambda i: dict(position=[round(float(x), 4) for x in cs[i]], target=[round(float(x), 4) for x in cs[i] + look * fs[i]], fov=round(fov, 1))
    fn = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)                                 # spots: the exact filmed pose, as their thumbnail
    exact = lambda i: dict(position=[round(float(x), 4) for x in cam[i]], target=[round(float(x), 4) for x in cam[i] + look * fn[i]], fov=round(fov, 1))
    centre = (lo + hi) / 2; start = int(np.argmin(np.linalg.norm((cam - centre)[:, [0, 2]], axis=1)))   # nearest to the middle
    keys = np.unique(np.linspace(0, len(cam) - 1, min(len(cam), 120)).astype(int)); dur = round(len(cam) / 6.0, 1)
    frames = [int(round(k / (len(cam) - 1) * dur * 30)) for k in keys]                        # keyframe times are frame numbers
    track = dict(name='Walk', duration=dur, frameRate=30, loopMode='repeat', interpolation='spline', smoothness=1,
                 keyframes=dict(times=frames,
                                values=dict(position=[float(x) for k in keys for x in pose(k)['position']],
                                            target=[float(x) for k in keys for x in pose(k)['target']], fov=[fov] * len(keys))))
    spots = np.linspace(0, len(cam) - 1, 8).astype(int)[1:-1]
    ann = [dict(position=exact(i)['target'], title=f'Spot {n + 1}', text='Real filmed view', camera=dict(initial=exact(i))) for n, i in enumerate(spots)]
    oann, oitems = objects_layer(run, tour, cam, fov, len(ann)); ann += oann
    settings = dict(version=2, tonemapping='none', highPrecisionRendering=False, background=dict(color=[0.08, 0.08, 0.09]),
                    postEffectSettings=dict(sharpness=dict(enabled=True, amount=0.2), bloom=dict(enabled=False, intensity=0.1, blurLevel=2),
                                            grading=dict(enabled=False, brightness=1, contrast=1, saturation=1, tint=[1, 1, 1]),
                                            vignette=dict(enabled=False, intensity=0.5, inner=0.3, outer=0.75, curvature=1),
                                            fringing=dict(enabled=False, intensity=0.5)),
                    animTracks=[track], cameras=[dict(initial=pose(start))], annotations=ann, startMode='animTrack')
    json.dump(settings, open(f'{tour}/settings.json', 'w'))
    Wp, Hp, px = floorplan(run, tour, M)                                                   # top-down picture for the minimap
    from PIL import Image
    os.makedirs(f'{tour}/thumbs', exist_ok=True)
    for n, i in enumerate(spots):
        Image.open(p.image_paths[i]).convert('RGB').resize((240, int(240 * Hd / Wd))).save(f'{tour}/thumbs/spot{n + 1}.jpg', quality=82)
    overlay = dict(floor=dict(x0=float(lo[0]), z0=float(lo[2]), px=px, w=Wp, h=Hp),
                   spots=[dict(title=a_['title'], x=float(cam[i][0]), z=float(cam[i][2]), thumb=f'thumbs/spot{n + 1}.jpg') for n, (i, a_) in enumerate(zip(spots, ann))], objects=oitems)
    json.dump(overlay, open(f'{tour}/overlay.json', 'w'))
    info = dict(metres_per_unit=round(s, 4), frames=len(cam), start_frame=start, fov=round(fov, 1), room_box=[lo.round(3).tolist(), hi.round(3).tolist()], walk_seconds=dur, spots=len(spots), objects=len(oitems))
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR PLAN', json.dumps(info), flush=True)
    return info

def objects_layer(run, tour, cam, fov, first):
    """The object layer (objects3d.py: <run>/objects.json + crops/) in the tour: a labelled hotspot on top of each object; clicking
    it, its search hit or the object itself flies to the real filmed frame that showed it best, looking at it (cameras = video order,
    room coordinates; objects3d writes boxes in the same coordinates: level matrix x metres per unit). No objects.json: nothing."""
    f = f'{run}/objects.json'
    if not os.path.exists(f): return [], []
    from PIL import Image
    os.makedirs(f'{tour}/objects', exist_ok=True); ann, items = [], []
    for o in json.load(open(f))['objects']:
        lo, hi = np.array(o['box_m'][0], float), np.array(o['box_m'][1], float); c = (lo + hi) / 2; sz = hi - lo
        fi = o.get('frame')
        if fi is None or fi >= len(cam):                                                    # no best frame: the nearest one >= 1 m away
            d = np.linalg.norm(cam - c, axis=1); fi = int(np.argmin(np.where(d >= 1.0, d, 1e9)))
        eye = cam[fi]; r = lambda v: [round(float(x), 3) for x in v]
        w, dp = sorted([sz[0], sz[2]], reverse=True)
        size = f'about {w:.1f} x {dp:.1f} m, {sz[1]:.1f} m tall'
        thumb = None
        if o.get('crop') and os.path.exists(f'{run}/{o["crop"]}'):
            t = Image.open(f'{run}/{o["crop"]}').convert('RGB'); t.thumbnail((200, 200)); thumb = f'objects/{o["id"]}.jpg'; t.save(f'{tour}/{thumb}', quality=82)
        label = o.get('label') or f'object {o["id"]}'
        ann.append(dict(position=r([c[0], hi[1] + 0.06, c[2]]), title=label[:1].upper() + label[1:], text=size,
                        camera=dict(initial=dict(position=r(eye), target=r(c), fov=round(fov, 1)))))
        words = ' '.join(x for x in (label, o.get('vlm', ''), o.get('clip_label', '')) if x).lower()
        items.append(dict(i=first + len(ann) - 1, id=o['id'], label=label, size=size, words=words, thumb=thumb, lo=r(lo), hi=r(hi)))
    print('tour objects:', len(items), [x['label'] for x in items], flush=True)
    return ann, items

def collision(run, tour, faces=150000):
    """Walk-mode collision = the room box (floor slab + 4 walls: never a hole to fall or walk through) + the furniture from the
    room mesh (mesh_tsdf.py), cut to the room box above the floor and simplified to ~150k faces. Run 4's full mesh was 3M faces
    / 63 MB with holes in walls (59 % of voxels observed): too big to download and leaky on its own."""
    import trimesh
    pl = json.load(open(f'{tour}/plan.json')); s = pl['scale']; lo, hi = np.array(pl['box'][:3]), np.array(pl['box'][3:])
    parts = [trimesh.load(f'{tour}/room_collision.glb', force='mesh')] if os.path.exists(f'{tour}/room_collision.glb') else []
    if os.path.exists(f'{tour}/walls.json'):                                               # inner walls from the floor plan (a house)
        wj = json.load(open(f'{tour}/walls.json'))
        for x0, z0, x1, z1 in wj['walls']:
            L = float(np.hypot(x1 - x0, z1 - z0)); hgt = float(wj.get('height', 2.5))
            if L < 0.3: continue
            T = np.eye(4); u = np.array([x1 - x0, z1 - z0]) / L; T[[0, 2], 0] = u; T[[0, 2], 2] = [-u[1], u[0]]
            T[0, 3], T[1, 3], T[2, 3] = (x0 + x1) / 2, hgt / 2, (z0 + z1) / 2
            parts.append(trimesh.creation.box(extents=(L, hgt, 0.12), transform=T))
    mesh_meta = f'{run}/mesh/mesh.json'
    mesh_kind = ''
    if os.path.exists(mesh_meta):
        try: mesh_kind = json.load(open(mesh_meta)).get('kind', '')
        except Exception: pass
    if os.path.exists(f'{run}/mesh/mesh.glb') and mesh_kind != '2dgs':
        m = trimesh.load(f'{run}/mesh/mesh.glb', force='mesh'); m.apply_scale(s)
        c = m.triangles_center; keep = np.all((c >= lo + [0, 0.08, 0]) & (c <= hi), 1)            # inside, not the floor itself
        m.update_faces(keep); m.remove_unreferenced_vertices()
        if len(m.faces) > faces:
            try: m = m.simplify_quadric_decimation(face_count=faces)
            except Exception as e: print('mesh simplify failed, keeping', len(m.faces), 'faces:', str(e)[:120], flush=True)
        if len(m.faces): parts.append(m)
    if parts:
        trimesh.util.concatenate(parts).export(f'{tour}/collision.glb')
        print('collision:', sum(len(x.faces) for x in parts), 'faces,', round(os.path.getsize(f'{tour}/collision.glb') / 1e6, 1), 'MB', flush=True)


def smear_mask(v, box, scale):
    """Spiky streaks: longest axis > 8x the middle one and > 1 % of the room diagonal (box in metres, scale = metres per unit)."""
    sc = np.sort(np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)), 1)
    room = float(np.linalg.norm(np.array(box[3:]) - np.array(box[:3]))) / scale
    return (sc[:, 2] > 8 * sc[:, 1]) & (sc[:, 2] > 0.01 * room)

def clean(run, tour, out):
    """The model as the tour ships it before conversion (smear filter only), in the model's own coordinates, for measuring."""
    from plyfile import PlyData, PlyElement
    pl = json.load(open(f'{tour}/plan.json')); v = PlyData.read(f'{run}/full/model.ply')['vertex'].data
    bad = smear_mask(v, pl['box'], pl.get('scale', 1.0)); PlyData([PlyElement.describe(v[~bad], 'vertex')]).write(out)
    print(f'smears removed: {bad.sum()} of {len(v)}', flush=True)

def overlay(html):
    """The floor plan / spots / controls overlay (OVERLAY) put in, or swapped for an earlier version (tour_publish.py head)."""
    if 'id="nm-plan"' in html:
        i = html.rfind('<style>', 0, html.index('id="nm-plan"')); j = html.index('</body>', i); html = html[:i] + html[j:]
    return html.replace('</body>', OVERLAY + '\n    </body>', 1)


def patch_js(js):
    """The viewer script, patched: walk mode also in rooms smaller than 5 x 5 m, and its walk controller exposed as
    window.nammaWalk (eye height and the touch jump button of OVERLAY). Idempotent."""
    if 'window.nammaWalk' not in js:
        js, k = re.subn(r'walk\s*:\s*new\s+([\w$]+)\(\)', r'walk: (window.nammaWalk = new \1())', js, count=1)
        if not k: print('patch_js: walk controller not found in this viewer build: eye height stays hidden', flush=True)
    return re.sub(r'WALK_MIN_HORIZONTAL_RANGE\s*=\s*5\b', 'WALK_MIN_HORIZONTAL_RANGE = 2', js)


def head(html, fmt, bud=4):
    """The viewer start options, put first in <head>. Also used to update a published tour's page (tour_publish.py head)."""
    html = re.sub(r'\n    <script>\(function\(\)\{var q=new URLSearchParams.*?</script>(<style>body:not\(\.namma-ready\).*?</script>)?', '',
                  html, count=1)                                                         # an earlier version
    # streamed LOD: no fixed budget (the viewer sizes it for the device), but 'fullload': the loading screen stays until full
    # detail is in, so nobody sees a blurry room first (owner, 30 Oct: "load full room at once ... else people think it's bad")
    pre = 'if(!q.has("fullload"))q.set("fullload","1");' if fmt == 'lod' else 'if(!q.has("budget"))q.set("budget","%d");' % bud
    # Defaults per device, written once (marker key), so a visitor's own choice is kept afterwards. Laptop: Quality High (8 M
    # splats), Performance Mode off. Phone (owner, 31 Oct): Quality Performance, Performance Mode and gaming controls on.
    pre += ('try{if(localStorage.getItem("nammaQ")!=="3"){var m=/Mobi|Android|iPhone|iPad/.test(navigator.userAgent)||'
            '(navigator.maxTouchPoints>1&&/Macintosh/.test(navigator.userAgent));localStorage.setItem("quality",m?"performance":"high");'
            'localStorage.setItem("performanceMode",m?"true":"false");if(m)localStorage.setItem("gamingControls","true");'
            'localStorage.setItem("controlsPanelOpen","false");localStorage.setItem("nammaQ","3");}}catch(e){}')   # one help card: ours
    # The viewer draws the room while it streams in (pieces popping in behind the loading bar); hide the 3D view until the
    # viewer reports it loaded (all detail in, with fullload), at most 2 min; dark meanwhile. Owner, 31 Oct: a clear loading
    # state in the centre of the screen -> our spinner + the viewer's own percent, until loaded; a hint if it stalls.
    cover = ('<style>body:not(.namma-ready){background:#111}body:not(.namma-show) canvas{visibility:hidden}'
             '#nm-load{position:fixed;inset:0;z-index:40;display:flex;flex-direction:column;align-items:center;justify-content:center;'
             'gap:14px;pointer-events:none;color:#e0dcdd;font:14px Arial,sans-serif;text-align:center;padding:0 24px}'
             'body.namma-ready #nm-load{display:none}body:not(.namma-ready) .sse-loadingWrap{display:none}'
             '#nm-load i{width:44px;height:44px;border-radius:50%;border:3px solid rgba(255,255,255,.12);border-top-color:#e8a96b;'
             'animation:nmspin .9s linear infinite}#nm-load small{color:#aaa;font-size:12px}'
             '@keyframes nmspin{to{transform:rotate(360deg)}}</style>'
             '<script>(function(){var t0=Date.now(),L=null,k=setInterval(function(){var v=window.viewer,d=document.body,s=(Date.now()-t0)/1000;'
             'if(!d)return;if(!L){L=document.createElement("div");L.id="nm-load";L.setAttribute("role","status");'
             'L.innerHTML="<i></i><b>Loading the room</b><small>Getting everything ready\u2026</small>";d.appendChild(L);}'
             'var t=document.querySelector(".sse-loadingText"),p=t&&/(\\d+)\\s*%/.exec(t.textContent||""),m=L.querySelector("small");'
             'if(p)m.textContent=p[1]+"%";else if(s>90&&!(v&&v.state))m.textContent="Taking longer than usual. Check your connection and reload.";'
             'if((v&&v.state&&v.state.loaded)||s>120){d.classList.add("namma-ready","namma-show");clearInterval(k);}},200);})();</script>')
    return html.replace('<head>', '<head>\n    <script>(function(){var q=new URLSearchParams(location.search);' + pre +
                        'if(!q.has("hpr"))q.set("hpr","1");history.replaceState(null,"",location.pathname+"?"+q.toString()+location.hash);})();</script>' +
                        cover, 1)


def build(run, tour, public, st='npx --no-install splat-transform'):
    """plan.json + settings.json (from plan) + the model -> a static tour folder."""
    pl = json.load(open(f'{tour}/plan.json')); E, t, b, s = pl['rotate'], pl['translate'], pl['box'], pl.get('scale', 1.0)
    # Remove the obsolete walk-path metadata from a tour being rebuilt.
    if os.path.exists(f'{tour}/walkpath.json'): os.remove(f'{tour}/walkpath.json')
    # The tour must show what the trained model.ply shows (owner, 30 Oct: floor, ceiling and wall patches were missing, black
    # patches). Two of our own steps deleted real structure, so both are now off by default:
    #  - the smear filter (big elongated splats are also how plain walls/ceilings are made): TOUR_SMEAR=1 to use it again;
    #  - the room-box crop (-B): the SfM box ended below the ceiling and cut floor/walls, and it also deleted what lies beyond
    #    the walls (a mirror is a second room behind it, a window shows the outside): TOUR_CROP=1 to use it again.
    # Only invalid splats go (-N). TOUR_FLOATERS=1 adds splat-transform's voxel floater filter (-F).
    from plyfile import PlyData, PlyElement
    v = PlyData.read(f"{run}/{os.environ.get('TOUR_MODEL', 'full/model.ply')}")['vertex'].data   # TOUR_MODEL=solid/clean.ply: cleaned
    bad = smear_mask(v, b, s) if os.environ.get('TOUR_SMEAR') == '1' else np.zeros(len(v), bool)
    PlyData([PlyElement.describe(v[~bad], 'vertex')]).write(f'{tour}/clean.ply'); print(f'smears removed: {bad.sum()} of {len(v)}', flush=True)
    # colour: SOG's spherical-harmonics clustering takes hours on a CPU (150k splats: SH1 208 s, SH3 > 300 s, SH0 2 s), so the
    # tour is PlayCanvas compressed PLY with SH band 1 (view-dependent colour). Measured on run 4's held-out frames
    # (final/tour_fidelity.py, fidelity/20261027-0745) all 3 bands scored 2 dB higher, but the owner rejected that tour
    # (27 Oct rebuild: dark walls and ceiling in the viewer), so SH1 stays the default; TOUR_SH=3 for all bands (91 MB vs 37.5 MB
    # for 1.5M splats). TOUR_FORMAT=sog gives base colour only at ~11 bytes/splat (-2.8 dB).
    fmt = os.environ.get('TOUR_FORMAT', 'cply')
    # 30 Oct, measured in the real conversion on run 4 and run 6 held-out photos: all 3 bands -0.02 dB vs the trained model, band 1
    # only -2.2 / -2.4 dB. The "dark walls" of 27 Oct came from the room crop and smear filter (both gone), so all bands is the default.
    out, sh = (f'{tour}/scene.compressed.ply', int(os.environ.get('TOUR_SH', 3))) if fmt == 'cply' else (f'{tour}/scene.sog', 0)
    if fmt == 'lod':                     # 30 Oct, whole-house plan: Streamed SOG = the same model at several levels of detail in spatial
        sh = int(os.environ.get('TOUR_SH', 3)); out = f'{tour}/lod/lod-meta.json'   # chunks; the viewer opens on a coarse whole and
        shutil.rmtree(f'{tour}/lod', ignore_errors=True); os.makedirs(f'{tour}/lod')   # streams full detail near the camera
    cmd = (f'{st} -w -g cpu {tour}/clean.ply -H {sh} -r {E[0]:.6f},{E[1]:.6f},{E[2]:.6f} -t {t[0]:.6f},{t[1]:.6f},{t[2]:.6f} -s {s:.6f} '
           + (f'-B {",".join(f"{x:.4f}" for x in b)} ' if os.environ.get('TOUR_CROP') == '1' else '')
           + ('-F ' if os.environ.get('TOUR_FLOATERS') == '1' else '') + f'-N {tour}/full_t.ply' if fmt == 'lod' else
           f'{st} -w -g cpu {tour}/clean.ply -H {sh} -r {E[0]:.6f},{E[1]:.6f},{E[2]:.6f} -t {t[0]:.6f},{t[1]:.6f},{t[2]:.6f} -s {s:.6f} '
           + (f'-B {",".join(f"{x:.4f}" for x in b)} ' if os.environ.get('TOUR_CROP') == '1' else '')
           + ('-F ' if os.environ.get('TOUR_FLOATERS') == '1' else '') + f'-N {out}')
    print(cmd, flush=True); subprocess.run(cmd, shell=True, check=True); os.remove(f'{tour}/clean.ply')
    if fmt == 'lod':                                                       # levels: 100 % (LOD 0), then decimated copies
        lv = [x for x in os.environ.get('TOUR_LOD_LEVELS', '30,10,3').split(',') if x]; parts = [f'{tour}/full_t.ply -l 0']
        for k, pct in enumerate(lv):
            c = f'{st} -w -g cpu {tour}/full_t.ply -d {pct}% {tour}/lod{k + 1}.ply'; print(c, flush=True); subprocess.run(c, shell=True, check=True)
            parts.append(f'{tour}/lod{k + 1}.ply -l {k + 1}')
        c = f'{st} -w -g cpu ' + ' '.join(parts) + f" -H {sh} -i {os.environ.get('TOUR_SH_ITER', '1')} --lod-chunk-count {os.environ.get('TOUR_LOD_CHUNK_K', '16')} {out}"   # small chunks: SH palette <= chunk size (512K chunks with a 64K palette ran > 1.5 h)
        print(c, flush=True); subprocess.run(c, shell=True, check=True)
        assert os.path.exists(out) and len(os.listdir(os.path.dirname(out))) > 1, 'streamed LOD not written'
        for f in [f'{tour}/full_t.ply'] + [f'{tour}/lod{k + 1}.ply' for k in range(len(lv))]: os.remove(f)
        subprocess.run(f'{st} {out} --info null | head -20', shell=True)
    boot = dict(contentUrl='./' + os.path.relpath(out, tour), settingsUrl='./settings.json')
    # collision for walk mode: the room mesh (mesh_tsdf.py, room coordinates before scaling) if there is one, else the room box
    collision(run, tour)
    if os.path.exists(f'{tour}/collision.glb'): boot['collisionUrl'] = './collision.glb'
    shutil.copy(f'{public}/index.css', tour)
    js = open(f'{public}/index.js').read()
    assert 'WALK_MIN_HORIZONTAL_RANGE = 5' in js; open(f'{tour}/index.js', 'w').write(patch_js(js))
    html = open(f'{public}/index.html').read(); a = html.index('id="sse-bootstrap">') + len('id="sse-bootstrap">'); c = html.index('</script>', a)
    html = (html[:a] + '\n' + json.dumps(boot) + '\n        ' + html[c:]).replace('<title>SuperSplat Viewer</title>', '<title>NAMMA 3D tour</title>')
    nsp = len(v) - int(bad.sum()); bud = max(4, int(np.ceil(nsp / 1e6)) + 1)                 # the viewer's splat budget, millions
    html = head(html, fmt, bud)
    html = html.replace("createViewer({ container: document.body, settings, ...options });",
                        "window.viewer = await createViewer({ container: document.body, settings, ...options });")
    if os.path.exists(f'{tour}/overlay.json'): html = overlay(html)
    open(f'{tour}/index.html', 'w').write(html)
    info = json.load(open(f'{tour}/tour.json')); info.update(smears_removed=int(bad.sum()), collision='collisionUrl' in boot, scene_mb=round((sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.dirname(out)) for f in fs) if fmt == 'lod' else os.path.getsize(out)) / 1e6, 1), format=fmt, sh_bands=sh)
    json.dump(info, open(f'{tour}/tour.json', 'w'), indent=1); print('TOUR', json.dumps(info), flush=True)
    return info

if __name__ == '__main__':
    {'plan': plan, 'build': build, 'floorplan': floorplan, 'collision': collision, 'clean': clean, 'respot': respot}[sys.argv[1]](*sys.argv[2:])
