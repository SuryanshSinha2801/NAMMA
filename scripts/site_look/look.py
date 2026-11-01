"""Look at public web pages like a visitor and keep screenshots (GitHub Actions: the Claude container cannot reach most sites).
URLS.txt: one start URL per line. For each: full-page screenshot, page text, all links; links that look like tours/demos are
opened too (up to MAX_TOURS): screenshot after load, then after dragging / pressing keys to move, so the viewer quality shows.
Everything goes to out/ (+ the private HF dataset <user>/namma-v3-out/site_look/<time>/ if HF_TOKEN is set). Public pages only."""
import json, os, re, time
from playwright.sync_api import sync_playwright
OUT = 'out'; os.makedirs(OUT, exist_ok=True); MAX_TOURS = int(os.environ.get('MAX_TOURS', 4))
urls = [u.strip() for u in open(os.path.join(os.path.dirname(__file__), 'URLS.txt')) if u.strip() and not u.startswith('#')]
log = {}; SECRET = None
for u in list(urls):                     # URLS.txt line 'namma <run> [url file]': one of our own tours (link stays private)
    m = re.fullmatch(r'namma (\S+)(?: (\S+))?', u)
    if m:
        urls.remove(u)
        if not os.environ.get('TOUR_RUN'): os.environ['TOUR_RUN'] = m.group(1); os.environ['TOUR_URL_NAME'] = m.group(2) or ''   # the caller's tour wins
if os.environ.get('TOUR_RUN'):           # our own tour: link read from the private results dataset, never printed (public logs)
    from huggingface_hub import hf_hub_download
    t = open(hf_hub_download('anasrazy/namma-v3-out', f"{os.environ['TOUR_RUN']}/{os.environ.get('TOUR_URL_NAME') or 'tour_url.txt'}", repo_type='dataset', token=os.environ['HF_TOKEN'])).read().strip()
    SECRET = t.split('/')[-2]; urls = ['tour ' + t + '?webgl']
hide = lambda x: x.replace(SECRET, '<folder>') if SECRET else x

CAM = 'window.app ? (() => { const c = window.app.root.findComponents("camera")[0]; if (!c) return null; const p = c.entity.getPosition(); return [p.x, p.y, p.z].map(v => Math.round(v * 100) / 100); })() : null'

def shot(page, name):
    try: page.screenshot(path=f'{OUT}/{name}.jpg', type='jpeg', quality=85, timeout=240000); print('shot', name, flush=True)
    except Exception as e: print('shot failed', name, str(e)[:120], flush=True)
    try: print('camera', name, page.evaluate(CAM), flush=True)                             # PlayCanvas viewers: where are we?
    except Exception as e: print('camera unknown', name, str(e)[:80], flush=True)

HIDE_UI = """(on) => { for (const e of document.querySelectorAll('body *')) {
  if (e.tagName === 'CANVAS' || e.querySelector('canvas')) continue;
  if (on) { e.dataset.nmVis = e.style.visibility; e.style.visibility = 'hidden'; } else if ('nmVis' in e.dataset) { e.style.visibility = e.dataset.nmVis; delete e.dataset.nmVis; } } }"""


def still(page, timeout_s=90):
    """Wait until the camera stops moving (fly-overs between rooms are slow under software WebGL): 3 equal polls 1 s apart."""
    last, same, t0 = None, 0, time.time()
    while time.time() - t0 < timeout_s:
        try: c = page.evaluate(CAM)
        except Exception: c = None
        same = same + 1 if c is not None and last is not None and max(abs(a - b) for a, b in zip(c, last)) < 1e-3 else 0
        if same >= 3: return c
        last = c; page.wait_for_timeout(1000)
    return last


def arrive(page, pos, timeout_s=180):
    """Wait until the camera is at pos (the spot's camera, room coordinates = the viewer's) and has stopped there. Under software
    WebGL the first frames after a jump can take seconds, so 'not moving' alone is caught before the move starts (1 Nov)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try: c = page.evaluate(CAM)
        except Exception: c = None
        if c is not None and sum((a - b) ** 2 for a, b in zip(c, pos)) ** 0.5 < 0.05: return still(page, 20)
        page.wait_for_timeout(1000)
    return None


def view_shot(page, path):
    """The 3D view alone (our UI and the viewer's hidden for the shot), as a PIL image."""
    from PIL import Image
    import io
    page.evaluate(HIDE_UI, True); page.wait_for_timeout(1500)
    try: b = page.screenshot(type='jpeg', quality=85, timeout=240000)
    finally: page.evaluate(HIDE_UI, False)
    open(path, 'wb').write(b); return Image.open(io.BytesIO(b)).convert('RGB')


def match(view, thumb, zoom):
    """How well the view shows what its spot photo shows: correlation of the blurred grey images, the photo cropped to the part of
    the frame the viewer shows (zoom = the view's width as a share of the frame's). Calibrated 1 Nov on the house: views that
    matched their photo 0.85-0.99, views inside furniture / blurred -0.02-0.39."""
    from PIL import ImageFilter
    import numpy as np
    W, H = view.size; tw, th = thumb.size; fw = zoom * tw; fh = fw * H / W
    tc = thumb.crop(tuple(int(round(v)) for v in (tw / 2 - fw / 2, th / 2 - fh / 2, tw / 2 + fw / 2, th / 2 + fh / 2)))
    pr = lambda im: np.asarray(im.convert('L').resize((110, 70)).filter(ImageFilter.GaussianBlur(1)), np.float32)
    a, b = pr(view), pr(tc); a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def sharp(im):
    import numpy as np
    g = np.asarray(im.convert('L').resize((480, 300)), np.float32) / 255
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def assert_tour(page, url, rec, k):
    """ASSERT=1 (our tours, after every publish): fail = the publish is rolled back (tour_publish.py --verify).
    Loaded, no script errors, viewer code present; every spot's view matches its real photo; walk mode at every spot is not blurred."""
    from PIL import Image
    import io, numpy as np
    fails = []; base = url.split('index.html')[0]; get = lambda f: page.request.get(base + f, timeout=60000)
    if rec.get('load_wait_error'): fails.append(f"the room did not load in {int(os.environ.get('LOAD_WAIT_S', 600)) // 60} min")
    if rec.get('page_errors'): fails.append('script errors: ' + '; '.join(rec['page_errors'][:3]))
    try:
        js = get('index.js'); n = len(js.body()) if js.ok else 0
        if n < 1_000_000: fails.append(f'viewer code index.js is {n} bytes')
    except Exception as e: fails.append('index.js: ' + str(e)[:80])
    try:                                                                                  # walk collision: a real mesh, not just the room box
        cg = get('collision.glb'); n = len(cg.body()) if cg.ok else 0; rec['collision_bytes'] = n
        if n < 100_000: fails.append(f'walk collision is only the room box / walls ({n} bytes): no room mesh')
    except Exception as e: fails.append('collision.glb: ' + str(e)[:80])
    spots = []
    try:
        st = get('settings.json').json(); ti = get('tour.json').json() if get('tour.json').ok else {}
        ov = get('overlay.json').json() if get('overlay.json').ok else {'spots': []}
        for i, sp in enumerate(ov.get('spots', [])):
            th = Image.open(io.BytesIO(get(sp['thumb']).body())).convert('RGB'); aspect = th.size[0] / th.size[1]
            fv = st['annotations'][i]['camera']['initial']['fov']                                  # the viewer: horizontal on landscape
            ff = fv if ti.get('fov_kind') == 'horizontal' else np.degrees(2 * np.arctan(np.tan(np.radians(fv) / 2) * aspect))   # the frame
            zoom = float(np.tan(np.radians(fv) / 2) / np.tan(np.radians(ff) / 2))
            page.evaluate(f'window.viewer.selectAnnotation({i})'); cam = arrive(page, st['annotations'][i]['camera']['initial']['position'])
            if cam is None: fails.append(f'spot {i + 1}: the camera never arrived at the spot'); continue
            v = view_shot(page, f'{OUT}/tour{k}_v{i + 1}.jpg'); m = match(v, th, zoom)
            r = dict(spot=i + 1, match=round(m, 3), camera=cam)
            if m < 0.6: fails.append(f"spot {i + 1}: the view does not show its photo (match {m:.2f} < 0.6)")
            try:                                                                            # walk mode right there
                page.evaluate("window.viewer.toggleWalk ? window.viewer.toggleWalk() : (window.viewer.state.cameraMode = 'walk')")
                still(page, 40); w = view_shot(page, f'{OUT}/tour{k}_w{i + 1}.jpg'); ratio = sharp(w) / max(sharp(v), 1e-6)
                r.update(walk_sharp=round(ratio, 2), walk_mode=page.evaluate('window.viewer.state.cameraMode'))
                if ratio < 0.35: fails.append(f'spot {i + 1}: walk view blurred ({ratio:.2f} of the spot view): camera inside geometry?')
            except Exception as e: r['walk_error'] = str(e)[:120]
            page.evaluate("window.viewer.state.cameraMode = 'orbit'"); spots.append(r); print('ASSERT spot', r, flush=True)
    except Exception as e: fails.append('spot check could not run: ' + str(e)[:150])
    res = dict(ok=not fails, fails=fails, spots=spots); rec['assert'] = res
    json.dump(res, open(f'{OUT}/assert.json', 'w'), indent=1); print('ASSERT', json.dumps(res), flush=True)


def visit_tour(ctx, url, k):
    page = ctx.new_page(); rec = dict(url=url, requests=[], page_errors=[])
    page.on('pageerror', lambda e: rec['page_errors'].append(str(e)[:160]))
    page.on('request', lambda r: rec['requests'].append(r.url) if re.search(r'\.(sog|splat|ply|spz|ksplat|json|webp|jpg)(\?|$)', r.url) else None)
    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000); page.wait_for_timeout(15000); shot(page, f'tour{k}_0_loading')
        t0 = time.time()                                                                  # until the viewer reports the scene in (max 10 min)
        try: page.wait_for_function('window.viewer && window.viewer.state && window.viewer.state.loaded', timeout=int(os.environ.get('LOAD_WAIT_S', 600)) * 1000)
        except Exception as e: rec['load_wait_error'] = str(e)[:200]
        rec['loaded_after_s'] = round(15 + time.time() - t0); print('loaded after', rec['loaded_after_s'], 's', flush=True)
        page.wait_for_timeout(3000); shot(page, f'tour{k}_a_loaded')
        if os.environ.get('ASSERT') == '1': assert_tour(page, url, rec, k)
        try:                                                                              # our overlay: controls card, floor plan 2D / 3D
            # software WebGL renders a real scene at a few frames/s, so Playwright's 'stable element' wait can need long
            if page.locator('#nm-hgo').is_visible(): page.click('#nm-hgo', timeout=30000)
            if page.locator('#nm-exp').count():
                page.click('#nm-exp', timeout=30000); page.wait_for_timeout(1500); shot(page, f'tour{k}_a2_plan2d')
                page.click('#nm-fp .tog button[data-m="3d"]', timeout=30000); page.wait_for_timeout(1500); shot(page, f'tour{k}_a3_plan3d')
                page.click('#nm-fpx', timeout=30000); page.wait_for_timeout(500)
            if page.locator('.sse-settings').count():                                     # gear: eye height row on top
                page.click('.sse-settings', timeout=30000, force=True); page.wait_for_timeout(1500); shot(page, f'tour{k}_a4_settings')
                rec['eye_row'] = page.locator('#nm-eyegrp').count(); page.click('.sse-settings', timeout=30000, force=True)
        except Exception as e: rec['overlay_error'] = str(e)[:200]
        try:                                                                              # every spot as a visitor sees it (photo bar)
            for i in range(page.locator('#nm-strip button').count()):
                page.locator('#nm-strip button').nth(i).click(timeout=30000, force=True); page.wait_for_timeout(int(os.environ.get('SPOT_WAIT_MS', 30000))); shot(page, f'tour{k}_s{i + 1}')   # software WebGL: the fly-over to a far spot takes long
        except Exception as e: rec['spots_error'] = str(e)[:200]
        W, H = page.viewport_size['width'], page.viewport_size['height']
        page.mouse.move(W / 2, H / 2); page.mouse.down(); page.mouse.move(W / 2 + 300, H / 2, steps=20); page.mouse.up(); page.wait_for_timeout(4000)
        shot(page, f'tour{k}_b_turned')
        for key in ['w'] * 12 + ['ArrowUp'] * 12: page.keyboard.press(key); page.wait_for_timeout(120)
        page.wait_for_timeout(4000); shot(page, f'tour{k}_c_moved')
        page.mouse.click(W / 2, H * 0.7); page.wait_for_timeout(6000); shot(page, f'tour{k}_d_clicked')
        if page.is_visible('#nm-q'):                                                        # our tours: object search box
            q = os.environ.get('LOOK_QUERY', 'bed'); page.fill('#nm-q', q); page.wait_for_timeout(1500); shot(page, f'tour{k}_f_search')
            rec['search'] = [q, page.eval_on_selector_all('#nm-hits button', 'els => els.map(e => e.innerText.split("\\n")[0])')]
            page.press('#nm-q', 'Enter'); page.wait_for_timeout(9000); shot(page, f'tour{k}_g_found')
        rec['buttons'] = page.eval_on_selector_all('button, [role=button]', 'els => els.map(e => (e.innerText || e.getAttribute("aria-label") || e.title || "").trim()).filter(t => t)')
        for n, label in enumerate([b for b in rec['buttons'] if re.search(r'plan|floor|3d|doll|overview|map|room|bed|living|kitchen|bath|hall|salon|cocina', b, re.I)][:6]):
            try:
                page.get_by_text(label, exact=True).first.click(timeout=5000); page.wait_for_timeout(9000); shot(page, f'tour{k}_e{n}_' + re.sub(r'\W+', '_', label)[:20])
            except Exception as e: print('click failed', label, str(e)[:80], flush=True)
        rec['text'] = page.inner_text('body')[:3000]
        for n, img in enumerate([r for r in dict.fromkeys(rec['requests']) if re.search(r'\.(jpg|webp)', r)][:14]):   # what the page shows
            try:
                r = page.request.get(img, timeout=30000)
                if r.ok: open(f"{OUT}/tour{k}_img{n:02d}_" + re.sub(r'\W+', '_', img.split('/')[-1].split('?')[0])[:40] + '.jpg', 'wb').write(r.body())
            except Exception as e: print('img failed', img, str(e)[:80], flush=True)
    except Exception as e: rec['error'] = str(e)[:300]
    page.close(); return rec

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get('CHROME') or None, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'])
    ctx = b.new_context(viewport={'width': 1100, 'height': 700}, locale='en-US')
    tok = os.environ.get('HF_TOKEN')
    if tok:                                                                                 # private HF Spaces: token only to *.hf.space
        ctx.route(re.compile(r'https://[^/]+\.hf\.space/.*'), lambda route: route.continue_(headers={**route.request.headers, 'authorization': f'Bearer {tok}'}))
    for i, u in enumerate(urls):
        if u.startswith('tour '):                                                           # a tour page: walk around inside it
            log[f'tour{i}'] = visit_tour(ctx, u[5:].strip(), i); print('TOUR', i, hide(json.dumps(log[f'tour{i}']))[:3000], flush=True); continue
        page = ctx.new_page(); page.goto(u, wait_until='networkidle', timeout=90000); page.wait_for_timeout(5000)
        page.screenshot(path=f'{OUT}/site{i}_full.jpg', type='jpeg', quality=75, full_page=True); shot(page, f'site{i}_top')
        links = sorted({a for a in page.eval_on_selector_all('a[href]', 'els => els.map(e => e.href)')})
        text = page.inner_text('body'); log[u] = dict(links=links, text=text[:12000])
        print('PAGE', u, '\nLINKS', json.dumps(links, indent=0), '\nTEXT', text[:8000], flush=True); page.close()
    b.close()
open(f'{OUT}/log.json', 'w').write(hide(json.dumps(log, indent=1)))
tok = os.environ.get('HF_TOKEN')
if tok:
    from huggingface_hub import HfApi
    api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; dst = 'site_look/' + time.strftime('%Y%m%d-%H%M')
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True); api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    print(f'uploaded: hf://datasets/{repo}/{dst}', flush=True)
