"""Look at public web pages like a visitor and keep screenshots (GitHub Actions: the Claude container cannot reach most sites).
URLS.txt: one start URL per line. For each: full-page screenshot, page text, all links; links that look like tours/demos are
opened too (up to MAX_TOURS): screenshot after load, then after dragging / pressing keys to move, so the viewer quality shows.
Everything goes to out/ (+ the private HF dataset <user>/namma-v3-out/site_look/<time>/ if HF_TOKEN is set). Public pages only."""
import json, os, re, time
from playwright.sync_api import sync_playwright
OUT = 'out'; os.makedirs(OUT, exist_ok=True); MAX_TOURS = int(os.environ.get('MAX_TOURS', 4))
urls = [u.strip() for u in open(os.path.join(os.path.dirname(__file__), 'URLS.txt')) if u.strip() and not u.startswith('#')]
log = {}; SECRET = None
if os.environ.get('TOUR_RUN'):           # our own tour: link read from the private results dataset, never printed (public logs)
    from huggingface_hub import hf_hub_download
    t = open(hf_hub_download('anasrazy/namma-v3-out', f"{os.environ['TOUR_RUN']}/tour_url.txt", repo_type='dataset', token=os.environ['HF_TOKEN'])).read().strip()
    SECRET = t.split('/')[-2]; urls = ['tour ' + t + '?webgl']
hide = lambda x: x.replace(SECRET, '<folder>') if SECRET else x

CAM = 'window.app ? (() => { const c = window.app.root.findComponents("camera")[0]; if (!c) return null; const p = c.entity.getPosition(); return [p.x, p.y, p.z].map(v => Math.round(v * 100) / 100); })() : null'

def shot(page, name):
    try: page.screenshot(path=f'{OUT}/{name}.jpg', type='jpeg', quality=85, timeout=240000); print('shot', name, flush=True)
    except Exception as e: print('shot failed', name, str(e)[:120], flush=True)
    try: print('camera', name, page.evaluate(CAM), flush=True)                             # PlayCanvas viewers: where are we?
    except Exception as e: print('camera unknown', name, str(e)[:80], flush=True)

def visit_tour(ctx, url, k):
    page = ctx.new_page(); rec = dict(url=url, requests=[])
    page.on('request', lambda r: rec['requests'].append(r.url) if re.search(r'\.(sog|splat|ply|spz|ksplat|json|webp|jpg)(\?|$)', r.url) else None)
    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000); page.wait_for_timeout(45000); shot(page, f'tour{k}_a_loaded')
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
