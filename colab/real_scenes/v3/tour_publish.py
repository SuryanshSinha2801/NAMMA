"""Publish a built tour folder (make_tour.py build) to the public static Hugging Face Space <user>/namma-tour, like splattour's
share links: each tour in a folder with a random name (only people with the link find it), no listing page. A private static
Space cannot serve its files to a browser (the page loads, its scene/scripts do not), so the owner chose public. The link is
NOT printed (GitHub logs of this repo are public): it goes to <run>/tour_url.txt in the private results dataset.
Used by gh_run.py after a run, and on its own (.github/workflows/tour-publish.yml, v3/PUBLISH.json) to republish a run's tour:
  python tour_publish.py <run, e.g. 20261026-0309 | world/<time> | flashworld/<time>> [floorplan] [collision]   (HF_TOKEN in the environment; tour taken from the results
  dataset; 'floorplan': redraw floorplan.png from the run's model first (final/make_tour.py floorplan) and save it back there;
  'collision': rebuild collision.glb = room box + simplified mesh furniture; 'rebuild': convert the run's model.ply into the tour's
  scene file again with the current make_tour.build settings, e.g. all SH bands; 'head': only rewrite the stored tour's page
  (<run>/TOUR_DST) with the current viewer start options, make_tour.head)"""
import json, os, re, secrets, shutil, sys


def restore_js(api, repo, path, dst):
    """The newest version of `path` in the results dataset's history that is not empty, copied to dst."""
    from huggingface_hub import hf_hub_download
    for c in api.list_repo_commits(repo, repo_type='dataset'):
        info = api.get_paths_info(repo, [path], repo_type='dataset', revision=c.commit_id)
        if info and getattr(info[0], 'size', 0) > 100000:
            f = hf_hub_download(repo, path, repo_type='dataset', revision=c.commit_id, local_dir='restore_js')
            print('index.js restored from commit', c.commit_id[:8], os.path.getsize(f), 'bytes', flush=True)
            shutil.copy(f, dst); return dst
    raise SystemExit(f'no non-empty {path} in the history of {repo}')


def prune(api, sp, results_repo, url_path, say=print):
    """The free Space holds 1 GB (30 Oct: full, publish refused). Keep only the folders some tour*_url.txt in the results dataset
    still points to (the links handed out), except the one about to be replaced, then squash the history so the space is freed."""
    from huggingface_hub import CommitOperationDelete, hf_hub_download
    keep = set(x for x in os.environ.get('TOUR_KEEP', '').split(',') if x)               # explicit list (PUBLISH.json "keep")
    for f in ([] if keep else api.list_repo_files(results_repo, repo_type='dataset')):       # else: the main runs' and worlds' links
        if re.fullmatch(r'(\d{8}-\d{4}|world/\d{8}-\d{4})/tour\d*_url\.txt', f) and f != url_path:
            try: keep.add(open(hf_hub_download(results_repo, f, repo_type='dataset', token=api.token, local_dir='urls')).read().strip().split('/')[-2])
            except Exception: pass
    folders = {x.split('/')[0] for x in api.list_repo_files(sp, repo_type='space') if '/' in x}
    drop = sorted(folders - keep)
    if drop:
        api.create_commit(sp, repo_type='space', operations=[CommitOperationDelete(path_in_repo=f + '/', is_folder=True) for f in drop],
                          commit_message=f'remove {len(drop)} tour folders no link points to')
        try: api.super_squash_history(repo_id=sp, repo_type='space')
        except Exception as e: say('history squash failed:', str(e)[:150])
    say('space: kept', len(folders & keep), 'linked tour folders, removed', len(drop))


def lfs_rules(api, sp, kinds=('ply', 'sog', 'spz', 'splat', 'glb', 'bin', 'png', 'jpg', 'webp', 'mp4', 'wasm')):
    """A new Space has no large-file rule for .ply etc. and the Hub rejects binary files sent as plain git blobs (30 Oct, namma-tour-2)."""
    from huggingface_hub import hf_hub_download
    try: ga = open(hf_hub_download(sp, '.gitattributes', repo_type='space', token=api.token, force_download=True, local_dir='ga_space')).read()
    except Exception: ga = ''
    add = [f'*.{k} filter=lfs diff=lfs merge=lfs -text' for k in kinds if f'*.{k} filter=lfs' not in ga]
    if add: api.upload_file(path_or_fileobj=(ga.rstrip('\n') + '\n' + '\n'.join(add) + '\n').encode(), path_in_repo='.gitattributes',
                            repo_id=sp, repo_type='space', commit_message='large-file storage for tour files')


def remember(api, sp, folder, results_repo, url_path):
    """What is live before this publish (Space commit, whether the folder and the link file existed): --verify can put it back."""
    try: prev = api.list_repo_commits(sp, repo_type='space')[0].commit_id
    except Exception: prev = None
    had = bool(prev) and any(f.startswith(folder + '/') for f in api.list_repo_files(sp, repo_type='space'))
    url_had = api.file_exists(results_repo, url_path, repo_type='dataset')
    json.dump(dict(space=sp, folder=folder, prev=prev, had_folder=had, results=results_repo, url_path=url_path, url_had=url_had),
              open('publish_state.json', 'w'), indent=1)


def verify(api, say=print, state='publish_state.json', check='out/assert.json'):
    """After the live check (scripts/site_look/look.py with ASSERT=1 -> out/assert.json): keep the publish, or put the previous
    version back (same folder as it was, or remove a new folder and its link) and fail. 31 Oct: a broken publish stayed live for
    40 min until the owner saw a white screen; this makes that impossible."""
    st = json.load(open(state)); a = json.load(open(check)) if os.path.exists(check) else dict(ok=False, fails=['the live check did not run'])
    if a.get('ok'):
        say('LIVE CHECK PASSED:', json.dumps(a.get('spots', [])))
        for w in a.get('warnings', []): say('WARNING (not rolled back):', w)
        if a.get('warnings') and os.environ.get('GITHUB_STEP_SUMMARY'):
            open(os.environ['GITHUB_STEP_SUMMARY'], 'a').write('Live check warnings:\n' + '\n'.join(f'- {w}' for w in a['warnings']) + '\n')
        return True
    say('LIVE CHECK FAILED:', a.get('fails')); sp, folder = st['space'], st['folder']
    from huggingface_hub import snapshot_download
    if st.get('had_folder') and st.get('prev'):                                            # the previous version of this link, exactly
        d = snapshot_download(sp, repo_type='space', revision=st['prev'], allow_patterns=[f'{folder}/*', f'{folder}/**'], local_dir='rollback', token=api.token)
        api.upload_folder(folder_path=os.path.join(d, folder), path_in_repo=folder, repo_id=sp, repo_type='space', delete_patterns='*',
                          commit_message='roll back: the live check failed')
        say('ROLLED BACK', folder, 'to', st['prev'][:8])
    else:                                                                                  # a new link: take it down again
        api.delete_folder(folder, repo_id=sp, repo_type='space', commit_message='remove: the live check failed')
        if not st.get('url_had'):
            try: api.delete_file(st['url_path'], repo_id=st['results'], repo_type='dataset')
            except Exception: pass
        say('REMOVED the new tour', folder)
    json.dump(dict(st, rolled_back=True, fails=a.get('fails')), open(state, 'w'), indent=1)
    return False


def publish(api, user, results_repo, run, tour_dir, say=print, url_path=None):
    sp = f'{user}/namma-tour'; api.create_repo(sp, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    if os.environ.get('TOUR_PRUNE') == '1':                                               # deletes old tours: only with the owner's OK
        prune(api, sp, results_repo, url_path or f'{run}/tour_url.txt', say)
    for f in {x.split('/')[0] for x in api.list_repo_files(sp, repo_type='space') if '/' in x}:      # old guessable folders out
        if not re.fullmatch(r'\d{8}-\d{4}(-[a-z0-9]+)?-[0-9a-f]{20}', f): api.delete_folder(f, repo_id=sp, repo_type='space')   # (digits: 'world1')
    folder = os.environ.get('TOUR_FOLDER') or f'{run}-{secrets.token_hex(10)}'          # TOUR_FOLDER: update a published tour in place (same link)
    for k in range(1, 10):                         # a full free Space (1 GB) spills over to namma-tour-2, -3, ... (old links stay)
        try:
            lfs_rules(api, sp); remember(api, sp, folder, results_repo, url_path or f'{run}/tour_url.txt')     # for --verify / roll back
            api.upload_folder(folder_path=tour_dir, path_in_repo=folder, repo_id=sp, repo_type='space', ignore_patterns=['plan.json'], delete_patterns=['walkpath.json']); break
        except Exception as e:
            if 'Storage limit' not in str(e) or k == 9: raise
            sp = f'{user}/namma-tour-{k + 1}'; say('Space full, publishing to', sp)
            api.create_repo(sp, repo_type='space', space_sdk='static', private=False, exist_ok=True)
    page = '<!doctype html><meta charset="utf-8"><title>NAMMA</title><body style="font:16px sans-serif;padding:24px;background:#111;color:#ddd"><h1>NAMMA 3D tours</h1><p>Open a tour with its link.</p>'
    open('tours_index.html', 'w').write(page); api.upload_file(path_or_fileobj='tours_index.html', path_in_repo='index.html', repo_id=sp, repo_type='space')
    open('space_readme.md', 'w').write('---\ntitle: NAMMA tour\nemoji: 🏠\ncolorFrom: gray\ncolorTo: gray\nsdk: static\npinned: false\nhf_oauth: true\nhf_oauth_expiration_minutes: 480\nhf_oauth_scopes:\n- read-repos\n- write-repos\n---\n')
    api.upload_file(path_or_fileobj='space_readme.md', path_in_repo='README.md', repo_id=sp, repo_type='space')
    try: api.update_repo_settings(sp, repo_type='space', private=False)
    except Exception: api.update_repo_visibility(sp, private=False, repo_type='space')
    editor_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collision_editor.html')
    if os.path.isfile(editor_file):
        api.upload_file(path_or_fileobj=editor_file, path_in_repo='editor/index.html', repo_id=sp, repo_type='space',
                        commit_message='Deploy NAMMA live collision editor')
        say('collision editor deployed at', f'https://{sp.replace("/", "-")}.static.hf.space/editor/index.html')
    open('tour_url.txt', 'w').write(f"https://{sp.replace('/', '-')}.static.hf.space/{folder}/index.html\n")
    api.upload_file(path_or_fileobj='tour_url.txt', path_in_repo=url_path or f'{run}/tour_url.txt', repo_id=results_repo, repo_type='dataset')
    say('tour published (public Space, random folder; link in the private results dataset: tour_url.txt)')


if __name__ == '__main__' and sys.argv[1:2] == ['--verify']:                         # keep or roll back after the live check
    from huggingface_hub import HfApi
    sys.exit(0 if verify(HfApi(token=os.environ['HF_TOKEN'])) else 'live check failed: rolled back (see the log above)')
if __name__ == '__main__':
    from huggingface_hub import HfApi, snapshot_download
    run = sys.argv[1]; tok = os.environ.get('HF_TOKEN'); assert tok, 'HF_TOKEN missing'
    api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'
    if run.startswith(('world/', 'flashworld/')):                                            # generated worlds: every sample's viewer folder
        import glob
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/*/index.html', f'{run}/*/world.*'], local_dir='pub', token=tok)
        pages = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, run, '*', 'index.html')))
        assert pages, f'no viewer pages in {repo}/{run}'
        for k, pg in enumerate(pages):
            publish(api, user, repo, run.split('/')[-1] + '-world', pg, url_path=f'{run}/{"tour_url.txt" if k == 0 else f"tour{k + 1}_url.txt"}')
            print('published', os.path.basename(pg), 'as', 'tour_url.txt' if k == 0 else f'tour{k + 1}_url.txt', flush=True)
        sys.exit(0)
    fp, co, rb, hd, rs = ('floorplan' in sys.argv[2:], 'collision' in sys.argv[2:], 'rebuild' in sys.argv[2:],
                          'head' in sys.argv[2:], 'respot' in sys.argv[2:])
    rs = rs or ((hd or rb) and 'norespot' not in sys.argv[2:])      # every page update repairs spots + field of view itself (checked, safe)
    extra = ([f'{run}/full/{x}' for x in ('model.ply', 'level.json', 'nav.json')] if fp or rb else []) + ([f'{run}/mesh/{x}' for x in ('mesh.glb', 'mesh.json')] if co or rb else [])
    if rb and os.environ.get('TOUR_MODEL'): extra.append(f"{run}/{os.environ['TOUR_MODEL']}")
    if rs: extra += [f'{run}/sparse/**', f'{run}/full/level.json']
    if rb and os.environ.get('TOUR_MODEL'):                                               # model still being made by another job: wait (max 90 min)
        import time; t0 = time.time()
        while not api.file_exists(repo, f"{run}/{os.environ['TOUR_MODEL']}", repo_type='dataset'):
            assert time.time() - t0 < 5400, 'model never appeared'; print('waiting for', os.environ['TOUR_MODEL'], flush=True); time.sleep(30)
    src = os.environ.get('TOUR_DST', 'tour') if hd and not rb else 'tour'                # 'head': the stored tour it updates
    d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/{src}/*', f'{run}/{src}/**'] + extra, local_dir='pub', token=tok)
    tour = os.path.join(d, run, src); assert os.path.exists(f'{tour}/index.html'), f'no built tour in {repo}/{run}/{src}'
    if fp or co or rb or hd or rs: sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final')); import make_tour
    if rs:                                                                                # spots at the exact filmed poses (make_tour.respot)
        import subprocess
        if not os.path.isdir('gsplat'): subprocess.run('git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat gsplat', shell=True, check=True)
        try: print('respot:', make_tour.respot(os.path.join(d, run), tour, os.path.abspath('gsplat/examples')), flush=True)
        except (AssertionError, FileNotFoundError, StopIteration) as e: print('respot skipped (tour left as it was):', str(e)[:200], flush=True)
    if fp:
        print('floor plan redrawn:', make_tour.floorplan(os.path.join(d, run), tour), flush=True)
        for x in ('floorplan.png', 'walls.json'):
            api.upload_file(path_or_fileobj=f'{tour}/{x}', path_in_repo=f'{run}/{src}/{x}', repo_id=repo, repo_type='dataset')
    if co:
        make_tour.collision(os.path.join(d, run), tour)
        api.upload_file(path_or_fileobj=f'{tour}/collision.glb', path_in_repo=f'{run}/{src}/collision.glb', repo_id=repo, repo_type='dataset')   # the tour being updated (was always tour/)
    if rb:                                                                                # scene file again from the model (current settings)
        import subprocess; W = os.path.abspath('st')
        subprocess.run(f'mkdir -p {W} && cd {W} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 '
                       '@playcanvas/supersplat-viewer@1.41.0 @playcanvas/splat-transform@3.10.3', shell=True, check=True)   # 1.41: streamed LOD
        for f in ('scene.compressed.ply', 'scene.sog'):
            if os.path.exists(f'{tour}/{f}'): os.remove(f'{tour}/{f}')
        shutil.rmtree(f'{tour}/lod', ignore_errors=True)
        print('rebuilt:', make_tour.build(os.path.join(d, run), tour, f'{W}/node_modules/@playcanvas/supersplat-viewer/public',
                                          f'{W}/node_modules/.bin/splat-transform'), flush=True)
        api.upload_folder(folder_path=tour, path_in_repo=f"{run}/{os.environ.get('TOUR_DST', 'tour')}", repo_id=repo, repo_type='dataset', delete_patterns=['walkpath.json'])
    if hd and not rb:                                                                     # page only: current viewer start options
        h = open(f'{tour}/index.html').read(); b = re.search(r'q\.set\("budget","(\d+)"\)', h)
        h = make_tour.head(h, 'lod' if os.path.isdir(f'{tour}/lod') else 'sog', int(b.group(1)) if b else 4)
        if os.path.exists(f'{tour}/overlay.json'): h = make_tour.overlay(h)                # current floor plan / spots / controls UI
        js_path = f'{tour}/index.js'
        if not os.path.exists(js_path) or os.path.getsize(js_path) == 0:                 # emptied by the 31 Oct publish bug: take it back
            js_path = restore_js(api, repo, f'{run}/{src}/index.js', js_path)               # from the newest commit where it wasn't empty
        js = open(js_path).read(); assert len(js) > 100000, f'viewer index.js looks wrong ({len(js)} bytes)'
        open(f'{tour}/index.js', 'w').write(make_tour.patch_js(js))                       # read first, then write (never truncate before reading)
        assert os.path.getsize(f'{tour}/index.js') > 100000
        open(f'{tour}/index.html', 'w').write(h)
        print('page start options updated', flush=True)
        for x in ('index.html', 'index.js', 'settings.json', 'overlay.json', 'tour.json'):     # page + repaired spots / fov kept with the run
            if os.path.exists(f'{tour}/{x}'): api.upload_file(path_or_fileobj=f'{tour}/{x}', path_in_repo=f'{run}/{src}/{x}', repo_id=repo, repo_type='dataset')
    publish(api, user, repo, run, tour, url_path=f"{run}/{os.environ.get('TOUR_URL_NAME', 'tour_url.txt')}")   # a new name keeps the old link
    if os.environ.get('INBOX_JOB'):                                                       # finish an app job whose publish failed
        import inbox; inbox.set_status(os.environ['INBOX_JOB'], 'done', run=run, link=open('tour_url.txt').read().strip())
