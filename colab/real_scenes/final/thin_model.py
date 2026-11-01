"""Keep every k-th image (by name, i.e. walk order) of a COLMAP model (cameras.bin, images.bin, points3D.bin): the whole-house
plan of 31 Oct trains the house's main camera model at run 4's frame density without solving the cameras again. Point tracks
keep only the kept images; points seen by fewer than 2 kept images are dropped (and unlinked from their 2D observations).
Plain numpy reader/writer of COLMAP's binary model format (the layout of COLMAP's scripts/python/read_write_model.py, BSD).
  python thin_model.py <sparse/0 in> <sparse/0 out> <k> [names.txt]   ->  report JSON on stdout (+ the kept names)"""
import json, os, struct, sys
import numpy as np

NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}             # COLMAP camera model id -> parameters


def _r(f, fmt): b = f.read(struct.calcsize('<' + fmt)); return struct.unpack('<' + fmt, b)


def read_model(d):
    cams, ims, pts = {}, {}, {}
    with open(f'{d}/cameras.bin', 'rb') as f:
        for _ in range(_r(f, 'Q')[0]):
            cid, mid, w, h = _r(f, 'iiQQ'); cams[cid] = (mid, w, h, _r(f, 'd' * NPARAMS[mid]))
    with open(f'{d}/images.bin', 'rb') as f:
        for _ in range(_r(f, 'Q')[0]):
            iid, *qt, cid = _r(f, 'i7di'); name = b''
            while (c := f.read(1)) != b'\x00': name += c
            n = _r(f, 'Q')[0]; p2 = np.frombuffer(f.read(24 * n), dtype=[('x', '<f8'), ('y', '<f8'), ('p', '<i8')]).copy()
            ims[iid] = dict(qt=qt, cam=cid, name=name.decode(), p2=p2)
    with open(f'{d}/points3D.bin', 'rb') as f:
        for _ in range(_r(f, 'Q')[0]):
            pid, x, y, z, r, g, b, e = _r(f, 'Q3d3Bd'); n = _r(f, 'Q')[0]
            pts[pid] = dict(xyz=(x, y, z), rgb=(r, g, b), err=e, track=np.frombuffer(f.read(8 * n), dtype='<i4').reshape(-1, 2).copy())
    return cams, ims, pts


def write_model(d, cams, ims, pts):
    os.makedirs(d, exist_ok=True)
    with open(f'{d}/cameras.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(cams)))
        for cid, (mid, w, h, prm) in cams.items(): f.write(struct.pack('<iiQQ', cid, mid, w, h) + struct.pack('<' + 'd' * len(prm), *prm))
    with open(f'{d}/images.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(ims)))
        for iid, im in ims.items():
            f.write(struct.pack('<i7di', iid, *im['qt'], im['cam']) + im['name'].encode() + b'\x00')
            f.write(struct.pack('<Q', len(im['p2'])) + im['p2'].tobytes())
    with open(f'{d}/points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))
        for pid, p in pts.items():
            f.write(struct.pack('<Q3d3Bd', pid, *p['xyz'], *p['rgb'], p['err']) + struct.pack('<Q', len(p['track'])) + p['track'].astype('<i4').tobytes())


def thin(src, dst, k):
    cams, ims, pts = read_model(src)
    order = sorted(ims, key=lambda i: ims[i]['name']); keep = set(order[::k])
    ims2 = {i: ims[i] for i in order if i in keep}; pts2 = {}
    for pid, p in pts.items():
        t = p['track'][np.isin(p['track'][:, 0], list(keep))]
        if len({int(a) for a in t[:, 0]}) >= 2: pts2[pid] = dict(p, track=t)
    for im in ims2.values():                                                                    # unlink dropped points
        p = im['p2']['p']; gone = (p >= 0) & ~np.isin(p, np.fromiter(pts2.keys(), np.int64, len(pts2)))
        im['p2']['p'][gone] = -1
    write_model(dst, cams, ims2, pts2)
    no_pts = sum(1 for im in ims2.values() if not (im['p2']['p'] >= 0).any())
    return dict(images_in=len(ims), images=len(ims2), points_in=len(pts), points=len(pts2), every=k, frames_without_points=no_pts,
                names=[ims2[i]['name'] for i in sorted(ims2, key=lambda i: ims2[i]['name'])])


if __name__ == '__main__':
    rep = thin(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    if len(sys.argv) > 4: open(sys.argv[4], 'w').write('\n'.join(rep['names']))
    print(json.dumps({k: v for k, v in rep.items() if k != 'names'}), flush=True)
