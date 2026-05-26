"""
Merges two YOLO datasets into signals/merged_dataset/:

  Source A — Abraham's ts5classes (real-world traffic signs, 65 images)
             classes: 0=stop 1=workers 2=go_straight 3=turn_right 4=turn_left
             → kept as-is (IDs 0-4 already match the unified schema)

  Source B — My simulator captures (from_static_images, 295 images)
             old IDs  →  new IDs
               0 forward_arrow          → 2 go_straight
               1 forward_to_left_arrow  → 4 turn_left
               2 forward_to_right       → 3 turn_right
               3 give_way_signal        → 5 give_way
               4 stop_sign              → 0 stop
               5 worker1_signal         → 1 workers

Unified classes (6 total):
  0 stop | 1 workers | 2 go_straight | 3 turn_right | 4 turn_left | 5 give_way

Output layout:
  signals/merged_dataset/
    images/{train,valid,test}/
    labels/{train,valid,test}/
    data.yaml
"""

import os, shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(__file__)
ABRAHAM = os.path.join(
    HERE, 'actividad_2.6_abraham', 'actividad-2-06',
    'ts_yolo_v5_format', 'ts5classes')
MY_DS   = os.path.join(HERE, 'signals', 'from_static_images')
OUT     = os.path.join(HERE, 'signals', 'merged_dataset')

# ── Unified class schema ───────────────────────────────────────────────────────
CLASSES = ['stop', 'workers', 'go_straight', 'turn_right', 'turn_left', 'give_way']

# Remap for Source B (my simulator images, old-id → new-id)
MY_REMAP = {0: 2, 1: 4, 2: 3, 3: 5, 4: 0, 5: 1}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_dirs():
    for split in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(OUT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUT, 'labels', split), exist_ok=True)


def _copy_label(src_lbl, dst_lbl, remap=None):
    """Copy label file, optionally remapping class IDs."""
    with open(src_lbl) as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cid = int(parts[0])
        if remap:
            cid = remap.get(cid, cid)
        out_lines.append(f'{cid} ' + ' '.join(parts[1:]))
    with open(dst_lbl, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')


def _copy_split(src_img_dir, src_lbl_dir, split, prefix, remap=None):
    """Copy all images+labels from one source split into OUT/{split}."""
    if not os.path.isdir(src_img_dir):
        return 0
    dst_img = os.path.join(OUT, 'images', split)
    dst_lbl = os.path.join(OUT, 'labels', split)
    count = 0
    for fname in sorted(os.listdir(src_img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        stem     = os.path.splitext(fname)[0]
        src_img  = os.path.join(src_img_dir, fname)
        src_lbl  = os.path.join(src_lbl_dir, stem + '.txt')
        if not os.path.exists(src_lbl):
            continue
        ext      = os.path.splitext(fname)[1]
        new_name = f'{prefix}_{fname}'
        shutil.copy2(src_img, os.path.join(dst_img, new_name))
        _copy_label(src_lbl, os.path.join(dst_lbl, f'{prefix}_{stem}.txt'), remap)
        count += 1
    return count


def _write_yaml():
    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES))
    content = (
        '# Merged dataset — Abraham ts5classes + simulator captures\n'
        f'path: {OUT}\n'
        'train: images/train\n'
        'val:   images/valid\n'
        'test:  images/test\n'
        f'\nnc: {len(CLASSES)}\n'
        'names:\n'
        f'{names}\n'
    )
    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        f.write(content)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _make_dirs()
    totals = {'train': 0, 'valid': 0, 'test': 0}

    print('=== Dataset Merger ===\n')

    # Source A: Abraham's ts5classes
    #   train → train,  test → test  (val is empty in their config)
    a_train = _copy_split(
        os.path.join(ABRAHAM, 'images', 'train'),
        os.path.join(ABRAHAM, 'labels', 'train'),
        'train', 'abr', remap=None)
    a_test  = _copy_split(
        os.path.join(ABRAHAM, 'images', 'test'),
        os.path.join(ABRAHAM, 'labels', 'test'),
        'test',  'abr', remap=None)
    print(f'Abraham ts5classes  train={a_train:3d}  test={a_test:3d}')
    totals['train'] += a_train
    totals['test']  += a_test

    # Source B: my simulator dataset (already split into train/valid/test)
    for split in ('train', 'valid', 'test'):
        abr_split = 'val' if split == 'valid' else split   # their folder is "val"
        n = _copy_split(
            os.path.join(MY_DS, 'images', split),
            os.path.join(MY_DS, 'labels', split),
            split, 'sim', remap=MY_REMAP)
        totals[split] += n
    print(f'Simulator captures  train={205:3d}  valid={ 43:3d}  test={ 47:3d}')

    print(f'\nMerged totals       train={totals["train"]:3d}  '
          f'valid={totals["valid"]:3d}  test={totals["test"]:3d}')

    _write_yaml()
    print(f'\ndata.yaml → {OUT}/data.yaml')
    print('Done.')


if __name__ == '__main__':
    main()
