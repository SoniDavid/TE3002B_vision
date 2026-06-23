"""
Reorganises from_static_images/ into the standard YOLO folder layout:

  from_static_images/
    images/train/   images/valid/   images/test/
    labels/train/   labels/valid/   labels/test/
    data.yaml  (updated)

Split per class (stratified): 70% train / 15% valid / 15% test.
Original class sub-folders are left untouched.
"""

import os
import random
import shutil

BASE   = os.path.join(os.path.dirname(__file__), 'signals', 'from_static_images')
CLASSES = [
    'forward_arrow', 'forward_to_left_arrow', 'forward_to_right',
    'give_way_signal', 'stop_sign', 'worker1_signal',
]
SPLITS = {'train': 0.70, 'valid': 0.15, 'test': 0.15}
SEED   = 42


def make_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(BASE, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(BASE, 'labels', split), exist_ok=True)


def split_class(cls):
    cls_dir = os.path.join(BASE, cls)
    images  = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith('.png'))
    random.shuffle(images)

    n      = len(images)
    n_tr   = max(1, round(n * SPLITS['train']))
    n_vl   = max(1, round(n * SPLITS['valid']))
    # remainder goes to test
    groups = {
        'train': images[:n_tr],
        'valid': images[n_tr:n_tr + n_vl],
        'test':  images[n_tr + n_vl:],
    }
    return groups


def copy_files(cls, groups):
    cls_dir = os.path.join(BASE, cls)
    counts  = {}
    for split, files in groups.items():
        for fname in files:
            src_img = os.path.join(cls_dir, fname)
            src_lbl = os.path.splitext(src_img)[0] + '.txt'
            dst_img = os.path.join(BASE, 'images', split, fname)
            dst_lbl = os.path.join(BASE, 'labels', split,
                                   os.path.splitext(fname)[0] + '.txt')
            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
        counts[split] = len(files)
    return counts


def write_yaml():
    names_lines = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES))
    content = (
        '# YOLO dataset config\n'
        f'path: {BASE}\n'
        'train: images/train\n'
        'val:   images/valid\n'
        'test:  images/test\n'
        '\n'
        f'nc: {len(CLASSES)}\n'
        'names:\n'
        f'{names_lines}\n'
    )
    with open(os.path.join(BASE, 'data.yaml'), 'w') as f:
        f.write(content)


def main():
    random.seed(SEED)
    make_dirs()

    print('Reorganising dataset  (70 / 15 / 15 stratified split)\n')
    totals = {'train': 0, 'valid': 0, 'test': 0}

    for cls in CLASSES:
        groups = split_class(cls)
        counts = copy_files(cls, groups)
        for s, c in counts.items():
            totals[s] += c
        print(f'  {cls:<28s}  train={counts["train"]:3d}  valid={counts["valid"]:3d}  test={counts["test"]:3d}')

    print(f'\n  {"TOTAL":<28s}  train={totals["train"]:3d}  valid={totals["valid"]:3d}  test={totals["test"]:3d}')

    write_yaml()
    print(f'\ndata.yaml updated → {BASE}/data.yaml')
    print('Done.')


if __name__ == '__main__':
    main()
