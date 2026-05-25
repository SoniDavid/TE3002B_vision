"""
Run SignalDetector on a recorded video and save an annotated output.

Modes
-----
    (default)    headless, YOLO + blur pre-filter, saves output video fast
    --show       live preview windows, YOLO + blur pre-filter
    --realtime   live preview at original video fps, YOLO + blur pre-filter
    --benchmark  prints timing/FPS stats and realtime viability summary
    --cpu        allow CPU fallback if CUDA is unavailable

Usage:
    python3 test_against_video.py                                        # headless, fast
    python3 test_against_video.py recording_mode2_20260423_113645.avi
    python3 test_against_video.py input.avi output.avi
    python3 test_against_video.py input.avi --show                       # live preview
    python3 test_against_video.py input.avi --realtime                   # realtime @ fps
    python3 test_against_video.py input.avi --benchmark                  # viability report
    python3 test_against_video.py --cpu                                  # force CPU fallback
"""

import sys
import os
import datetime
import cv2

from actividad_2_06 import SignalDetector, CLR_TMPL, CLR_YOLO, CLR_BOTH

SOURCE_COLOR  = {'tmpl': CLR_TMPL, 'yolo': CLR_YOLO, 'both': CLR_BOTH}
DEFAULT_INPUT = 'recording_mode2_20260423_113645.avi'
OUTPUT_DIR    = os.path.join('outputs', 'videos')


def _put_text(img, text, pos, scale=0.4, color=(255, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    cv2.putText(img, text, (x,   y  ), cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thickness)


def _draw_detections(canvas, detections):
    for det in detections:
        color = SOURCE_COLOR.get(det['source'], (200, 200, 200))
        x, y, w, h = det['bbox']
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        label = f"{det['class']} {det['conf']:.2f} [{det['source']}]"
        cv2.putText(canvas, label, (x, max(y-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(canvas, label, (x, max(y-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def _safe_avg(values):
    return (sum(values) / len(values)) if values else 0.0


def _print_benchmark_stats(stats, src_fps):
    frames = stats['frames_total']
    if frames == 0:
        print('\nBenchmark: no frames processed.')
        return

    avg_total_ms = _safe_avg(stats['total_ms'])
    avg_blur_ms  = _safe_avg(stats['blur_ms'])
    avg_tmpl_ms  = _safe_avg(stats['tmpl_ms'])
    avg_yolo_ms  = _safe_avg(stats['yolo_ms'])
    avg_fuse_ms  = _safe_avg(stats['fuse_ms'])

    effective_fps = (1000.0 / avg_total_ms) if avg_total_ms > 0 else 0.0
    viable = effective_fps >= src_fps

    print('\nBenchmark summary:')
    print(f"  Frames processed      : {frames}")
    print(f"  Blurry frames skipped : {stats['frames_blurry']} ({stats['frames_blurry']/frames*100:.1f}%)")
    print(f"  Avg total time/frame  : {avg_total_ms:.2f} ms")
    print(f"  Avg blur check        : {avg_blur_ms:.2f} ms")
    print(f"  Avg template match    : {avg_tmpl_ms:.2f} ms")
    print(f"  Avg YOLO inference    : {avg_yolo_ms:.2f} ms")
    print(f"  Avg fusion            : {avg_fuse_ms:.3f} ms")
    print(f"  Effective FPS         : {effective_fps:.2f}")
    print(f"  Source FPS            : {src_fps:.2f}")
    print(f"  Realtime viable?      : {'YES' if viable else 'NO'}")


def run(input_path, output_path, debug_path, show=False, realtime=False,
        benchmark=False, allow_cpu=False):
    try:
        detector = SignalDetector(
            debug=show,
            yolo_device='cuda',
            allow_cpu_fallback=allow_cpu,
        )
    except RuntimeError as exc:
        print(f'ERROR: {exc}')
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'ERROR: cannot open {input_path}')
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 40.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    writer       = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    debug_writer = None

    print(f'Input  : {input_path}  ({w}×{h} @ {fps:.1f} fps, {total} frames)')
    print(f'Output : {output_path}')
    if show:
        print(f'Debug  : {debug_path}')
    mode_str = 'realtime (YOLO-only)' if realtime else ('live preview' if show else 'headless (fast)')
    print(f'Mode   : {mode_str}')
    print('Press Q to quit.\n' if (show or realtime) else 'Processing...\n')

    frame_ms  = int(1000 / fps)    # target ms per frame for realtime playback
    show_win  = show or realtime

    frame_idx  = 0
    detect_log = {}    # class → first frame seen
    LOG_EVERY  = 100   # print progress every N frames
    stats = {
        'frames_total': 0,
        'frames_blurry': 0,
        'blur_ms': [],
        'tmpl_ms': [],
        'yolo_ms': [],
        'fuse_ms': [],
        'total_ms': [],
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LANCZOS4)

        detections = detector.detect(img)

        if benchmark:
            timing = detector.last_timing
            stats['frames_total'] += 1
            stats['frames_blurry'] += int(detector.is_blurry)
            stats['blur_ms'].append(timing['blur_ms'])
            stats['tmpl_ms'].append(timing['tmpl_ms'])
            stats['yolo_ms'].append(timing['yolo_ms'])
            stats['fuse_ms'].append(timing['fuse_ms'])
            stats['total_ms'].append(timing['total_ms'])

        for det in detections:
            cls = det['class']
            if cls not in detect_log:
                detect_log[cls] = frame_idx
                print(f'  [{frame_idx:5d}] DETECTED: {cls:<28s} conf={det["conf"]:.2f}  [{det["source"]}]')

        display = img.copy()
        _draw_detections(display, detections)
        pct = frame_idx / total * 100 if total else 0
        _put_text(display, f'{frame_idx}/{total}  {pct:.0f}%', (4, 13))
        if not detections:
            _put_text(display, 'no detection', (4, 26), color=(80, 80, 80))

        writer.write(display)

        if show_win:
            cv2.imshow('Signal Detection', display)
            if show and detector.debug_frame is not None:
                dbg = detector.debug_frame
                if debug_writer is None:
                    dh, dw = dbg.shape[:2]
                    debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (dw, dh))
                debug_writer.write(dbg)
                cv2.imshow('Pipeline Debug', dbg)
            # realtime: honour frame interval; show: just pump events
            wait = max(1, frame_ms) if realtime else 1
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % LOG_EVERY == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f'  [{frame_idx:5d}/{total}]  {pct:.0f}%  detected so far: {list(detect_log)}')

    cap.release()
    writer.release()
    if debug_writer:
        debug_writer.release()
    if show_win:
        cv2.destroyAllWindows()

    print(f'\nDone. {frame_idx} frames processed.')
    print('\nDetection summary (first appearance per class):')
    for cls in sorted(detect_log):
        print(f'  {cls:<28s} → frame {detect_log[cls]}')
    print(f'\nClasses detected: {len(detect_log)}/6  →  {set(detect_log)}')
    print(f'Output saved to: {output_path}')
    if benchmark:
        _print_benchmark_stats(stats, fps)


def main():
    args       = [a for a in sys.argv[1:] if not a.startswith('--')]
    show       = '--show' in sys.argv
    realtime   = '--realtime' in sys.argv
    benchmark  = '--benchmark' in sys.argv
    allow_cpu  = '--cpu' in sys.argv

    input_path = args[0] if len(args) > 0 else DEFAULT_INPUT
    if len(args) > 1:
        output_path = args[1]
    else:
        base        = os.path.splitext(os.path.basename(input_path))[0]
        ts          = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f'detected_{base}_{ts}.avi')

    debug_dir = os.path.dirname(output_path) or OUTPUT_DIR
    os.makedirs(debug_dir, exist_ok=True)
    debug_name = os.path.basename(output_path).replace('detected_', 'debug_detected_')
    debug_path = os.path.join(debug_dir, debug_name)
    run(
        input_path,
        output_path,
        debug_path,
        show=show,
        realtime=realtime,
        benchmark=benchmark,
        allow_cpu=allow_cpu,
    )


if __name__ == '__main__':
    main()
