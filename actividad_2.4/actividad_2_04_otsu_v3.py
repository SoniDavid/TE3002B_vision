from collections import deque
import numpy as np
import cv2


class CenterLineDetector:
    """
    Center-line detector using Otsu thresholding + contour selection.

    Pipeline:
      S1  ROI crop (bottom 1/3)
      S2  Grayscale
      S3  Gaussian blur (5x5, σ=1.4)
      S4  Otsu threshold (inverted — dark line on light track)
      S5  Morphological OPEN (remove small noise blobs)
      S6  Find external contours
      S7  Smart selection — three cases by contour count:

          n >= 3  Sort valid contours by x-position, return the MIDDLE one.
                  The center line always lies between the two boundaries.

          n == 2  Photometric bilateral check: sample the blurred grayscale
                  SAMPLE_OFFSET pixels to the left and right of each contour
                  centroid. Track surface is bright on both sides of the
                  center line; one side is dark for a boundary.
                  Pick the contour that is flanked by bright on both sides.
                  Fall back to scoring if the check is inconclusive.

          n == 1  Use directly.

          n == 0  Hold prev_cx or use image center.

      S8  Median filter (K=5) over the last 5 raw detections.
          Eliminates single-frame outliers and oscillation between two
          lines without adding perceptible lag (≤2 frames at 40 fps).
          prev_cx is updated from the smoothed value so the bilateral
          fallback always has a stable anchor.
    """

    MIN_AREA      = 50   # px² — ignore blobs smaller than this
    SAMPLE_OFFSET = 28   # px — how far left/right to sample for brightness check
    BRIGHT_THR    = 120  # grayscale — track surface is typically 150-220
    MEDIAN_K      = 3    # median filter window — rejects up to K//2 consecutive outliers

    def __init__(self, debug=False):
        self.cameraWidth  = 320
        self.cameraHeight = 240
        self.prev_cx      = None              # smoothed cx used as bilateral fallback anchor
        self.history      = deque(maxlen=self.MEDIAN_K)  # raw cx values for median filter
        self.debug        = debug             # when True, populate self.debug_frame each call
        self.debug_frame  = None             # tiled pipeline stages (BGR); None if debug=False

    # ── main detection ────────────────────────────────────────────────────────

    def detect_center_line(self, image):
        """
        Detect the center line of the track.

        :param image: BGR image (OpenCV), expected 320×240.
        :return: (cx, cy) pixel coordinates in the original image space.
        """
        h, w = image.shape[:2]

        # S1: ROI — bottom third of the frame
        y_start = (2 * h) // 3
        roi = image[y_start:h, :]

        # S2: Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # S3: Gaussian blur — smooth noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # S4: Otsu threshold (inverted — line is dark, track surface is light)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # S5: Morphological OPEN — remove small isolated noise blobs
        kernel  = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # S6: Find external contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # S7: Smart contour selection → raw detection
        raw_cx, raw_cy = self._select_best_contour(contours, blurred, w, h, y_start)

        # S8: Median filter — smooth out single-frame outliers and oscillation.
        # The window holds the last MEDIAN_K raw cx values; the median is stable
        # as long as more than half the window contains correct detections.
        self.history.append(raw_cx)
        smoothed_cx  = int(np.median(self.history))
        self.prev_cx = float(smoothed_cx)   # anchor always tracks the smoothed value

        cx = max(0, min(w - 1, smoothed_cx))
        cy = raw_cy

        # Optional: build debug tile strip
        if self.debug:
            self.debug_frame = self._build_debug(
                roi, gray, blurred, binary, cleaned, contours,
                cx, cy, y_start, w, h
            )

        return (cx, cy)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _contour_centroid(self, cnt):
        """Return (cx_float, cy_float) centroid or None if degenerate."""
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        return M['m10'] / M['m00'], M['m01'] / M['m00']

    def _is_flanked_bright(self, blurred, cx_cnt, cy_cnt):
        """
        Returns True if the pixel SAMPLE_OFFSET to the left AND right of
        (cx_cnt, cy_cnt) are both bright (= track surface, not floor).

        A center line is flanked by bright track on both sides.
        A boundary line is dark on one side (floor/edge) and bright on the
        other (track interior).
        """
        h, w   = blurred.shape[:2]
        y      = int(np.clip(cy_cnt, 0, h - 1))
        x_left  = int(np.clip(cx_cnt - self.SAMPLE_OFFSET, 0, w - 1))
        x_right = int(np.clip(cx_cnt + self.SAMPLE_OFFSET, 0, w - 1))

        # 5×5 window mean for robustness against single-pixel noise
        def win_mean(x, y_):
            r0, r1 = max(0, y_ - 2), min(h, y_ + 3)
            c0, c1 = max(0, x  - 2), min(w, x  + 3)
            patch  = blurred[r0:r1, c0:c1]
            return float(patch.mean()) if patch.size > 0 else 0.0

        left_bright  = win_mean(x_left,  y) > self.BRIGHT_THR
        right_bright = win_mean(x_right, y) > self.BRIGHT_THR
        return left_bright and right_bright

    # ── contour selection ─────────────────────────────────────────────────────

    def _select_best_contour(self, contours, blurred, w, h, y_start):
        """
        Select the contour corresponding to the center line.

        Three strategies depending on how many valid (area >= MIN_AREA)
        contours exist.
        """
        img_cx    = w / 2.0
        search_cx = self.prev_cx if self.prev_cx is not None else img_cx

        # Collect valid contours with their centroids, sorted by x
        valid = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.MIN_AREA:
                continue
            c = self._contour_centroid(cnt)
            if c is None:
                continue
            valid.append((c[0], c[1], cnt))   # (cx, cy, contour)
        valid.sort(key=lambda v: v[0])         # sort by x-position

        chosen_cx = chosen_cy = None

        if len(valid) == 0:
            pass  # fall through to fallback

        elif len(valid) == 1:
            # Only one candidate — use it directly
            chosen_cx, chosen_cy, _ = valid[0]

        elif len(valid) == 2:
            # ── Bilateral brightness check ────────────────────────────────
            # The center line has track surface (bright) on both sides.
            # A boundary has floor/edge (dark) on one side.
            cx0, cy0, _ = valid[0]
            cx1, cy1, _ = valid[1]
            flanked0 = self._is_flanked_bright(blurred, cx0, cy0)
            flanked1 = self._is_flanked_bright(blurred, cx1, cy1)

            if flanked0 and not flanked1:
                chosen_cx, chosen_cy = cx0, cy0
            elif flanked1 and not flanked0:
                chosen_cx, chosen_cy = cx1, cy1
            else:
                # Inconclusive (both or neither flanked) — fall back to
                # scoring: prefer continuity over geometry here so that
                # in steady state we don't jump away from a good track
                best_score = float('inf')
                for (cx_v, cy_v, _) in valid:
                    dist_center = abs(cx_v - img_cx)    / w
                    dist_prev   = abs(cx_v - search_cx) / w
                    score = 0.4 * dist_center + 0.6 * dist_prev
                    if score < best_score:
                        best_score = score
                        chosen_cx, chosen_cy = cx_v, cy_v

        else:
            # n >= 3: center line is always the MIDDLE contour by x-position
            mid         = len(valid) // 2
            chosen_cx, chosen_cy, _ = valid[mid]

        if chosen_cx is not None:
            cx = max(0, min(w - 1, int(round(chosen_cx))))
            cy = y_start + int(round(chosen_cy))
            return (cx, cy)

        # Fallback: hold previous smoothed position or image center
        cx = int(round(self.prev_cx)) if self.prev_cx is not None else w // 2
        cy = y_start + (h - y_start) // 2
        return (cx, cy)

    # ── debug visualization ───────────────────────────────────────────────────

    def _build_debug(self, roi, gray, blurred, binary, cleaned, contours,
                     cx, cy, y_start, w, h):
        """
        Build a horizontal strip of 6 tiles showing each pipeline stage.

        Tiles (left → right):
          S1: ROI (color) | S2: gray | S3: blur | S4: Otsu | S5: OPEN | S6: contours
        """
        roi_h = roi.shape[0]

        def to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

        def labeled(img, text):
            out = to_bgr(img)
            cv2.putText(out, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 0), 2)
            cv2.putText(out, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1)
            return out

        # Tile 6: contours + selected centroid
        cnt_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cnt_vis, contours, -1, (0, 255, 0), 1)
        cy_roi = cy - y_start
        if 0 <= cy_roi < roi_h:
            cv2.drawMarker(cnt_vis, (cx, cy_roi),
                           (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        n_valid = sum(1 for c in contours if cv2.contourArea(c) >= self.MIN_AREA)
        cv2.putText(cnt_vis, f'n={n_valid} cx={cx}',
                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0),    2)
        cv2.putText(cnt_vis, f'n={n_valid} cx={cx}',
                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        tiles = [
            labeled(roi,     'S1:ROI'),
            labeled(gray,    'S2:gray'),
            labeled(blurred, 'S3:blur'),
            labeled(binary,  'S4:Otsu'),
            labeled(cleaned, 'S5:open'),
            cnt_vis,
        ]

        # Resize all tiles to the same height, then hstack
        resized = []
        for t in tiles:
            sc = roi_h / t.shape[0]
            nw = max(1, int(t.shape[1] * sc))
            resized.append(cv2.resize(t, (nw, roi_h)))
        return np.hstack(resized)
