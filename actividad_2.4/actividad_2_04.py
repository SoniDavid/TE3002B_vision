from collections import deque
import numpy as np
import cv2


class CenterLineDetector:
    """
    Center-line detector: Otsu + contour selection + temporal smoothing.

    Pipeline
    --------
    S1   ROI crop (bottom 1/3)
    S2   Grayscale
    S3   Gaussian blur  (5x5, sigma=1.4)
    S4   Canny edges    (used by S7 classifier)
    S5   Otsu threshold (THRESH_BINARY_INV — dark line on light track)
    S6   Morphological OPEN  (remove small noise blobs)
    S7   Find external contours -> smart selection:

         Contours are split into SIGNIFICANT (area >= SIGNIFICANT_AREA,
         i.e. real track lines) and TINY (noise / engraving fragments).
         Selection always prefers significant blobs.

         WINDOW  If prev_cx is known, restrict significant candidates to
                 those within WINDOW_HALF px.  If the window contains no
                 significant blobs (line moved away or only noise inside),
                 skip the window and jump to the nearest significant blob
                 globally instead.  This lets the tracker re-acquire the
                 real centre line after turns without chasing noise.

         n >= 3  Sort windowed contours by x-position, pick the MIDDLE.
         n == 2  Canny edge-asymmetry check (falls back to brightness).
         n == 1  Use directly.
         n == 0  Hold prev_cx or image centre.

    S8   Velocity gate: if |raw_cx - prev_cx| > MAX_JUMP, hold prev_cx.
         Gate is ONLY applied when the detection came from a tiny/noise
         blob (area < SIGNIFICANT_AREA).  Jumps to real track lines are
         always allowed so the tracker can recover after turns.

    S9   Median filter (K = MEDIAN_K): final temporal smoothing.
         prev_cx updated from the smoothed value so the window and
         classifier always use a stable anchor.
    """

    MIN_AREA        = 50    # px2  -- ignore blobs smaller than this
    MAX_AREA        = 4000  # px2  -- ignore large merged blobs
    SIGNIFICANT_AREA= 500   # px2  -- real track lines are >= this (1500-2500 typical)
    WINDOW_HALF     = 20    # px   -- x-radius of the tracking search window
    SAMPLE_OFFSET   = 25    # px   -- distance from centroid to edge sample strip
    ASYM_THR        = 0.40  # [0,1]-- edge asymmetry above this -> boundary
    BRIGHT_THR      = 120   # gray -- fallback: track surface brightness
    MAX_JUMP        = 40    # px   -- velocity gate threshold (tiny blobs only)
    MEDIAN_K        = 3     # frames - median window length
    WARMUP_FRAMES   = 8     # frames -- ignore prev_cx, pick closest to image center

    def __init__(self, debug=False):
        self.cameraWidth  = 320
        self.cameraHeight = 240
        self.prev_cx      = None
        self.history      = deque(maxlen=self.MEDIAN_K)
        self.warmup       = self.WARMUP_FRAMES
        self.debug        = debug
        self.debug_frame  = None

    # main detection 

    def detect_center_line(self, image):
        """
        :param image: BGR image (OpenCV), expected 320x240.
        :return: (cx, cy) pixel coordinates in original image space.
        """
        h, w = image.shape[:2]

        # S1: ROI
        y_start = (2 * h) // 3
        roi     = image[y_start:h, :]

        # S2: Grayscale
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # S3: Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # S4: Canny edges (used for contour classification in S7)
        edges   = cv2.Canny(blurred, 30, 90)

        # S5: Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # S6: Morphological OPEN
        kernel  = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # S7: Smart selection — returns (cx, cy, area_of_picked_contour)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        raw_cx, raw_cy, raw_area = self._select_best_contour(
            contours, blurred, edges, w, h, y_start)

        # S8: Velocity gate — disabled during warmup and for significant
        # detections.  Only tiny/noise blobs are gated; a real track line
        # (area >= SIGNIFICANT_AREA) is always trusted even after a large
        # jump so the tracker can recover from turns.
        if (self.warmup == 0
                and self.prev_cx is not None
                and raw_area < self.SIGNIFICANT_AREA
                and abs(raw_cx - self.prev_cx) > self.MAX_JUMP):
            raw_cx = int(self.prev_cx)

        # S9: Median filter
        self.history.append(raw_cx)
        smoothed_cx  = int(np.median(self.history))
        self.prev_cx = float(smoothed_cx)
        if self.warmup > 0:
            self.warmup -= 1

        cx = max(0, min(w - 1, smoothed_cx))
        cy = raw_cy

        if self.debug:
            self.debug_frame = self._build_debug(
                roi, gray, blurred, edges, binary, cleaned, contours,
                cx, cy, y_start, w, h)

        return (cx, cy)

    #  helpers 

    def _contour_centroid(self, cnt):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        return M['m10'] / M['m00'], M['m01'] / M['m00']

    def _edge_asymmetry(self, edges, cx_cnt, cy_cnt):
        """
        Measure Canny edge asymmetry on both sides of a contour centroid.
        Returns (asymmetry, L, R):  asym~0 -> symmetric -> centre line.
        """
        h, w  = edges.shape[:2]
        y     = int(np.clip(cy_cnt, 0, h - 1))
        y0, y1 = max(0, y - 10), min(h, y + 11)
        SW    = 5

        xl = int(np.clip(cx_cnt - self.SAMPLE_OFFSET, SW, w - SW - 1))
        xr = int(np.clip(cx_cnt + self.SAMPLE_OFFSET, SW, w - SW - 1))

        L = int(edges[y0:y1, xl - SW : xl + SW].sum()) // 255
        R = int(edges[y0:y1, xr - SW : xr + SW].sum()) // 255

        asym = abs(L - R) / (L + R + 1)
        return asym, L, R

    def _is_flanked_bright(self, blurred, cx_cnt, cy_cnt):
        """Brightness fallback: both sides bright -> centre line."""
        h, w = blurred.shape[:2]
        y    = int(np.clip(cy_cnt, 0, h - 1))
        xl   = int(np.clip(cx_cnt - self.SAMPLE_OFFSET, 0, w - 1))
        xr   = int(np.clip(cx_cnt + self.SAMPLE_OFFSET, 0, w - 1))

        def win_mean(x):
            r0, r1 = max(0, y - 2), min(h, y + 3)
            c0, c1 = max(0, x - 2), min(w, x + 3)
            p = blurred[r0:r1, c0:c1]
            return float(p.mean()) if p.size else 0.0

        return win_mean(xl) > self.BRIGHT_THR and win_mean(xr) > self.BRIGHT_THR

    def _classify_center(self, blurred, edges, cx_cnt, cy_cnt):
        """True if this contour is likely the centre line."""
        asym, L, R = self._edge_asymmetry(edges, cx_cnt, cy_cnt)
        if L + R >= 4:
            return asym < self.ASYM_THR
        return self._is_flanked_bright(blurred, cx_cnt, cy_cnt)

    #  contour selection 

    def _select_best_contour(self, contours, blurred, edges, w, h, y_start):
        """
        Returns (cx, cy, area) where area is the area of the chosen contour
        (0 if the fallback hold value was used).
        """
        img_cx    = w / 2.0
        search_cx = self.prev_cx if self.prev_cx is not None else img_cx

        # Build all_valid as (cx_f, cy_f, area, cnt), sorted by x.
        # Exclude tiny noise (< MIN_AREA) and large merged blobs (> MAX_AREA).
        all_valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            c = self._contour_centroid(cnt)
            if c is None:
                continue
            all_valid.append((c[0], c[1], area, cnt))
        all_valid.sort(key=lambda v: v[0])

        # Separate real track lines from noise/engraving fragments.
        sig = [v for v in all_valid if v[2] >= self.SIGNIFICANT_AREA]

        #  Warmup: ignore prev_cx, always pick closest to image centre ────────
        # Prefer significant blobs; fall back to all_valid only if none exist.
        if self.warmup > 0:
            pool = sig if sig else all_valid
            if pool:
                chosen = min(pool, key=lambda v: abs(v[0] - img_cx))
                return (max(0, min(w - 1, int(round(chosen[0])))),
                        y_start + int(round(chosen[1])),
                        chosen[2])
            return (w // 2, y_start + (h - y_start) // 2, 0)

        #  Normal operation: window on significant blobs 
        # If the window contains at least one significant blob, use those.
        # If not (line moved out or only noise inside window), jump directly
        # to the nearest significant blob to re-acquire after a turn.
        # Only if no significant blobs exist at all do we fall back to tiny ones.
        if self.prev_cx is not None:
            sig_in_win = [v for v in sig
                          if abs(v[0] - self.prev_cx) <= self.WINDOW_HALF]
            if sig_in_win:
                valid = sig_in_win
            elif sig:
                # No sig inside window — pick the globally closest significant
                # blob.  This is the turn-recovery path.
                closest = min(sig, key=lambda v: abs(v[0] - self.prev_cx))
                valid   = [closest]
            else:
                # No significant blobs anywhere; fall back to tiny contours.
                windowed = [v for v in all_valid
                            if abs(v[0] - self.prev_cx) <= self.WINDOW_HALF]
                valid = windowed if windowed else all_valid
        else:
            valid = sig if sig else all_valid

        chosen_cx = chosen_cy = chosen_area = None

        if len(valid) == 0:
            pass

        elif len(valid) == 1:
            chosen_cx, chosen_cy, chosen_area, _ = valid[0]

        elif len(valid) == 2:
            cx0, cy0, a0, _ = valid[0]
            cx1, cy1, a1, _ = valid[1]
            is_c0 = self._classify_center(blurred, edges, cx0, cy0)
            is_c1 = self._classify_center(blurred, edges, cx1, cy1)

            if is_c0 and not is_c1:
                chosen_cx, chosen_cy, chosen_area = cx0, cy0, a0
            elif is_c1 and not is_c0:
                chosen_cx, chosen_cy, chosen_area = cx1, cy1, a1
            else:
                best_score = float('inf')
                for (cx_v, cy_v, a_v, _) in valid:
                    dist_center = abs(cx_v - img_cx)    / w
                    dist_prev   = abs(cx_v - search_cx) / w
                    score = 0.4 * dist_center + 0.6 * dist_prev
                    if score < best_score:
                        best_score = score
                        chosen_cx, chosen_cy, chosen_area = cx_v, cy_v, a_v

        else:
            mid = len(valid) // 2
            chosen_cx, chosen_cy, chosen_area, _ = valid[mid]

        if chosen_cx is not None:
            return (max(0, min(w - 1, int(round(chosen_cx)))),
                    y_start + int(round(chosen_cy)),
                    chosen_area)

        cx = int(round(self.prev_cx)) if self.prev_cx is not None else w // 2
        cy = y_start + (h - y_start) // 2
        return (cx, cy, 0)

    #  debug visualization 

    def _build_debug(self, roi, gray, blurred, edges, binary, cleaned,
                     contours, cx, cy, y_start, w, h):
        """
        7-tile strip: ROI | gray | blur | Canny | Otsu | open | result+window
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

        # Tile 7: contours (colour-coded by size) + search window + detected centre
        result_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA:
                continue
            col = (0, 200, 0) if area >= self.SIGNIFICANT_AREA else (0, 180, 180)
            cv2.drawContours(result_vis, [cnt], -1, col, 1)
        if self.prev_cx is not None:
            wx0 = max(0, int(self.prev_cx - self.WINDOW_HALF))
            wx1 = min(w - 1, int(self.prev_cx + self.WINDOW_HALF))
            cv2.rectangle(result_vis, (wx0, 0), (wx1, roi_h - 1), (255, 128, 0), 1)
        cy_roi = cy - y_start
        if 0 <= cy_roi < roi_h:
            cv2.drawMarker(result_vis, (cx, cy_roi),
                           (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        n_sig = sum(1 for c in contours
                    if self.MIN_AREA <= cv2.contourArea(c) <= self.MAX_AREA
                    and cv2.contourArea(c) >= self.SIGNIFICANT_AREA)
        cv2.putText(result_vis, f'sig={n_sig} cx={cx}',
                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(result_vis, f'sig={n_sig} cx={cx}',
                    (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        tiles = [
            labeled(roi,     'S1:ROI'),
            labeled(gray,    'S2:gray'),
            labeled(blurred, 'S3:blur'),
            labeled(edges,   'S4:Canny'),
            labeled(binary,  'S5:Otsu'),
            labeled(cleaned, 'S6:open'),
            result_vis,
        ]

        resized = []
        for t in tiles:
            sc = roi_h / t.shape[0]
            nw = max(1, int(t.shape[1] * sc))
            resized.append(cv2.resize(t, (nw, roi_h)))
        return np.hstack(resized)
