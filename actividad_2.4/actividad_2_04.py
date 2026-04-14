from collections import deque
import numpy as np
import cv2


class CenterLineDetector:
    """
    Robust center-line detector for the TE3002B line-following track.

    Pipeline (clase 3 & 5 techniques):

      S1  ROI crop (bottom 1/4) + Gaussian blur
      S2  HSV inRange → track surface mask     (clase 3, slides 15, 27)
          + MORPH_CLOSE/OPEN cleanup            (clase 5, slides 6-8)
      S3  Morphological band-pass filter        (clase 5, morphology)
            band = BLACKHAT(gray, se_large) − BLACKHAT(gray, se_small)
          where se_small < center_line_width < se_large.
          • BLACKHAT highlights features narrower than its SE.
          • Features narrower than se_small (thin panel borders, ~3-5 px)
            appear in BOTH tophat images with equal magnitude → difference ≈ 0.
          • Features wider than se_large (large dark areas) produce near-zero
            response in both → difference ≈ 0.
          • Only features whose width falls BETWEEN se_small and se_large
            (the center line, ~10-20 px) produce non-zero difference → isolated.
      S4  Estimate track geometric centre from mask row-midpoints → track_mid_x.
          Restricts the column-histogram search to ±SEARCH_WIN of track_mid_x,
          preventing far panel borders from hijacking the result.
      S5  Peak-centred column histogram weighted centroid within the search band:
            argmax(smooth col-hist) → dominant peak → weighted centroid ±PEAK_WIN.
      S6  Fallback chain:
            1. Band-pass peak near track centre          (primary)
            2. CLAHE + Otsu darkest column near centre   (secondary)
            3. Track mask centroid                        (tertiary)
            4. Previous smoothed position                 (hold)
            5. Image centre                               (last resort)
      S7  Median filter temporal smoothing (K=7, tolerates up to 3 wrong frames)
    """

    # Band-pass SE widths (px). Center line observed to be ~10-20 px in ROI.
    SE_SMALL   = 6     # must be < center_line_width: suppresses thin panel borders
    SE_LARGE   = 28    # must be > center_line_width: passes center line

    SEARCH_WIN = 107   # ±px around estimated track center (= w//3 for w=320)
    PEAK_WIN   = 35    # ±px for centroid refinement around detected peak

    def __init__(self):
        self.cameraWidth  = 320
        self.cameraHeight = 240
        self.prev_cx      = None             # smoothed output (search anchor)
        self.history      = deque(maxlen=7)  # raw detections buffer for median filter
        self._smooth_kern = cv2.getGaussianKernel(15, 5).flatten()
        self._clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _track_mid_x(self, track_mask, w, roi_h):
        """Estimate horizontal centre of the track from the mask row-midpoints."""
        mids = []
        for r in range(roi_h):
            xs = np.where(track_mask[r] > 0)[0]
            if len(xs) > 5:
                mids.append(float(xs[0] + xs[-1]) / 2.0)
        return float(np.median(mids)) if len(mids) >= max(1, roi_h // 4) else w / 2.0

    def _peak_cx(self, col_signal, track_mid_x, w, min_signal=1.0):
        """
        Smooth col_signal, restrict search to ±SEARCH_WIN of track_mid_x,
        find argmax peak, then refine with weighted centroid ±PEAK_WIN.
        Returns None when signal is too weak.
        """
        smooth = np.convolve(col_signal, self._smooth_kern, mode='same')

        # Restrict to search band around track centre
        lo = max(0, int(track_mid_x) - self.SEARCH_WIN)
        hi = min(w, int(track_mid_x) + self.SEARCH_WIN)
        search = smooth.copy()
        search[:lo] = 0.0
        search[hi:]  = 0.0

        if search.max() < min_signal:
            return None

        peak  = int(np.argmax(search))
        cl    = max(0, peak - self.PEAK_WIN)
        cr    = min(w, peak + self.PEAK_WIN)
        region = search.copy()
        region[:cl] = 0.0
        region[cr:]  = 0.0
        total = region.sum()
        if total < 1e-6:
            return float(peak)
        return float(np.dot(np.arange(w, dtype=np.float32), region) / total)

    # ── main detection ────────────────────────────────────────────────────────

    def detect_center_line(self, image):
        """
        Detect the center line of the track in the bottom 1/4 of `image`.

        :param image: BGR image (OpenCV).
        :return: (cx, cy) in original image coordinates.
        """
        h, w = image.shape[:2]

        # ── S1: ROI + blur ────────────────────────────────────────────────────
        roi_y   = (3 * h) // 4
        roi     = image[roi_y:h, :]
        roi_h   = roi.shape[0]
        blurred = cv2.GaussianBlur(roi, (5, 5), 1.4)

        # ── S2: Track surface mask ────────────────────────────────────────────
        hsv         = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_track = np.array([ 5,  15,  60], dtype=np.uint8)
        upper_track = np.array([35, 180, 230], dtype=np.uint8)
        track_mask  = cv2.inRange(hsv, lower_track, upper_track)
        k3          = np.ones((3, 3), np.uint8)
        track_mask  = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, k3, iterations=2)
        track_mask  = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN,  k3, iterations=1)

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # ── S3: Morphological band-pass ───────────────────────────────────────
        # Large SE → highlights features narrower than SE_LARGE (includes center line)
        # Small SE → highlights features narrower than SE_SMALL (panel borders only)
        # Difference → only features with width in [SE_SMALL, SE_LARGE] survive
        se_s   = cv2.getStructuringElement(cv2.MORPH_RECT, (self.SE_SMALL, 1))
        se_l   = cv2.getStructuringElement(cv2.MORPH_RECT, (self.SE_LARGE, 1))
        th_s   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_s).astype(np.int16)
        th_l   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_l).astype(np.int16)
        band   = np.clip(th_l - th_s, 0, 255).astype(np.uint8)

        # Zero out non-track pixels
        band_track              = band.copy()
        band_track[track_mask == 0] = 0

        # ── S4: Track geometric centre ────────────────────────────────────────
        mid_x = self._track_mid_x(track_mask, w, roi_h)

        # ── S5: Peak-centred column histogram ─────────────────────────────────
        # Anchor search window 70% to previous smoothed position, 30% to geometric
        # track centre — keeps the window near the last confirmed line location
        # during turns, preventing distant panel borders from being selected.
        search_cx = (0.7 * self.prev_cx + 0.3 * mid_x) if self.prev_cx is not None else mid_x
        col_band  = np.sum(band_track, axis=0).astype(np.float32)
        raw_cx    = self._peak_cx(col_band, search_cx, w, min_signal=100.0)

        # ── S6: Fallback chain ────────────────────────────────────────────────
        if raw_cx is None:               # Level 2: CLAHE + Otsu darkest column
            gray_eq    = self._clahe.apply(gray)
            _, bin_cl  = cv2.threshold(gray_eq, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            bin_track  = cv2.bitwise_and(bin_cl, bin_cl, mask=track_mask)
            col_clahe  = np.sum(bin_track, axis=0).astype(np.float32)
            raw_cx     = self._peak_cx(col_clahe, search_cx, w, min_signal=255.0 * 3)

        if raw_cx is None:               # Level 3: track centroid
            M = cv2.moments(track_mask)
            if M['m00'] > 300:
                raw_cx = M['m10'] / M['m00']

        if raw_cx is None and self.prev_cx is not None:   # Level 4: hold
            raw_cx = self.prev_cx

        if raw_cx is None:               # Level 5: image centre
            raw_cx = w / 2.0

        # ── S7: Median filter temporal smoothing ──────────────────────────────
        # K=7 buffer: immune to up to 3 consecutive wrong detections — covers the
        # 3-5 frame drift events observed at turns in the recorded sessions.
        self.history.append(raw_cx)
        smoothed_cx  = float(np.median(self.history))
        self.prev_cx = smoothed_cx

        cx = max(0, min(w - 1, int(round(smoothed_cx))))
        cy = roi_y + roi_h // 2
        return (cx, cy)
