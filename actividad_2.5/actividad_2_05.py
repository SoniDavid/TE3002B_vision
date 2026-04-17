import numpy as np
import cv2


class TrafficLightDetection:
    # HSV color ranges for each bulb color
    RED_LO1  = np.array([  0,  50,  80])
    RED_HI1  = np.array([ 10, 255, 255])
    RED_LO2  = np.array([160,  50,  80])
    RED_HI2  = np.array([180, 255, 255])
    YEL_LO   = np.array([ 15, 100, 150])
    YEL_HI   = np.array([ 35, 255, 255])
    GRN_LO   = np.array([ 40,  80, 120])
    GRN_HI   = np.array([ 85, 255, 255])

    # Blob quality thresholds (used when searching inside a housing)
    MIN_AREA        = 8
    DETECT_THRESH   = 8
    MIN_CIRCULARITY = 0.30
    MIN_MEAN_V      = 140

    # Housing detection parameters
    HOUSING_V_MAX      = 70    # housing pixels are darker than this
    HOUSING_MIN_AREA   = 80    # px² — smallest housing we accept
    HOUSING_ASPECT_MIN = 1.5   # height/width — housing is tall
    HOUSING_ASPECT_MAX = 6.0
    HOUSING_BULB_V_MIN = 150   # a lit bulb inside the housing must be this bright
    HOUSING_PAD        = 4     # px — padding around housing when cropping color ROI

    def __init__(self, debug=False):
        self.cameraWidth  = 320
        self.cameraHeight = 240
        self.debug        = debug
        self.debug_frame  = None
        self._kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_k     = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))

    # ── Stage 1: housing detector ────────────────────────────────────────────

    def _find_housing(self, v_channel):
        """Return (x, y, w, h) of the most likely traffic-light housing, or None.

        Strategy: threshold dark pixels → find tall rectangular contours →
        verify that at least one bright spot (a lit bulb) exists inside.
        """
        dark = (v_channel < self.HOUSING_V_MAX).astype(np.uint8) * 255
        # Close fills the small gaps between bulb holes in the housing
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._close_k)

        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.HOUSING_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0:
                continue
            aspect = h / w
            if not (self.HOUSING_ASPECT_MIN < aspect < self.HOUSING_ASPECT_MAX):
                continue
            # Verify: housing must contain at least one bright bulb-level pixel
            roi_v = v_channel[y:y + h, x:x + w]
            if roi_v.max() < self.HOUSING_BULB_V_MIN:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        return best

    # ── Stage 2: color classifier ─────────────────────────────────────────────

    def _best_area(self, mask, v_channel):
        """Area of the largest bright circular-ish contour in mask, or 0."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.MIN_AREA:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter * perimeter) < self.MIN_CIRCULARITY:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.3 < (w / h if h > 0 else 0) < 3.0):
                continue
            blob_mask = np.zeros(v_channel.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            if cv2.mean(v_channel, mask=blob_mask)[0] < self.MIN_MEAN_V:
                continue
            if area > best:
                best = area
        return best

    def _color_masks(self, hsv):
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.RED_LO1, self.RED_HI1),
            cv2.inRange(hsv, self.RED_LO2, self.RED_HI2),
        )
        yel_mask = cv2.inRange(hsv, self.YEL_LO, self.YEL_HI)
        grn_mask = cv2.inRange(hsv, self.GRN_LO, self.GRN_HI)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self._kernel)
        yel_mask = cv2.morphologyEx(yel_mask, cv2.MORPH_OPEN, self._kernel)
        grn_mask = cv2.morphologyEx(grn_mask, cv2.MORPH_OPEN, self._kernel)
        return red_mask, yel_mask, grn_mask

    def _score_colors(self, hsv, v):
        """Return (areas_dict, red_mask, yel_mask, grn_mask) over hsv/v arrays."""
        red_mask, yel_mask, grn_mask = self._color_masks(hsv)
        areas = {
            "red":    self._best_area(red_mask, v),
            "yellow": self._best_area(yel_mask, v),
            "green":  self._best_area(grn_mask, v),
        }
        return areas, red_mask, yel_mask, grn_mask

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_state(self, image):
        """
        Detecta el estado del semáforo y lo reporta como texto.
        :return: String con alguno de los siguientes contenidos: green, yellow, red, none
        """
        h, w = image.shape[:2]
        roi = image[:(2 * h) // 3, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v   = hsv[:, :, 2]

        housing = self._find_housing(v)

        if housing is not None:
            # Stage 2a: restrict color search to the housing region
            hx, hy, hw, hh = housing
            rh, rw = roi.shape[:2]
            x1 = max(0, hx - self.HOUSING_PAD)
            y1 = max(0, hy - self.HOUSING_PAD)
            x2 = min(rw, hx + hw + self.HOUSING_PAD)
            y2 = min(rh, hy + hh + self.HOUSING_PAD)
            areas, red_mask, yel_mask, grn_mask = self._score_colors(
                hsv[y1:y2, x1:x2], v[y1:y2, x1:x2]
            )
        else:
            # Stage 2b: fallback — search full ROI
            areas, red_mask, yel_mask, grn_mask = self._score_colors(hsv, v)

        best_color, best_area = max(areas.items(), key=lambda kv: kv[1])
        state = best_color if best_area >= self.DETECT_THRESH else "none"

        if self.debug:
            self._build_debug(roi, red_mask, yel_mask, grn_mask,
                              state, areas, housing)

        return state

    # ── Debug view ────────────────────────────────────────────────────────────

    def _build_debug(self, roi, red_mask, yel_mask, grn_mask,
                     state, areas, housing=None):
        def to_bgr(mask):
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        roi_vis = roi.copy()
        if housing is not None:
            hx, hy, hw, hh = housing
            cv2.rectangle(roi_vis, (hx, hy), (hx + hw, hy + hh), (0, 255, 255), 1)

        # Pad masks to match roi width if they were cropped
        def pad(mask):
            if mask.shape[1] != roi.shape[1]:
                out = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                return out
            return mask

        row1 = np.hstack([roi_vis,
                          to_bgr(pad(red_mask)),
                          to_bgr(pad(yel_mask)),
                          to_bgr(pad(grn_mask))])

        info = np.zeros((40, row1.shape[1], 3), dtype=np.uint8)
        tag  = "HSG" if housing is not None else "FBK"
        label = (f"[{tag}] state={state}  "
                 f"R={areas['red']:.0f}  Y={areas['yellow']:.0f}  G={areas['green']:.0f}")
        cv2.putText(info, label, (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        self.debug_frame = np.vstack([row1, info])
