"""
debug_pipeline.py — Visualise every stage of the CenterLineDetector pipeline.

Usage:
  python3 debug_pipeline.py video [path/to/recording.avi]   (default)
  python3 debug_pipeline.py sim   [mode]

Composite PNG layout (written to /tmp/pipeline_debug/):
  Row 0: full frame with final cx (green), track_mid_x (yellow), search window (cyan)
  Row 1 (ROI): blur | track-mask overlay | tophat_small | tophat_large
  Row 2 (ROI): band (=th_l-th_s) | band_track | OLD Otsu | H channel
  Row 3 (hist): band_track hist | OLD Otsu hist | (smoothed in colour)
"""

import sys, os, time
import numpy as np
import cv2

OUT_DIR = '/tmp/pipeline_debug'
os.makedirs(OUT_DIR, exist_ok=True)

SE_SMALL   = 6
SE_LARGE   = 28
SEARCH_WIN = 107
PEAK_WIN   = 35

def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

def label(img, text, col=(255,255,255)):
    cv2.putText(img, text, (2,11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
    cv2.putText(img, text, (2,11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col,     1)

def hist_canvas(pairs, w, h=70):
    c = np.zeros((h, w, 3), np.uint8)
    mx = max((a.max() for a,_ in pairs), default=1.0); mx = max(mx,1.0)
    for arr, col in pairs:
        for x in range(w):
            bar = int(arr[x]/mx*(h-2))
            if bar > 0: cv2.line(c,(x,h-1),(x,h-1-bar),col,1)
    return c

def rh(img, th):
    sc = th/img.shape[0]; return cv2.resize(img,(int(img.shape[1]*sc),th))


def run_debug(frame, idx):
    h, w   = frame.shape[:2]
    roi_y  = (3*h)//4
    roi    = frame[roi_y:h,:]
    roi_h  = roi.shape[0]
    kern   = cv2.getGaussianKernel(15,5).flatten()
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    # S1
    blurred = cv2.GaussianBlur(roi,(5,5),1.4)
    s1 = to_bgr(blurred); label(s1,'S1: blur')

    # S2
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    tmask   = cv2.inRange(hsv, np.array([5,15,60],np.uint8),
                                np.array([35,180,230],np.uint8))
    k3      = np.ones((3,3),np.uint8)
    tmask   = cv2.morphologyEx(tmask, cv2.MORPH_CLOSE, k3, iterations=2)
    tmask   = cv2.morphologyEx(tmask, cv2.MORPH_OPEN,  k3, iterations=1)

    s2 = blurred.copy()
    s2[tmask>0] = (s2[tmask>0]*0.4 + np.array([255,255,0])*0.6).astype(np.uint8)
    label(s2, f'S2: track ({tmask.sum()//255}px)')

    h_ch = cv2.split(hsv)[0]
    s_hue = cv2.applyColorMap(h_ch, cv2.COLORMAP_HSV); label(s_hue,'H channel')

    # S3 band-pass
    gray  = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    se_s  = cv2.getStructuringElement(cv2.MORPH_RECT,(SE_SMALL,1))
    se_l  = cv2.getStructuringElement(cv2.MORPH_RECT,(SE_LARGE,1))
    th_s  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_s)
    th_l  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_l)
    band  = np.clip(th_l.astype(np.int16) - th_s.astype(np.int16), 0, 255).astype(np.uint8)

    band_track = band.copy(); band_track[tmask==0] = 0

    # Normalised display tiles
    def norm_tile(img, cmap=cv2.COLORMAP_INFERNO):
        n = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(n, cmap)

    s3_ths  = norm_tile(th_s); label(s3_ths, f'tophat_small (se={SE_SMALL})')
    s3_thl  = norm_tile(th_l); label(s3_thl, f'tophat_large (se={SE_LARGE})')
    s3_band = norm_tile(band); label(s3_band, f'band=th_l-th_s ({band_track.sum()//255}px on track)')
    s3_bt   = to_bgr(band_track); label(s3_bt, f'band_track ({band_track.sum()//255}px)')

    # OLD Otsu comparison
    _, bin_old = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    line_old   = cv2.bitwise_and(bin_old, bin_old, mask=tmask)
    s_old      = to_bgr(line_old); label(s_old, f'OLD Otsu ({line_old.sum()//255}px)')

    # S4: track mid_x
    mids = []
    for r in range(roi_h):
        xs = np.where(tmask[r]>0)[0]
        if len(xs)>5: mids.append(float(xs[0]+xs[-1])/2.0)
    mid_x = float(np.median(mids)) if len(mids)>=max(1,roi_h//4) else w/2.0

    # S5: detection
    def peak_cx(col_sig, min_sig=1.0):
        sm  = np.convolve(col_sig, kern, mode='same')
        lo  = max(0,int(mid_x)-SEARCH_WIN); hi = min(w,int(mid_x)+SEARCH_WIN)
        sc  = sm.copy(); sc[:lo]=0; sc[hi:]=0
        if sc.max()<min_sig: return None, sm
        pk  = int(np.argmax(sc))
        cl  = max(0,pk-PEAK_WIN); cr = min(w,pk+PEAK_WIN)
        reg = sc.copy(); reg[:cl]=0; reg[cr:]=0
        tot = reg.sum()
        if tot<1e-6: return float(pk), sm
        return float(np.dot(np.arange(w,dtype=np.float32),reg)/tot), sm

    col_band = np.sum(band_track, axis=0).astype(np.float32)
    col_old  = np.sum(line_old,   axis=0).astype(np.float32)

    raw_cx, sm_band = peak_cx(col_band, min_sig=100.0)
    method = 'band-pass'

    if raw_cx is None:
        gray_eq    = clahe.apply(gray)
        _, bin_cl  = cv2.threshold(gray_eq,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_track  = cv2.bitwise_and(bin_cl,bin_cl,mask=tmask)
        col_cl     = np.sum(bin_track, axis=0).astype(np.float32)
        raw_cx, _  = peak_cx(col_cl, min_sig=255.0*3)
        method     = 'CLAHE+Otsu'

    if raw_cx is None:
        M = cv2.moments(tmask)
        if M['m00']>300: raw_cx = M['m10']/M['m00']; method='track centroid'

    if raw_cx is None: raw_cx = w/2.0; method='center default'

    cx = max(0,min(w-1,int(round(raw_cx))))
    cy = roi_y+roi_h//2

    # histograms
    _, sm_old = peak_cx(col_old, min_sig=1.0)
    hc_band = hist_canvas([(col_band,(80,80,80)),(sm_band,(0,255,128))], w)
    label(hc_band, f'hist: band_track (cyan=smooth)  min_sig={50*roi_h}')
    hc_old  = hist_canvas([(col_old,(80,80,80)),(sm_old,(0,128,255))], w)
    label(hc_old,  'hist: OLD Otsu   (orange=smooth)')

    # annotate full frame
    ri = frame.copy()
    cv2.line(ri,(0,roi_y),(w,roi_y),(0,255,255),1)
    # search window on ROI boundary
    lo_s=max(0,int(mid_x)-SEARCH_WIN); hi_s=min(w,int(mid_x)+SEARCH_WIN)
    cv2.line(ri,(lo_s,roi_y),(lo_s,h-1),(255,255,0),1)
    cv2.line(ri,(hi_s,roi_y),(hi_s,h-1),(255,255,0),1)
    cv2.line(ri,(int(mid_x),roi_y),(int(mid_x),h-1),(0,200,200),1)  # track mid (teal)
    cv2.drawMarker(ri,(cx,cy),(0,255,0),cv2.MARKER_CROSS,20,2)
    cv2.line(ri,(cx,roi_y),(cx,h-1),(0,200,0),1)
    label(ri, f'cx={cx} [{method}]  mid_x={mid_x:.0f}  win=[{lo_s},{hi_s}]')

    # composite
    RW = 4*w
    row0 = cv2.resize(ri,(RW,int(h*RW/w)))
    row1 = np.hstack([rh(s1,roi_h),rh(s2,roi_h),rh(s3_ths,roi_h),rh(s3_thl,roi_h)])
    row2 = np.hstack([rh(s3_band,roi_h),rh(s3_bt,roi_h),rh(s_old,roi_h),rh(s_hue,roi_h)])
    def wh(img): return cv2.resize(img,(RW,img.shape[0]))
    hist_block = np.vstack([wh(hc_band),wh(hc_old)])
    comp = np.vstack([row0,row1,row2,hist_block])
    path = os.path.join(OUT_DIR,f'dbg_{idx:05d}.png')
    cv2.imwrite(path,comp)

    print(f'[{idx:5d}] cx={cx:3d}  method={method:<18s}  '
          f'band={band_track.sum()//255:4d}  '
          f'old={line_old.sum()//255:4d}  '
          f'mid_x={mid_x:.0f}  win=[{lo_s},{hi_s}]')
    return cx,cy


def from_video(path, max_frames=20):
    cap=cv2.VideoCapture(path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs=sorted(set(max(0,min(total-1,int(total*p/(max_frames-1))))
                    for p in range(max_frames)))
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ret,f=cap.read()
        if not ret: continue
        f=cv2.resize(f,(320,240),interpolation=cv2.INTER_LANCZOS4)
        run_debug(f,idx)
    cap.release()

def from_simulator(mode=2, n=30):
    sys.path.append('/home/soni/Documents/classes/IRS_6to/TE3002B_vision/3DGS_Simulator/Interface')
    import grpc,te3002b_pb2,te3002b_pb2_grpc,google.protobuf.empty_pb2
    ch=grpc.insecure_channel('127.0.0.1:7072')
    st=te3002b_pb2_grpc.TE3002BSimStub(ch)
    cfg=te3002b_pb2.ConfigurationData()
    cfg.resetRobot=True;cfg.mode=mode;cfg.cameraWidth=320
    cfg.cameraHeight=240;cfg.resetCamera=False;cfg.scene=2026
    st.SetConfiguration(cfg);cfg.resetRobot=False
    time.sleep(0.25);st.SetConfiguration(cfg)
    req=google.protobuf.empty_pb2.Empty()
    for _ in range(n):
        res=st.GetImageFrame(req)
        img=cv2.imdecode(np.frombuffer(res.data,np.uint8),cv2.IMREAD_COLOR)
        if img is None: continue
        img=cv2.resize(img,(320,240),interpolation=cv2.INTER_LANCZOS4)
        run_debug(img,int(res.seq)); time.sleep(0.05)

if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'video'
    if arg=='sim':
        from_simulator(mode=int(sys.argv[2]) if len(sys.argv)>2 else 2,n=30)
    else:
        vid=sys.argv[2] if len(sys.argv)>2 else \
            '/home/soni/Documents/classes/IRS_6to/TE3002B_vision/actividad_2.4/recording_mode2_20260414_155644.avi'
        from_video(vid,max_frames=20)
