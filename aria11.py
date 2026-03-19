import streamlit as st
import numpy as np
import math, random, time, os, warnings, json, hashlib
from datetime import datetime
from collections import deque, Counter
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import List, Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ── OpenCV ──────────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── MediaPipe (FIX 1: clean single import, no double-import confusion) ──────
MP_OK = False
mp    = None
try:
    import mediapipe as mp
    _test = mp.solutions.face_mesh   # verify sub-module exists
    MP_OK = True
except Exception:
    mp = None

# ── streamlit-webrtc ─────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

# ── PyTorch (optional) ───────────────────────────────────────────────────────
TORCH_OK = False
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    pass

import threading

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#F0F4F8"
SURFACE = "#FFFFFF"
BORDER  = "#CBD5E0"
ACCENT  = "#2B6CB0"
ACCENT2 = "#553C9A"
DANGER  = "#C53030"
WARN    = "#B45309"
SAFE    = "#276749"
TEXT    = "#1A202C"
MUTED   = "#4A5568"
LIGHT   = "#E2E8F0"

# ── Task constants ────────────────────────────────────────────────────────────
GRID_ROWS    = 4
GRID_COLS    = 4
GRID_CELLS   = GRID_ROWS * GRID_COLS
MEM_TRIALS   = 5
MEM_SHOW_SEC   = 8.0
MEM_RECALL_SEC = 12.0

SACC_TRIALS   = 5
SACC_FIX_SEC  = 1.4
SACC_STIM_SEC = 2.8

EYE_CAL_SEC  = 30
RESULTS_DIR  = "neuroscan_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "fixation_duration","saccade_frequency",
    "scan_path_length","gaze_variability",
    "reaction_time","accuracy",
    "antisaccade_error_rate","antisaccade_rt",
]

CELL_COLORS = [
    "#E53E3E","#2B6CB0","#276749","#B45309",
    "#553C9A","#2C7A7B","#C05621","#2D3748",
]

st.set_page_config(
    page_title="NeuroScan LBD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
    html,body,[data-testid="stAppViewContainer"]{{background:{BG}!important;color:{TEXT};font-family:'Inter',sans-serif;}}
    [data-testid="stHeader"]{{background:transparent!important;box-shadow:none;}}
    .main .block-container{{padding:1.1rem 1.8rem;max-width:1320px;}}
    .stButton>button{{background:{ACCENT}!important;color:#fff!important;font-weight:600!important;
        border:none!important;border-radius:6px!important;padding:0.5rem 1.4rem!important;
        font-size:0.86rem!important;transition:background 0.12s!important;}}
    .stButton>button:hover{{background:#2C5282!important;}}
    .card{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
        padding:1rem 1.2rem;margin-bottom:0.65rem;box-shadow:0 1px 3px rgba(0,0,0,.05);}}
    .card-blue{{background:linear-gradient(135deg,#EBF8FF,#F7FAFC);border:1px solid #BEE3F8;
        border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.65rem;}}
    .badge{{display:inline-flex;align-items:center;padding:1px 7px;border-radius:10px;
        font-size:0.67rem;font-family:'JetBrains Mono',monospace;font-weight:500;letter-spacing:.05em;}}
    .b-green{{background:#F0FFF4;color:{SAFE};border:1px solid #9AE6B4;}}
    .b-red{{background:#FFF5F5;color:{DANGER};border:1px solid #FEB2B2;}}
    .b-amber{{background:#FFFAF0;color:{WARN};border:1px solid #F6AD55;}}
    .b-grey{{background:{LIGHT};color:{MUTED};border:1px solid {BORDER};}}
    .b-blue{{background:#EBF8FF;color:{ACCENT};border:1px solid #90CDF4;}}
    .v-high{{background:linear-gradient(135deg,#FFF5F5,#FFFAF0);border:1.5px solid #FEB2B2;border-radius:8px;padding:1.1rem 1.3rem;}}
    .v-mod{{background:linear-gradient(135deg,#FFFAF0,#FEFCBF);border:1.5px solid #F6AD55;border-radius:8px;padding:1.1rem 1.3rem;}}
    .v-low{{background:linear-gradient(135deg,#F0FFF4,#EBF8FF);border:1.5px solid #9AE6B4;border-radius:8px;padding:1.1rem 1.3rem;}}
    .sec{{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:{MUTED};font-weight:500;
        letter-spacing:.10em;text-transform:uppercase;margin-bottom:.45rem;}}
    hr.div{{border:none;border-top:1px solid {BORDER};margin:.65rem 0;}}
    div[data-testid="stMetricValue"]{{color:{TEXT}!important;font-family:'JetBrains Mono',monospace;font-size:1.2rem!important;}}
    div[data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:.72rem!important;}}
    .stProgress>div>div>div{{background:{ACCENT}!important;}}
    </style>""", unsafe_allow_html=True)


# ── Anomaly types & metadata ──────────────────────────────────────────────────

class AnomalyType(Enum):
    PROLONGED_FIXATION    = auto()
    SHORT_FIXATION        = auto()
    FIXATION_INSTABILITY  = auto()
    EXPRESS_SACCADE       = auto()
    SLOW_SACCADE          = auto()
    ANTI_SACCADE_ERROR    = auto()
    EXCESSIVE_HEAD_YAW    = auto()
    HEAD_TREMOR           = auto()
    ABSENT_BLINK          = auto()
    EXCESSIVE_BLINK_RATE  = auto()
    GAZE_DRIFT            = auto()
    SQUARE_WAVE_JERK      = auto()
    NYSTAGMUS_LIKE        = auto()
    OFF_SCREEN_GAZE       = auto()
    LOST_TRACKING         = auto()
    ANTICIPATORY_RESPONSE = auto()
    PERSEVERATIVE_SACCADE = auto()

LBD_BIOMARKERS = {
    AnomalyType.EXPRESS_SACCADE, AnomalyType.ANTI_SACCADE_ERROR,
    AnomalyType.PROLONGED_FIXATION, AnomalyType.SLOW_SACCADE,
    AnomalyType.SQUARE_WAVE_JERK, AnomalyType.NYSTAGMUS_LIKE,
    AnomalyType.ABSENT_BLINK, AnomalyType.ANTICIPATORY_RESPONSE,
    AnomalyType.PERSEVERATIVE_SACCADE,
}

SEVERITY_MAP = {
    AnomalyType.PROLONGED_FIXATION:"MEDIUM", AnomalyType.SHORT_FIXATION:"LOW",
    AnomalyType.FIXATION_INSTABILITY:"LOW",  AnomalyType.EXPRESS_SACCADE:"HIGH",
    AnomalyType.SLOW_SACCADE:"HIGH",         AnomalyType.ANTI_SACCADE_ERROR:"HIGH",
    AnomalyType.EXCESSIVE_HEAD_YAW:"MEDIUM", AnomalyType.HEAD_TREMOR:"HIGH",
    AnomalyType.ABSENT_BLINK:"MEDIUM",       AnomalyType.EXCESSIVE_BLINK_RATE:"LOW",
    AnomalyType.GAZE_DRIFT:"MEDIUM",         AnomalyType.SQUARE_WAVE_JERK:"HIGH",
    AnomalyType.NYSTAGMUS_LIKE:"CRITICAL",   AnomalyType.OFF_SCREEN_GAZE:"LOW",
    AnomalyType.LOST_TRACKING:"MEDIUM",      AnomalyType.ANTICIPATORY_RESPONSE:"HIGH",
    AnomalyType.PERSEVERATIVE_SACCADE:"HIGH",
}

CLINICAL_NOTES = {
    AnomalyType.EXPRESS_SACCADE:     "Saccade latency below 120ms - inhibition failure, key LBD frontal sign.",
    AnomalyType.SLOW_SACCADE:        "Peak velocity below 100 deg/s - brainstem integrity marker.",
    AnomalyType.PROLONGED_FIXATION:  "Dwelling over 900ms - visuospatial processing difficulty.",
    AnomalyType.ANTI_SACCADE_ERROR:  "Looked toward target - frontal executive dysfunction.",
    AnomalyType.SQUARE_WAVE_JERK:    "Conjugate saccadic intrusion - cerebellar or LBD sign.",
    AnomalyType.NYSTAGMUS_LIKE:      "Rhythmic gaze oscillation - requires neuro-ophthalmology review.",
    AnomalyType.ABSENT_BLINK:        "Reduced blink rate below 3 per minute - hypomimia, parkinsonism sign.",
    AnomalyType.HEAD_TREMOR:         "Rapid oscillatory head motion - parkinsonian rest tremor.",
    AnomalyType.GAZE_DRIFT:          "Slow drift from fixation target - smooth pursuit dysregulation.",
    AnomalyType.ANTICIPATORY_RESPONSE:"Response before stimulus onset - impulsivity or timing dysfunction.",
    AnomalyType.PERSEVERATIVE_SACCADE:"Repeated saccade same direction - perseveration, frontal sign.",
    AnomalyType.EXCESSIVE_HEAD_YAW:  "Head turned more than 25 degrees - compensatory gaze movement.",
    AnomalyType.LOST_TRACKING:       "Face or iris lost - possible tracking gap in data.",
    AnomalyType.FIXATION_INSTABILITY:"High fixation dispersion - poor visuospatial stability.",
    AnomalyType.OFF_SCREEN_GAZE:     "Gaze outside screen boundary - disengagement or distraction.",
    AnomalyType.EXCESSIVE_BLINK_RATE:"Blink rate above 30 per minute - anxiety or dry eye indicator.",
    AnomalyType.SHORT_FIXATION:      "Fixation under 80ms - fragmented scanning pattern.",
}


@dataclass
class AnomalyEvent:
    anomaly_id:       str
    anomaly_type:     str
    severity:         str
    timestamp:        float
    frame_index:      int
    clinical_note:    str
    is_lbd_biomarker: bool
    measured_value:   float
    threshold_value:  float
    unit:             str
    gaze_x:           float = 0.5
    gaze_y:           float = 0.5


@dataclass
class GazeSample:
    timestamp:   float
    gaze_x:      float
    gaze_y:      float
    left_ear:    float
    right_ear:   float
    head_pitch:  float
    head_yaw:    float
    head_roll:   float
    confidence:  float
    frame_index: int = 0


# ── Anomaly Detector ──────────────────────────────────────────────────────────

class AnomalyDetector:
    PROLONGED_MS     = 900.0
    SHORT_MS         = 80.0
    EXPRESS_MS       = 120.0
    SLOW_VEL         = 100.0
    YAW_TH           = 25.0
    BLINK_LO         = 3.0
    BLINK_HI         = 30.0
    BLINK_WINDOW_SEC = 20.0
    DRIFT_TH         = 0.10
    DISP_HIGH        = 0.09

    DEDUP = {
        "NYSTAGMUS_LIKE":3.0,"SQUARE_WAVE_JERK":2.0,
        "ABSENT_BLINK":12.0,"EXCESSIVE_BLINK_RATE":12.0,
        "OFF_SCREEN_GAZE":2.0,"LOST_TRACKING":3.0,
        "GAZE_DRIFT":4.0,"HEAD_TREMOR":5.0,"EXCESSIVE_HEAD_YAW":3.0,
    }

    def __init__(self):
        self.events:        List[AnomalyEvent] = []
        self._gaze_hist     = deque(maxlen=90)
        self._head_hist     = deque(maxlen=30)
        self._blink_times:  List[float] = []
        self._in_blink      = False
        self._frame_idx     = 0
        self._session_start = time.time()
        self._last_ev_t:    Dict[str,float] = {}
        self._fix_anchor    = None
        self._fix_start     = None

    def process_sample(self, s: GazeSample):
        self._frame_idx += 1
        self._gaze_hist.append((s.timestamp, s.gaze_x, s.gaze_y))
        self._head_hist.append((s.timestamp, s.head_pitch, s.head_yaw, s.head_roll))
        if abs(s.head_yaw) > self.YAW_TH:
            self._rec(AnomalyType.EXCESSIVE_HEAD_YAW, abs(s.head_yaw), self.YAW_TH, "deg", s)
        if s.confidence > 0.65:
            if not (0.05 < s.gaze_x < 0.95 and 0.05 < s.gaze_y < 0.95):
                self._rec(AnomalyType.OFF_SCREEN_GAZE,
                          max(abs(s.gaze_x-.5), abs(s.gaze_y-.5)), 0.45, "norm", s)
        avg_ear = (s.left_ear + s.right_ear) / 2.0
        if avg_ear < 0.21 and not self._in_blink:
            self._in_blink = True
            self._blink_times.append(s.timestamp)
        elif avg_ear >= 0.21:
            self._in_blink = False
        elapsed = s.timestamp - self._session_start
        if elapsed > self.BLINK_WINDOW_SEC:
            self._check_blink_rate(s)
        self._check_head_tremor(s)
        if s.confidence > 0.5:
            self._check_sqw_nystagmus(s)
        self._check_drift(s)

    def process_fixation(self, duration_ms: float, cx: float, cy: float, dispersion: float):
        dummy = GazeSample(time.time(), cx, cy, .30, .30, 0, 0, 0, .9, self._frame_idx)
        if duration_ms > self.PROLONGED_MS:
            self._rec(AnomalyType.PROLONGED_FIXATION, duration_ms, self.PROLONGED_MS, "ms", dummy)
        if duration_ms < self.SHORT_MS:
            self._rec(AnomalyType.SHORT_FIXATION, duration_ms, self.SHORT_MS, "ms", dummy)
        if dispersion > self.DISP_HIGH:
            self._rec(AnomalyType.FIXATION_INSTABILITY, dispersion, self.DISP_HIGH, "norm", dummy)

    def process_saccade(self, latency_ms:float, peak_vel:float, amplitude:float,
                        cx:float, cy:float, is_anti:bool=False, correct:bool=True):
        dummy = GazeSample(time.time(), cx, cy, .30, .30, 0, 0, 0, .9, self._frame_idx)
        if 0 < latency_ms < self.EXPRESS_MS:
            self._rec(AnomalyType.EXPRESS_SACCADE, latency_ms, self.EXPRESS_MS, "ms", dummy)
        if peak_vel < self.SLOW_VEL and amplitude > 2:
            self._rec(AnomalyType.SLOW_SACCADE, peak_vel, self.SLOW_VEL, "deg/s", dummy)
        if is_anti and not correct:
            self._rec(AnomalyType.ANTI_SACCADE_ERROR, 1.0, 0.0, "binary", dummy)

    def get_risk_score(self) -> float:
        if not self.events:
            return 0.0
        w = {"CRITICAL":4.0,"HIGH":2.0,"MEDIUM":0.8,"LOW":0.2}
        raw = sum(w.get(e.severity,.2) * (1.5 if e.is_lbd_biomarker else 1.0)
                  for e in self.events)
        elapsed = max(1.0, time.time() - self._session_start)
        rate    = raw / elapsed
        return float(np.clip(1.0/(1.0+math.exp(-9.0*(rate-0.09))), 0.0, 1.0))

    def get_blink_rate(self) -> float:
        now    = time.time()
        recent = [b for b in self._blink_times if b > now-60]
        if len(recent) < 2: return 0.0
        dur_min = (recent[-1]-recent[0])/60.0
        return round(len(recent)/max(dur_min,.1), 1)

    def get_fixation_stability(self) -> float:
        if len(self._gaze_hist) < 6: return 0.0
        xs = np.array([g[1] for g in self._gaze_hist])
        ys = np.array([g[2] for g in self._gaze_hist])
        return float(np.clip((np.std(xs)+np.std(ys))/0.30, 0.0, 1.0))

    def _check_blink_rate(self, s: GazeSample):
        now    = s.timestamp
        recent = [b for b in self._blink_times if b > now-60]
        self._blink_times = recent
        if len(recent) < 5: return
        dur_min = (recent[-1]-recent[0])/60.0
        if dur_min < 0.3: return
        rate = len(recent)/dur_min
        if rate < self.BLINK_LO:
            self._rec(AnomalyType.ABSENT_BLINK, rate, self.BLINK_LO, "b/min", s)
        elif rate > self.BLINK_HI:
            self._rec(AnomalyType.EXCESSIVE_BLINK_RATE, rate, self.BLINK_HI, "b/min", s)

    def _check_head_tremor(self, s: GazeSample):
        if len(self._head_hist) < 10: return
        yaws  = np.array([h[2] for h in self._head_hist])
        diffs = np.diff(yaws)
        sc    = int(np.sum(np.diff(np.sign(diffs)) != 0))
        rms   = float(np.sqrt(np.mean(diffs**2))) * 30
        if sc >= 6 and rms > 15.0:
            self._rec(AnomalyType.HEAD_TREMOR, rms, 15.0, "deg/s", s)

    def _check_sqw_nystagmus(self, s: GazeSample):
        if len(self._gaze_hist) < 15: return
        xs = np.array([g[1] for g in self._gaze_hist])
        ts = np.array([g[0] for g in self._gaze_hist])
        vels = np.abs(np.diff(xs))/(np.diff(ts)+1e-6)
        peaks, _ = find_peaks(vels, height=0.40, distance=3)
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                tb = ts[peaks[i+1]]-ts[peaks[i]]
                xe = abs(xs[peaks[i]]-xs[peaks[i+1]])
                if tb < 0.40 and xe > 0.025:
                    self._rec(AnomalyType.SQUARE_WAVE_JERK, xe, 0.025, "norm", s)
                    break
        zc   = int(np.sum(np.diff(np.sign(xs-np.mean(xs))) != 0))
        tt   = ts[-1]-ts[0]+1e-6
        freq = zc/(2*tt)
        if 0.5 < freq < 5.0 and np.std(xs) > 0.035:
            self._rec(AnomalyType.NYSTAGMUS_LIKE, freq, 0.5, "Hz", s)

    def _check_drift(self, s: GazeSample):
        if self._fix_anchor is None:
            self._fix_anchor = (s.gaze_x, s.gaze_y)
            self._fix_start  = s.timestamp
            return
        dist = math.hypot(s.gaze_x-self._fix_anchor[0], s.gaze_y-self._fix_anchor[1])
        if dist > 0.08:
            self._fix_anchor = (s.gaze_x, s.gaze_y)
            self._fix_start  = s.timestamp
        else:
            dur = s.timestamp-(self._fix_start or s.timestamp)
            if dur > 1.5 and dist > self.DRIFT_TH:
                self._rec(AnomalyType.GAZE_DRIFT, dist, self.DRIFT_TH, "norm", s)

    def _rec(self, atype:AnomalyType, measured:float, threshold:float,
             unit:str, s:GazeSample):
        now   = time.time()
        aname = atype.name
        gap   = self.DEDUP.get(aname, 0.4)
        if now - self._last_ev_t.get(aname, 0.0) < gap:
            return
        self._last_ev_t[aname] = now
        uid = hashlib.md5(f"{aname}{now}{s.frame_index}".encode()).hexdigest()[:12]
        ev  = AnomalyEvent(
            anomaly_id=uid, anomaly_type=aname,
            severity=SEVERITY_MAP.get(atype,"LOW"),
            timestamp=now, frame_index=s.frame_index,
            clinical_note=CLINICAL_NOTES.get(atype,""),
            is_lbd_biomarker=(atype in LBD_BIOMARKERS),
            measured_value=round(measured,4), threshold_value=round(threshold,4),
            unit=unit, gaze_x=round(s.gaze_x,4), gaze_y=round(s.gaze_y,4),
        )
        self.events.append(ev)


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    FIX_RADIUS = 0.05
    FIX_MIN_DUR = 0.10
    FPS = 30

    def extract(self, gaze_pos: list) -> dict:
        if len(gaze_pos) < 5:
            return {k:0.0 for k in FEATURE_COLS}
        xs  = np.array([p[0] for p in gaze_pos], float)
        ys  = np.array([p[1] for p in gaze_pos], float)
        dt  = 1.0/self.FPS
        diff = np.hypot(np.diff(xs), np.diff(ys))
        spl  = float(diff.sum())
        gv   = float(np.std(xs)+np.std(ys))
        vel  = diff/dt
        mv   = vel.mean()
        sv   = vel.std()
        thr  = mv+2*sv if sv > 0 else mv*2
        is_s = vel > thr
        sc   = int(np.sum(np.diff(is_s.astype(int))==1)) if is_s.any() else 0
        dur  = len(gaze_pos)*dt
        sf   = sc/max(dur,1.0)
        fix_durs, i, n = [], 0, len(gaze_pos)
        while i < n:
            j = i+1
            while j < n and np.hypot(xs[j]-xs[i],ys[j]-ys[i]) <= self.FIX_RADIUS:
                j += 1
            d = (j-i)*dt
            if d >= self.FIX_MIN_DUR:
                fix_durs.append(d)
            i = j
        mfd = float(np.mean(fix_durs)) if fix_durs else 0.28
        return {
            "fixation_duration": round(mfd,4),
            "saccade_frequency": round(sf,4),
            "scan_path_length":  round(spl,4),
            "gaze_variability":  round(gv,4),
        }


# ── Gaze Simulator (fallback when no camera) ─────────────────────────────────

class GazeSimulator:
    @staticmethod
    def simulate(anchors, n_seconds=4.0, fps=30, noise=0.012):
        if not anchors:
            anchors = [(0.5, 0.5)]
        rng   = np.random.default_rng()
        n_pts = int(n_seconds * fps)
        pts   = []
        cur   = list(random.choice(anchors))
        rem   = rng.integers(8, 22)
        for _ in range(n_pts):
            if rem <= 0:
                cur = list(random.choice(anchors))
                rem = rng.integers(8, 22)
            nx = rng.normal(0, noise, 2)
            pts.append((
                float(np.clip(cur[0]+nx[0], 0.02, 0.98)),
                float(np.clip(cur[1]+nx[1], 0.02, 0.98)),
            ))
            rem -= 1
        return pts


# ── PyTorch model (optional) ──────────────────────────────────────────────────

if TORCH_OK:
    class GazeRiskNet(nn.Module):
        def __init__(self, in_dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64,32),     nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32,16),     nn.ReLU(),
                nn.Linear(16,2),
            )
        def forward(self,x): return self.net(x)


def _load_torch_model():
    if not TORCH_OK or not os.path.exists("best_gaze_model.pth"):
        return None
    try:
        mdl  = GazeRiskNet(8)
        ckpt = torch.load("best_gaze_model.pth", map_location="cpu")
        mdl.load_state_dict(ckpt.get("model_state_dict", ckpt))
        mdl.eval()
        return mdl
    except Exception:
        return None


# ── Risk Classifier ───────────────────────────────────────────────────────────

@st.cache_resource
def load_classifier():
    return RiskClassifier()


class RiskClassifier:
    def __init__(self):
        self.rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        self.gb  = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=42)
        self.sc  = StandardScaler()
        self.torch_mdl = _load_torch_model()
        self._trained  = False
        pkl = os.path.join(RESULTS_DIR,"clf_v5.pkl")
        if os.path.exists(pkl):
            try:
                d = joblib.load(pkl)
                self.rf,self.gb,self.sc = d["rf"],d["gb"],d["sc"]
                self._trained = True
            except Exception:
                pass
        if not self._trained:
            self._train()

    def _train(self):
        rng = np.random.default_rng(42)
        n   = 160
        def gen(fd,sf,spl,gv,rt,ac,ae,ar,lbl):
            return pd.DataFrame({
                "fixation_duration":      np.clip(rng.normal(fd, fd*.14,n),  .06,1.6),
                "saccade_frequency":      np.clip(rng.normal(sf, sf*.18,n),  .3,14),
                "scan_path_length":       np.clip(rng.normal(spl,spl*.16,n), .06,3),
                "gaze_variability":       np.clip(rng.normal(gv, gv*.18,n),  .01,.6),
                "reaction_time":          np.clip(rng.normal(rt, rt*.18,n),  .15,10),
                "accuracy":               np.clip(rng.normal(ac, .10,n),     0,1),
                "antisaccade_error_rate": np.clip(rng.normal(ae, .10,n),     0,1),
                "antisaccade_rt":         np.clip(rng.normal(ar, ar*.16,n),  .08,3.5),
                "label":[lbl]*n,
            })
        low  = gen(.24, 3.0, .35, .08, 1.1, 0.84, 0.10, 0.36, 0)
        high = gen(.65, 6.5, .72, .22, 3.0, 0.42, 0.70, 1.20, 1)
        df   = pd.concat([low,high]).reset_index(drop=True)
        X    = df[FEATURE_COLS].values
        y    = df["label"].values
        self.sc.fit(X)
        Xs = self.sc.transform(X)
        self.rf.fit(Xs, y)
        self.gb.fit(Xs, y)
        self._trained = True
        joblib.dump({"rf":self.rf,"gb":self.gb,"sc":self.sc},
                    os.path.join(RESULTS_DIR,"clf_v5.pkl"))

    def predict(self, feats:dict) -> dict:
        x  = np.array([[feats.get(c,0.0) for c in FEATURE_COLS]])
        xs = self.sc.transform(x)
        p_rf = float(self.rf.predict_proba(xs)[0][1])
        p_gb = float(self.gb.predict_proba(xs)[0][1])
        p_pt = None
        if self.torch_mdl is not None and TORCH_OK:
            try:
                with torch.no_grad():
                    p_pt = float(torch.softmax(
                        self.torch_mdl(torch.FloatTensor(xs)),1)[0][1])
            except Exception:
                p_pt = None
        if p_pt is not None:
            prob   = p_rf*.50 + p_gb*.35 + p_pt*.15
            source = "RF + GB + PyTorch"
        else:
            prob   = p_rf*.58 + p_gb*.42
            source = "RF + GB ensemble"
        if prob >= .75:
            label = "CRITICAL RISK"
        elif prob >= .50:
            label = "HIGH RISK"
        elif prob >= .25:
            label = "MODERATE RISK"
        else:
            label = "LOW RISK"
        return {
            "probability": float(prob),
            "label": label,
            "rf":p_rf,"gb":p_gb,"torch":p_pt,
            "source":source,
        }


# ── Video Processor (FIX 2: async_processing=False, writeable flags) ──────────

if WEBRTC_OK and CV2_OK and MP_OK:
    class CalibrationProcessor(VideoProcessorBase):
        L_IRIS=[468,469,470,471,472]; R_IRIS=[473,474,475,476,477]
        L_OUT=33;  L_IN=133; R_OUT=362; R_IN=263
        L_TOP=159; L_BOT=145; R_TOP=386; R_BOT=374
        NOSE=1;    SMOOTH=6

        def __init__(self):
            self.lock         = threading.Lock()
            self.gaze_norm    = (0.5, 0.5)
            self.gaze_dir     = "CENTER"
            self.fixation_dur = 0.0
            self.blink_count  = 0
            self.left_ear     = 0.30
            self.right_ear    = 0.30
            self.confidence   = 0.0
            self.head_yaw     = 0.0
            self.history      = deque(maxlen=900)
            self.ear_history  = deque(maxlen=150)
            self.blink_times: List[float] = []
            self._last_blink  = False
            self._fix_anchor  = None
            self._fix_start   = None
            self._bx          = deque(maxlen=self.SMOOTH)
            self._by          = deque(maxlen=self.SMOOTH)
            # FIX: initialise FaceMesh with error guard
            self.fm = None
            try:
                fm_mod = mp.solutions.face_mesh
                self.fm = fm_mod.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.60,
                    min_tracking_confidence=0.60,
                )
            except Exception as e:
                print(f"[FaceMesh init error] {e}")

        @staticmethod
        def _ih(lm, iris, o, i_):
            ix = float(np.mean([lm[k].x for k in iris]))
            lo = min(lm[o].x, lm[i_].x)
            hi = max(lm[o].x, lm[i_].x)
            return float(np.clip((ix-lo)/(hi-lo+1e-6), 0, 1))

        @staticmethod
        def _iv(lm, iris, t, b):
            iy = float(np.mean([lm[k].y for k in iris]))
            t_ = min(lm[t].y, lm[b].y)
            b_ = max(lm[t].y, lm[b].y)
            return float(np.clip((iy-t_)/(b_-t_+1e-6), 0, 1))

        @staticmethod
        def _ear(lm, top, bot, ou, in_):
            return abs(lm[top].y-lm[bot].y) / (abs(lm[ou].x-lm[in_].x)+1e-6)

        def recv(self, frame):
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            fh, fw = img.shape[:2]
            now = time.time()

            if self.fm is None:
                cv2.putText(img, "MediaPipe not initialised", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # FIX: set writeable=False before MediaPipe, True after
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            try:
                res = self.fm.process(rgb)
            except Exception:
                res = None
            rgb.flags.writeable = True

            if not (res and res.multi_face_landmarks):
                with self.lock:
                    self.confidence = 0.0
                cv2.putText(img, "No face detected — centre your face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            lm = res.multi_face_landmarks[0].landmark

            lh = self._ih(lm, self.L_IRIS, self.L_OUT, self.L_IN)
            rh = self._ih(lm, self.R_IRIS, self.R_OUT, self.R_IN)
            lv = self._iv(lm, self.L_IRIS, self.L_TOP, self.L_BOT)
            rv = self._iv(lm, self.R_IRIS, self.R_TOP, self.R_BOT)
            self._bx.append((lh+rh)/2)
            self._by.append((lv+rv)/2)
            sh = float(np.mean(self._bx))
            sv = float(np.mean(self._by))

            l_ear   = self._ear(lm, self.L_TOP, self.L_BOT, self.L_OUT, self.L_IN)
            r_ear   = self._ear(lm, self.R_TOP, self.R_BOT, self.R_OUT, self.R_IN)
            avg_ear = (l_ear + r_ear) / 2.0
            self.ear_history.append(avg_ear)

            # Blink detection
            blink = avg_ear < 0.21
            if blink and not self._last_blink:
                with self.lock:
                    self.blink_count += 1
                    self.blink_times.append(now)
            self._last_blink = blink

            # Fixation duration
            if self._fix_anchor is None:
                self._fix_anchor = (sh, sv)
                self._fix_start  = now
            elif math.hypot(sh-self._fix_anchor[0], sv-self._fix_anchor[1]) > 0.06:
                self._fix_anchor = (sh, sv)
                self._fix_start  = now

            # Head yaw estimate
            nose_x  = lm[self.NOSE].x
            eye_mid = (lm[self.L_OUT].x + lm[self.R_OUT].x) / 2
            yaw_est = (nose_x - eye_mid) * 110

            gdir_h = "LEFT" if sh < .38 else ("RIGHT" if sh > .62 else "CENTER")
            gdir_v = "UP"   if sv < .33 else ("DOWN"  if sv > .67 else "CENTER")

            with self.lock:
                self.gaze_norm    = (sh, sv)
                self.gaze_dir     = f"{gdir_h}-{gdir_v}"
                self.fixation_dur = round(now-(self._fix_start or now), 2)
                self.left_ear     = float(l_ear)
                self.right_ear    = float(r_ear)
                self.confidence   = 0.95
                self.head_yaw     = float(yaw_est)
                self.history.append((float(sh), float(sv)))

            # Draw iris rings
            for iris_pts in [self.L_IRIS, self.R_IRIS]:
                ix_px = int(np.mean([lm[k].x for k in iris_pts]) * fw)
                iy_px = int(np.mean([lm[k].y for k in iris_pts]) * fh)
                cv2.circle(img, (ix_px, iy_px), 18, (200, 220, 255), 1)
                cv2.circle(img, (ix_px, iy_px), 9,  (0, 200, 255),   1)
                cv2.circle(img, (ix_px, iy_px), 5,  (0, 220, 255),  -1)
                cv2.circle(img, (ix_px, iy_px), 2,  (8, 8, 12),     -1)

            # EAR bar
            bar_x, bar_y = 10, fh-60
            bar_len = int(np.clip(avg_ear / 0.40, 0, 1) * 180)
            ear_col = (0, 200, 80) if avg_ear >= 0.21 else (0, 80, 255)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x+180, bar_y+14), (30,36,50), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_len, bar_y+14), ear_col, -1)
            cv2.putText(img, f"EAR {avg_ear:.3f}{'  BLINK' if avg_ear<0.21 else ''}",
                        (bar_x, bar_y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ear_col, 1)

            # Head yaw
            yaw_col = (0, 80, 255) if abs(yaw_est) > 22 else (0, 200, 80)
            cv2.putText(img, f"YAW {yaw_est:+.0f}deg",
                        (bar_x, bar_y+32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, yaw_col, 1)

            cv2.putText(img, "TRACKING OK", (fw-130, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 80), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def get_state(self):
            with self.lock:
                return {k: getattr(self, k) for k in
                        ("gaze_norm","gaze_dir","fixation_dur","blink_count",
                         "left_ear","right_ear","confidence","head_yaw")}

        def get_blink_rate_live(self) -> float:
            with self.lock:
                recent = [b for b in self.blink_times if b > time.time()-60]
                if len(recent) < 2:
                    return 0.0
                dur = (recent[-1]-recent[0]) / 60.0
                return round(len(recent)/max(dur, .01), 1)

        def get_ear_stability(self) -> float:
            with self.lock:
                if len(self.ear_history) < 10:
                    return 0.0
                return float(np.std(list(self.ear_history)))


# ── Navigation helpers ────────────────────────────────────────────────────────

STEPS = ["Welcome","Eye Baseline","Memory Task","Saccade Task","Report"]


def init_state():
    defs = dict(
        page="welcome", participant="P001", age=65,
        cal_start=None, cal_samples=[], cal_done=False,
        det=None, gaze_log=[],
        mem_trial=0, mem_trials=[], mem_pattern=[], mem_probe_count=0,
        mem_phase="show", mem_phase_start=None, mem_selections=[], mem_last={},
        sacc_trial=0, sacc_trials=[],
        sacc_side=None, sacc_correct=None,
        sacc_phase="fixation", sacc_phase_start=None, sacc_last={},
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.det is None:
        st.session_state.det = AnomalyDetector()


def nav(page):
    st.session_state.page = page


def hdr(subtitle=""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:11px;padding-bottom:.35rem;">
        <div style="background:{ACCENT};border-radius:5px;padding:4px 8px;color:#fff;
                    font-family:'JetBrains Mono',monospace;font-weight:600;font-size:.88rem;">NS</div>
        <div>
            <div style="font-size:.95rem;font-weight:700;color:{TEXT};">NeuroScan LBD v5</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.64rem;color:{MUTED};">
                LBD Assessment</div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:.77rem;color:{MUTED};">{subtitle}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.63rem;color:{MUTED};">
                {datetime.now().strftime('%H:%M  %d %b %Y')}</div>
        </div>
    </div><hr class="div">""", unsafe_allow_html=True)


def show_steps(active):
    cols = st.columns(len(STEPS))
    for i, (col, lbl) in enumerate(zip(cols, STEPS)):
        c = ACCENT if i==active-1 else (SAFE if i<active-1 else BORDER)
        t = TEXT   if i==active-1 else (MUTED if i<active-1 else BORDER)
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:2px 0;">'
                f'<div style="height:3px;background:{c};border-radius:2px;margin-bottom:3px;"></div>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.66rem;'
                f'color:{t};font-weight:{"600" if i==active-1 else "400"};">{lbl}</span></div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def grid_svg_bg(w, h, step=50):
    p = []
    for gx in range(0, w+1, step):
        p.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{h}" stroke="#E2E8F0" stroke-width="0.8"/>')
    for gy in range(0, h+1, step):
        p.append(f'<line x1="0" y1="{gy}" x2="{w}" y2="{gy}" stroke="#E2E8F0" stroke-width="0.8"/>')
    return "".join(p)


# ── PAGE 1: WELCOME ───────────────────────────────────────────────────────────

def page_welcome():
    hdr()
    L, R = st.columns([3, 2], gap="large")
    with L:
        st.markdown(f"""
        <div style="padding:.6rem 0 .8rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        color:{ACCENT};letter-spacing:.14em;margin-bottom:.45rem;">CLINICAL RESEARCH TOOL</div>
            <h1 style="font-size:2.3rem;font-weight:700;color:{TEXT};margin:0;line-height:1.18;">
                LBD Cognitive<br>Assessment</h1>
            <p style="color:{MUTED};font-size:.90rem;margin:.7rem 0 0;max-width:460px;line-height:1.75;">
                Early detection of <strong style="color:{TEXT};">Lewy Body Dementia</strong> through
                live gaze analysis, grid-based spatial memory,
                and saccadic inhibition profiling.
            </p>
        </div>""", unsafe_allow_html=True)
        cards = [
            (ACCENT,  "Eye Baseline", "30s blink + concentration",    "STEP 1"),
            (SAFE,    "Grid Memory",  "4x4 spatial recall, 5 trials", "STEP 2"),
            (ACCENT2, "Antisaccade",  "Inhibition task, 5 trials",    "STEP 3"),
            (WARN,    "Risk Report",  "Ensemble ML classification",   "STEP 4"),
        ]
        for col, (c, ttl, dsc, tag) in zip(st.columns(4), cards):
            with col:
                st.markdown(f"""
                <div class="card" style="border-top:3px solid {c};">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{c};margin-bottom:3px;">{tag}</div>
                    <div style="font-weight:600;font-size:.83rem;color:{TEXT};margin-bottom:2px;">{ttl}</div>
                    <div style="font-size:.73rem;color:{MUTED};">{dsc}</div>
                </div>""", unsafe_allow_html=True)
    with R:
        st.markdown('<div class="card-blue">', unsafe_allow_html=True)
        st.markdown(f'<div class="sec" style="margin-bottom:.6rem;">PARTICIPANT DETAILS</div>',
                    unsafe_allow_html=True)
        pid = st.text_input("Participant ID", value=st.session_state.participant)
        age = st.number_input("Age", min_value=18, max_value=110, value=st.session_state.age)
        st.markdown('</div>', unsafe_allow_html=True)

        cam_ok = WEBRTC_OK and CV2_OK and MP_OK

        # FIX: show exact import status for debugging
        deps = [
            ("opencv-python-headless", CV2_OK,   "pip install opencv-python-headless"),
            ("mediapipe",              MP_OK,     "pip install mediapipe==0.10.14"),
            ("streamlit-webrtc + av",  WEBRTC_OK, "pip install streamlit-webrtc av"),
            ("best_gaze_model.pth",    os.path.exists("best_gaze_model.pth"), "optional"),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:7px;padding:2px 0;">'
            f'<div style="width:7px;height:7px;border-radius:50%;'
            f'background:{SAFE if ok else WARN};flex-shrink:0;"></div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.72rem;'
            f'color:{TEXT};flex:1;">{pkg}</div>'
            f'<div style="font-size:.68rem;color:{SAFE if ok else WARN};">'
            f'{"Ready" if ok else note}</div></div>'
            for pkg, ok, note in deps)
        card_bg = "#F0FFF4" if cam_ok else "#FFFAF0"
        card_bd = SAFE      if cam_ok else WARN
        msg     = "Live webcam ready." if cam_ok else "Dependency missing — simulation mode."
        st.markdown(
            f'<div style="background:{card_bg};border:1.5px solid {card_bd}44;'
            f'border-radius:6px;padding:.55rem .8rem;margin-bottom:.55rem;">'
            f'<div style="font-weight:600;font-size:.80rem;color:{card_bd};margin-bottom:.35rem;">{msg}</div>'
            f'{rows}</div>', unsafe_allow_html=True)

        if st.button("Begin Assessment", use_container_width=True):
            st.session_state.participant = pid or "P001"
            st.session_state.age = int(age)
            nav("calibration")
            st.rerun()

        st.markdown(
            f'<div style="background:#FFFAF0;border:1px solid {WARN}44;border-radius:5px;'
            f'padding:.5rem .7rem;font-size:.74rem;color:{WARN};line-height:1.55;">'
            f'Research use only. Not a clinical diagnosis.</div>',
            unsafe_allow_html=True)


# ── PAGE 2: EYE BASELINE ──────────────────────────────────────────────────────

def page_calibration():
    hdr(f"Participant: {st.session_state.participant}")
    show_steps(2)
    det: AnomalyDetector = st.session_state.det

    if st.session_state.cal_start is None:
        st.markdown(f"""
        <div class="card-blue">
            <div class="sec">EYE BASELINE  —  30-SECOND CONCENTRATION RECORDING</div>
            <p style="color:{MUTED};font-size:.86rem;margin:.3rem 0 .8rem;">
                Keep your gaze fixed on the central cross. The system measures blink rate,
                EAR (eye aspect ratio), head pose, and fixation stability.
            </p>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
                <div><b style="font-size:.82rem;">1. Allow camera</b>
                    <div style="font-size:.73rem;color:{MUTED};">Browser permission required.</div></div>
                <div><b style="font-size:.82rem;">2. Centre your face</b>
                    <div style="font-size:.73rem;color:{MUTED};">Iris rings appear when detected.</div></div>
                <div><b style="font-size:.82rem;">3. Fix gaze on cross</b>
                    <div style="font-size:.73rem;color:{MUTED};">Stare at the + in the middle.</div></div>
                <div><b style="font-size:.82rem;">4. Blink naturally</b>
                    <div style="font-size:.73rem;color:{MUTED};">EAR and blink rate recorded.</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Start Eye Baseline", use_container_width=True):
                st.session_state.cal_start   = time.time()
                st.session_state.cal_samples = []
                st.rerun()
        return

    elapsed = time.time() - st.session_state.cal_start
    rem     = max(0.0, EYE_CAL_SEC - elapsed)
    pct     = min(1.0, elapsed / EYE_CAL_SEC)

    cam_col, info_col = st.columns([3, 2], gap="large")

    with cam_col:
        st.markdown(f'<div class="sec">LIVE EAR  ·  BLINK  ·  HEAD POSE TRACKING</div>',
                    unsafe_allow_html=True)

        if WEBRTC_OK and CV2_OK and MP_OK:
            # FIX: async_processing=False, multiple STUN servers, constrained video size
            ctx = webrtc_streamer(
                key="cal",
                video_processor_factory=CalibrationProcessor,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                    ]
                }),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                async_processing=False,   # FIX: was True
            )
            if ctx.video_processor:
                vp  = ctx.video_processor
                st_ = vp.get_state()
                hist = list(vp.history)
                if hist:
                    st.session_state.cal_samples.extend(hist[-6:])
                    st.session_state.gaze_log.extend(hist[-6:])
                gx, gy = st_["gaze_norm"]
                samp = GazeSample(
                    timestamp=time.time(), gaze_x=gx, gaze_y=gy,
                    left_ear=st_["left_ear"], right_ear=st_["right_ear"],
                    head_pitch=0.0, head_yaw=st_["head_yaw"], head_roll=0.0,
                    confidence=st_["confidence"],
                    frame_index=len(st.session_state.cal_samples))
                det.process_sample(samp)

                bc   = st_["blink_count"]
                fix  = st_["fixation_dur"]
                yaw  = st_["head_yaw"]
                ear  = (st_["left_ear"]+st_["right_ear"]) / 2
                br   = vp.get_blink_rate_live()
                stab = det.get_fixation_stability()

                ear_c  = DANGER if ear < .18 else SAFE
                yaw_c  = DANGER if abs(yaw) > 22 else SAFE
                stab_c = DANGER if stab > .65 else (WARN if stab > .35 else SAFE)
                br_c   = DANGER if (br < 3 or br > 30) else SAFE

                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("EAR", f"{ear:.3f}")
                with m2: st.metric("Blink Rate", f"{br:.1f}/min")
                with m3: st.metric("Head Yaw", f"{yaw:+.0f}°")
                with m4: st.metric("Blinks", str(bc))

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:.5rem;">
                    <div class="card" style="padding:.6rem;text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{MUTED};">BLINK RATE</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{br_c};font-family:'JetBrains Mono',monospace;">{br:.1f}</div>
                        <div style="font-size:.70rem;color:{br_c};">/min{"  LOW" if br<3 else ("  HIGH" if br>25 else "  Normal")}</div>
                    </div>
                    <div class="card" style="padding:.6rem;text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{MUTED};">HEAD YAW</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{yaw_c};font-family:'JetBrains Mono',monospace;">{yaw:+.0f}°</div>
                        <div style="font-size:.70rem;color:{yaw_c};">{"Turned" if abs(yaw)>22 else "Centred"}</div>
                    </div>
                    <div class="card" style="padding:.6rem;text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{MUTED};">FIXATION</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{stab_c};font-family:'JetBrains Mono',monospace;">{fix:.1f}s</div>
                        <div style="font-size:.70rem;color:{stab_c};">{"Unstable" if stab>.65 else ("Variable" if stab>.35 else "Stable")}</div>
                    </div>
                    <div class="card" style="padding:.6rem;text-align:center;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{MUTED};">ANOMALIES</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{ACCENT};font-family:'JetBrains Mono',monospace;">{len(det.events)}</div>
                        <div style="font-size:.70rem;color:{MUTED};">detected</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                if det.events:
                    ev  = det.events[-1]
                    scol = {"CRITICAL":DANGER,"HIGH":WARN,"MEDIUM":ACCENT,"LOW":MUTED}
                    sbg  = {"CRITICAL":"#FFF5F5","HIGH":"#FFFAF0","MEDIUM":"#EBF8FF","LOW":LIGHT}
                    tc_  = scol.get(ev.severity, MUTED)
                    bg_  = sbg.get(ev.severity, LIGHT)
                    lbd_ = "  [LBD]" if ev.is_lbd_biomarker else ""
                    st.markdown(
                        f'<div style="background:{bg_};border:1px solid {tc_}44;border-radius:5px;'
                        f'padding:.35rem .7rem;margin-top:.4rem;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:{tc_};">'
                        f'[{ev.severity}]  {ev.anomaly_type.replace("_"," ")}'
                        f'  val={ev.measured_value:.3f} {ev.unit}{lbd_}</div>',
                        unsafe_allow_html=True)
            else:
                st.info("📷 Click **START** in the video widget above to begin camera tracking.")

        else:
            # Simulation mode
            sim_pts = GazeSimulator.simulate([(0.5, 0.5)], 0.28, 10, 0.006)
            st.session_state.cal_samples.extend(sim_pts)
            st.session_state.gaze_log.extend(sim_pts)
            for px_, py_ in sim_pts[-3:]:
                samp = GazeSample(
                    timestamp=time.time(), gaze_x=px_, gaze_y=py_,
                    left_ear=0.28+random.gauss(0, .01),
                    right_ear=0.28+random.gauss(0, .01),
                    head_pitch=0, head_yaw=float(random.gauss(0, 2.5)), head_roll=0,
                    confidence=0.88, frame_index=len(st.session_state.cal_samples))
                det.process_sample(samp)
            sb      = int(elapsed * 0.27)
            br      = sb / max(elapsed, 1) * 60
            ear_sim = 0.28 + random.gauss(0, .005)
            stab    = det.get_fixation_stability()
            st.markdown(
                f'<div style="background:#EBF8FF;border:1px solid #90CDF4;border-radius:6px;'
                f'padding:.65rem;margin-bottom:.45rem;font-size:.82rem;color:{ACCENT};">'
                f'Simulation mode — metrics being generated at centre fixation.</div>',
                unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("EAR (sim)",  f"{ear_sim:.3f}")
            with m2: st.metric("Blink Rate", f"{br:.1f}/min")
            with m3: st.metric("Stability",  f"{(1-stab)*100:.0f}%")
            with m4: st.metric("Anomalies",  str(len(det.events)))

    with info_col:
        st.markdown(f'<div class="sec">CONCENTRATION  +  EYE PHYSIOLOGY</div>',
                    unsafe_allow_html=True)
        cross_svg = f"""
        <div style="display:flex;justify-content:center;margin-bottom:.7rem;">
        <svg width="280" height="180"
             style="background:#F7FAFC;border:1.5px solid {BORDER};border-radius:8px;">
          {"".join(f'<line x1="{gx}" y1="0" x2="{gx}" y2="180" stroke="#E2E8F0" stroke-width="0.7"/>' for gx in range(0,281,40))}
          {"".join(f'<line x1="0" y1="{gy}" x2="280" y2="{gy}" stroke="#E2E8F0" stroke-width="0.7"/>' for gy in range(0,181,40))}
          <line x1="100" y1="90" x2="180" y2="90" stroke="{TEXT}" stroke-width="3"/>
          <line x1="140" y1="50" x2="140" y2="130" stroke="{TEXT}" stroke-width="3"/>
          <circle cx="140" cy="90" r="5" fill="{ACCENT}"/>
          <circle cx="140" cy="90" r="18" fill="none" stroke="{ACCENT}44" stroke-width="2"/>
          <text x="140" y="158" text-anchor="middle" font-size="11"
                font-family="monospace" fill="{MUTED}">Fix your gaze here</text>
        </svg>
        </div>"""
        st.markdown(cross_svg, unsafe_allow_html=True)

        br_now  = det.get_blink_rate()
        br_c    = DANGER if br_now < 3 else (WARN if br_now > 25 else SAFE)
        br_norm = min(1.0, br_now / 25.0)
        st.markdown(
            f'<div style="margin-bottom:.55rem;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.66rem;'
            f'color:{MUTED};margin-bottom:2px;">BLINK RATE: {br_now:.1f} / min</div>'
            f'<div style="background:{LIGHT};border-radius:3px;height:7px;">'
            f'<div style="width:{br_norm*100:.0f}%;height:100%;background:{br_c};border-radius:3px;"></div>'
            f'</div><div style="font-size:.62rem;color:{MUTED};margin-top:2px;">'
            f'Normal 3–25 / min</div></div>',
            unsafe_allow_html=True)

        lbd_n  = sum(1 for e in det.events if e.is_lbd_biomarker)
        sev_c  = Counter(e.severity for e in det.events)
        risk   = det.get_risk_score()
        a1, a2, a3, a4 = st.columns(4)
        with a1: st.metric("Total",     str(len(det.events)))
        with a2: st.metric("LBD Flags", str(lbd_n))
        with a3: st.metric("High+",     str(sev_c.get("HIGH",0)+sev_c.get("CRITICAL",0)))
        with a4: st.metric("Eye Risk",  f"{risk*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    pc, sc = st.columns([7, 1])
    with pc:
        st.progress(pct, text=f"Eye baseline: {elapsed:.1f}s / {EYE_CAL_SEC}s  |  {rem:.0f}s remaining")
    with sc:
        if st.button("Skip", use_container_width=True):
            st.session_state.cal_done = True
            nav("mem_intro")
            st.rerun()

    if rem <= 0:
        st.session_state.cal_done = True
        nav("mem_intro")
        st.rerun()
    else:
        time.sleep(0.28)
        st.rerun()


# ── PAGE 3: GRID MEMORY INTRO ─────────────────────────────────────────────────

def page_mem_intro():
    hdr(f"Participant: {st.session_state.participant}")
    show_steps(3)
    col, _ = st.columns([2, 1], gap="large")
    with col:
        st.markdown(f"""
        <div class="card-blue">
            <div class="sec">TASK 1 OF 2  -  GRID SPATIAL MEMORY</div>
            <h2 style="color:{TEXT};margin:.25rem 0 .35rem;font-size:1.4rem;font-weight:700;">
                4x4 Grid Location Recall</h2>
            <p style="color:{MUTED};font-size:.87rem;margin-bottom:.9rem;">
                You will complete {MEM_TRIALS} trials. The number of cells increases from 3 to 5.
                You have <strong>{int(MEM_SHOW_SEC)}s to study</strong> and
                <strong>{int(MEM_RECALL_SEC)}s to respond</strong>.
            </p><hr class="div">""", unsafe_allow_html=True)
        for i, (t, d) in enumerate([
            ("Study the coloured cells", "A 4x4 grid appears with highlighted cells."),
            (f"Pattern hides after {int(MEM_SHOW_SEC)}s", "Memorise positions before they disappear."),
            ("Click cells you remember", "A blank grid appears — click what you saw."),
            ("Click again to deselect",  "Deselect wrong cells before submitting."),
            ("Feedback shown instantly", "Hit, Miss, and False Alarm cells are colour-coded."),
        ]):
            st.markdown(f"""
            <div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:.6rem;">
                <div style="background:{ACCENT};color:#fff;font-weight:700;font-size:.68rem;
                            border-radius:50%;min-width:19px;height:19px;
                            display:flex;align-items:center;justify-content:center;">{i+1}</div>
                <div>
                    <div style="color:{TEXT};font-weight:600;font-size:.84rem;">{t}</div>
                    <div style="color:{MUTED};font-size:.78rem;margin-top:1px;">{d}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("Start Memory Task", use_container_width=True):
                st.session_state.mem_trial  = 0
                st.session_state.mem_trials = []
                _mem_gen()
                nav("mem_task")
                st.rerun()


def _mem_gen():
    trial  = st.session_state.mem_trial
    n_lit  = min(3 + (trial//2), 5)
    pattern = random.sample(range(GRID_CELLS), n_lit)
    st.session_state.mem_pattern     = sorted(pattern)
    st.session_state.mem_probe_count = n_lit
    st.session_state.mem_phase       = "show"
    st.session_state.mem_phase_start = time.time()
    st.session_state.mem_selections  = []


def _draw_grid_show(pattern: list, time_frac: float):
    cw, ch, gap = 110, 72, 8
    total_w = GRID_COLS*cw + (GRID_COLS+1)*gap
    total_h = GRID_ROWS*ch + (GRID_ROWS+1)*gap
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h+44}" '
        f'style="display:block;margin:auto;background:{SURFACE};border:1.5px solid {BORDER};'
        f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.07);">'
    ]
    for cell_id in range(GRID_CELLS):
        row = cell_id//GRID_COLS; col = cell_id%GRID_COLS
        x   = gap + col*(cw+gap); y = gap + row*(ch+gap)
        lit = cell_id in pattern
        if lit:
            ci    = pattern.index(cell_id) % len(CELL_COLORS)
            fill  = CELL_COLORS[ci]; stroke = fill; tc = "#fff"; fw_ = "700"
        else:
            fill = LIGHT; stroke = BORDER; tc = BORDER; fw_ = "400"
        label = f"{chr(ord('A')+row)}{col+1}"
        svg.append(
            f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x+cw//2}" y="{y+ch//2+7}" text-anchor="middle" '
            f'font-size="20" font-family="monospace" font-weight="{fw_}" fill="{tc}">{label}</text>'
        )
    by  = total_h+10
    bw  = int(time_frac*(total_w-20))
    c_t = ACCENT if time_frac > .40 else (WARN if time_frac > .15 else DANGER)
    svg.append(
        f'<rect x="10" y="{by}" width="{total_w-20}" height="9" rx="4" fill="{LIGHT}"/>'
        f'<rect x="10" y="{by}" width="{max(0,bw)}" height="9" rx="4" fill="{c_t}"/>'
        f'<text x="{total_w//2}" y="{by+26}" text-anchor="middle" '
        f'font-size="11" font-family="monospace" fill="{MUTED}">Memorise the highlighted cells</text>'
    )
    svg.append('</svg>')
    st.markdown("".join(svg), unsafe_allow_html=True)


def _draw_grid_recall(selected: list, n_needed: int, trial: int):
    cw, ch, gap = 108, 68, 8
    total_w = GRID_COLS*cw + (GRID_COLS+1)*gap
    total_h = GRID_ROWS*ch + (GRID_ROWS+1)*gap
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h+10}" '
        f'style="display:block;margin:auto;background:{SURFACE};border:1.5px solid {BORDER};'
        f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.06);">'
    ]
    for cell_id in range(GRID_CELLS):
        row_ = cell_id//GRID_COLS; col_ = cell_id%GRID_COLS
        x    = gap + col_*(cw+gap); y = gap + row_*(ch+gap)
        is_sel = cell_id in selected
        label  = f"{chr(ord('A')+row_)}{col_+1}"
        if is_sel:
            fill="#3182CE"; stroke="#2B6CB0"; tc="#FFFFFF"; fw_="700"
        else:
            fill=LIGHT; stroke=BORDER; tc=MUTED; fw_="400"
        svg.append(
            f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x+cw//2}" y="{y+ch//2+7}" text-anchor="middle" '
            f'font-size="18" font-family="monospace" font-weight="{fw_}" fill="{tc}">{label}</text>'
        )
    svg.append('</svg>')
    st.markdown("".join(svg), unsafe_allow_html=True)

    st.markdown(f'<div style="max-width:528px;margin:6px auto 0;">', unsafe_allow_html=True)
    for row_ in range(GRID_ROWS):
        cols_st = st.columns(GRID_COLS, gap="small")
        for col_ in range(GRID_COLS):
            cell_id = row_*GRID_COLS + col_
            label   = f"{chr(ord('A')+row_)}{col_+1}"
            is_sel  = cell_id in selected
            with cols_st[col_]:
                btn_lbl = f"✓ {label}" if is_sel else label
                if st.button(btn_lbl, key=f"cell_{trial}_{cell_id}", use_container_width=True):
                    if is_sel:
                        selected.remove(cell_id)
                    else:
                        selected.append(cell_id)
                    st.session_state.mem_selections = selected
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    n_left  = n_needed - len(selected)
    msg     = f"Select {n_left} more cell(s)" if n_left > 0 else "All cells selected — press Submit"
    msg_c   = WARN if n_left > 0 else SAFE
    sel_lbl = ", ".join(f"{chr(ord('A')+i//GRID_COLS)}{i%GRID_COLS+1}"
                        for i in sorted(selected)) if selected else "none"
    st.markdown(
        f'<div style="text-align:center;margin-top:.4rem;">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.78rem;color:{msg_c};">{msg}</div>'
        f'<div style="font-size:.72rem;color:{MUTED};margin-top:2px;">Selected: {sel_lbl}</div>'
        f'</div>', unsafe_allow_html=True)


def _draw_grid_feedback(pattern: list, selected: list):
    cw, ch, gap = 110, 72, 8
    total_w = GRID_COLS*cw + (GRID_COLS+1)*gap
    total_h = GRID_ROWS*ch + (GRID_ROWS+1)*gap
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h+50}" '
        f'style="display:block;margin:auto;background:{SURFACE};border:1.5px solid {BORDER};'
        f'border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.07);">'
    ]
    for cell_id in range(GRID_CELLS):
        r = cell_id//GRID_COLS; c = cell_id%GRID_COLS
        x = gap+c*(cw+gap); y = gap+r*(ch+gap)
        in_pat = cell_id in pattern
        in_sel = cell_id in selected
        label  = f"{chr(ord('A')+r)}{c+1}"
        if in_pat and in_sel:
            fill="#9AE6B4"; stroke=SAFE;   glyph="HIT";  tc=SAFE
        elif in_pat and not in_sel:
            fill="#FEB2B2"; stroke=DANGER; glyph="MISS"; tc=DANGER
        elif not in_pat and in_sel:
            fill="#F6AD55"; stroke=WARN;   glyph="FA";   tc=WARN
        else:
            fill=LIGHT;    stroke=BORDER;  glyph=label;  tc=MUTED
        svg.append(
            f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x+cw//2}" y="{y+ch//2-5}" text-anchor="middle" '
            f'font-size="11" font-family="monospace" fill="{tc}">{label}</text>'
            f'<text x="{x+cw//2}" y="{y+ch//2+13}" text-anchor="middle" '
            f'font-size="14" font-family="monospace" font-weight="700" fill="{tc}">{glyph}</text>'
        )
    ly = total_h+13
    for i, (txt, col_) in enumerate([("HIT","#9AE6B4"),("MISS","#FEB2B2"),("False Alarm","#F6AD55")]):
        lx = 20+i*158
        svg.append(
            f'<rect x="{lx}" y="{ly}" width="13" height="13" rx="3" fill="{col_}"/>'
            f'<text x="{lx+18}" y="{ly+11}" font-size="11" font-family="monospace" fill="{MUTED}">{txt}</text>'
        )
    svg.append('</svg>')
    st.markdown("".join(svg), unsafe_allow_html=True)


def _mem_record():
    pattern  = st.session_state.mem_pattern
    selected = st.session_state.mem_selections
    rt       = time.time() - (st.session_state.mem_phase_start or time.time())
    hits     = len(set(pattern) & set(selected))
    misses   = len(set(pattern) - set(selected))
    fa       = len(set(selected) - set(pattern))
    hr       = hits / max(len(pattern), 1)
    resp = dict(
        hits=hits, misses=misses, false_alarms=fa,
        hit_rate=round(hr,4), rt=round(rt,3),
        selected=list(selected), pattern=list(pattern),
        trial=st.session_state.mem_trial+1,
        n_cells=len(pattern), accuracy=round(hr,4),
    )
    st.session_state.mem_last  = resp
    st.session_state.mem_trials.append(resp)
    st.session_state.mem_phase = "feedback"
    st.session_state.gaze_log.extend(
        GazeSimulator.simulate([(0.5,0.5)], rt*0.5, 30, 0.01))


def _mem_next():
    idx = st.session_state.mem_trial + 1
    st.session_state.mem_trial = idx
    if idx >= MEM_TRIALS:
        nav("sacc_intro")
    else:
        _mem_gen()


def page_mem_task():
    trial   = st.session_state.mem_trial
    hdr(f"Grid Memory  |  Trial {trial+1} of {MEM_TRIALS}")
    show_steps(3)

    phase   = st.session_state.mem_phase
    pattern = st.session_state.mem_pattern
    elapsed = time.time() - (st.session_state.mem_phase_start or time.time())

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trial", f"{trial+1}/{MEM_TRIALS}")
    with m2:
        prev  = st.session_state.mem_trials
        avg_h = sum(t["hit_rate"] for t in prev) / max(len(prev), 1)
        st.metric("Avg Hit Rate", f"{avg_h*100:.0f}%" if prev else "--")
    with m3:
        avg_rt = sum(t["rt"] for t in prev) / max(len(prev), 1)
        st.metric("Avg RT", f"{avg_rt:.1f}s" if prev else "--")
    with m4:
        st.metric("Cells in Pattern", str(len(pattern)))

    if phase == "show":
        rem = max(0.0, MEM_SHOW_SEC - elapsed)
        st.markdown(
            f'<p style="text-align:center;color:{ACCENT};font-weight:600;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:.84rem;margin:.3rem 0 .65rem;">'
            f'Memorise the highlighted cells  —  {rem:.1f}s remaining</p>',
            unsafe_allow_html=True)
        _draw_grid_show(pattern, rem/MEM_SHOW_SEC)
        if rem <= 0:
            st.session_state.mem_phase       = "recall"
            st.session_state.mem_phase_start = time.time()
            st.rerun()
        else:
            time.sleep(0.30); st.rerun()

    elif phase == "recall":
        rem      = max(0.0, MEM_RECALL_SEC - elapsed)
        sel      = st.session_state.mem_selections
        n_needed = len(pattern)
        st.markdown(f"""
        <div style="background:#FFFAF0;border:1px solid {WARN}44;border-radius:6px;
                    padding:.55rem .85rem;margin-bottom:.7rem;
                    display:flex;justify-content:space-between;align-items:center;">
            <span style="color:{TEXT};font-weight:600;font-size:.86rem;">
                Click the cells you remember  —  {n_needed} cells were shown
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.80rem;color:{WARN};">
                Selected: {len(sel)} / {n_needed}  |  {rem:.1f}s left
            </span>
        </div>""", unsafe_allow_html=True)
        _draw_grid_recall(sel, n_needed, trial)
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns([2, 1, 2])
        with bc2:
            if st.button("Submit Answer", use_container_width=True, key=f"mem_sub_{trial}"):
                _mem_record(); st.rerun()
        if rem <= 0:
            _mem_record(); st.rerun()

    elif phase == "feedback":
        resp   = st.session_state.mem_last
        hits   = resp["hits"]; misses = resp["misses"]
        fa     = resp["false_alarms"]; hr = resp["hit_rate"]
        fc     = SAFE if hr>=.70 else (WARN if hr>=.40 else DANGER)
        bg_fb  = "#F0FFF4" if hr>=.70 else ("#FFFAF0" if hr>=.40 else "#FFF5F5")
        verdict= "Good recall" if hr>=.70 else ("Partial recall" if hr>=.40 else "Poor recall")
        _draw_grid_feedback(pattern, resp["selected"])
        st.markdown(
            f'<div style="background:{bg_fb};border:1.5px solid {fc}44;'
            f'border-radius:7px;padding:.7rem 1rem;text-align:center;'
            f'margin-top:.6rem;max-width:520px;margin-left:auto;margin-right:auto;">'
            f'<span style="color:{fc};font-size:.96rem;font-weight:700;">{verdict}</span>'
            f'<div style="color:{MUTED};font-size:.79rem;margin-top:3px;">'
            f'Hits: {hits}  |  Misses: {misses}  |  False alarms: {fa}'
            f'  |  Hit rate: {hr*100:.0f}%  |  RT: {resp["rt"]:.2f}s'
            f'</div></div>', unsafe_allow_html=True)
        time.sleep(3.0)
        _mem_next(); st.rerun()


# ── PAGE 4: SACCADE ───────────────────────────────────────────────────────────

def page_sacc_intro():
    hdr(f"Participant: {st.session_state.participant}")
    show_steps(4)
    col, _ = st.columns([2, 1], gap="large")
    with col:
        st.markdown(f"""
        <div class="card" style="border-color:{ACCENT2}44;background:linear-gradient(135deg,#FAF5FF,#F7FAFC);">
            <div class="sec" style="color:{ACCENT2};">TASK 2 OF 2  -  SACCADIC INHIBITION</div>
            <h2 style="color:{TEXT};margin:.25rem 0 .35rem;font-size:1.4rem;font-weight:700;">
                Antisaccade Task</h2>
            <p style="color:{MUTED};font-size:.87rem;margin-bottom:.9rem;">
                Measures executive inhibitory control. You will complete {SACC_TRIALS} trials.
            </p><hr class="div">""", unsafe_allow_html=True)
        for i, (t, d) in enumerate([
            ("Fixation cross appears",        "Keep your eyes locked on the central +."),
            ("A red dot appears on one side",  "It flashes LEFT or RIGHT."),
            ("Look to the OPPOSITE side",      "Suppress the reflex. Look to the other side."),
            ("Press LEFT or RIGHT button",     "Click the button OPPOSITE to the dot."),
            ("RT and accuracy recorded",       "Express saccades (<120ms) and errors are flagged."),
        ]):
            st.markdown(f"""
            <div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:.55rem;">
                <div style="background:{ACCENT2};color:#fff;font-weight:700;font-size:.67rem;
                            border-radius:50%;min-width:19px;height:19px;
                            display:flex;align-items:center;justify-content:center;">{i+1}</div>
                <div>
                    <div style="color:{TEXT};font-weight:600;font-size:.83rem;">{t}</div>
                    <div style="color:{MUTED};font-size:.77rem;margin-top:1px;">{d}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#FFF5F5;border:1px solid {DANGER}44;border-radius:5px;'
            f'padding:.5rem .7rem;margin-top:.3rem;font-size:.78rem;color:{DANGER};font-weight:600;">'
            f'Remember: Press the button OPPOSITE to where the dot appears.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("Start Saccade Task", use_container_width=True):
                st.session_state.sacc_trial  = 0
                st.session_state.sacc_trials = []
                _sacc_gen()
                nav("sacc_task"); st.rerun()


def _sacc_gen():
    side = random.choice(["left","right"])
    st.session_state.sacc_side        = side
    st.session_state.sacc_correct     = "right" if side=="left" else "left"
    st.session_state.sacc_phase       = "fixation"
    st.session_state.sacc_phase_start = time.time()
    st.session_state.sacc_last        = {}


def page_sacc_task():
    trial   = st.session_state.sacc_trial
    hdr(f"Antisaccade  |  Trial {trial+1} of {SACC_TRIALS}")
    show_steps(4)

    phase   = st.session_state.sacc_phase
    elapsed = time.time() - (st.session_state.sacc_phase_start or time.time())
    W_S, H_S = 700, 260

    n_done = len(st.session_state.sacc_trials)
    err    = sum(t["error"] for t in st.session_state.sacc_trials)
    avg_rt = sum(t["rt"] for t in st.session_state.sacc_trials) / max(n_done, 1)
    er_pct = err / max(n_done, 1) * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trial",   f"{trial+1}/{SACC_TRIALS}")
    with m2: st.metric("Errors",  f"{err}/{n_done}" if n_done else "--")
    with m3: st.metric("Avg RT",  f"{avg_rt*1000:.0f}ms" if n_done else "--")
    with m4:
        er_c = DANGER if er_pct > 40 else (WARN if er_pct > 20 else SAFE)
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;color:{MUTED};">ERROR RATE</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:{er_c};font-family:\'JetBrains Mono\',monospace;">'
            f'{"--" if not n_done else f"{er_pct:.0f}%"}</div></div>',
            unsafe_allow_html=True)

    cells = ""
    for i in range(SACC_TRIALS):
        if i < len(st.session_state.sacc_trials):
            t_  = st.session_state.sacc_trials[i]
            ok  = not t_["error"]
            bg_ = "#F0FFF4" if ok else "#FFF5F5"
            fg_ = SAFE     if ok else DANGER
            bd_ = "#9AE6B4" if ok else "#FEB2B2"
            sym = "+" if ok else "x"
        elif i == trial:
            bg_="#EBF8FF"; fg_=ACCENT; bd_="#90CDF4"; sym=str(i+1)
        else:
            bg_=LIGHT; fg_=BORDER; bd_=BORDER; sym=str(i+1)
        cells += (
            f'<div style="background:{bg_};border:1.5px solid {bd_};border-radius:5px;'
            f'padding:5px 0;text-align:center;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.78rem;font-weight:700;color:{fg_};">{sym}</div>'
        )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin-bottom:.65rem;">'
        f'{cells}</div>', unsafe_allow_html=True)

    def sacc_canvas(show_dot=False):
        cx_, cy_ = W_S//2, H_S//2
        p = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_S}" height="{H_S}" '
            f'style="background:#F7FAFC;border:1.5px solid {BORDER};border-radius:8px;'
            f'display:block;margin:auto;box-shadow:0 1px 4px rgba(0,0,0,.05);">',
            grid_svg_bg(W_S, H_S),
        ]
        p.append(
            f'<line x1="{cx_}" y1="18" x2="{cx_}" y2="{H_S-18}" '
            f'stroke="{BORDER}" stroke-width="1" stroke-dasharray="4,3"/>'
            f'<text x="{W_S//4}" y="{H_S//2+6}" text-anchor="middle" '
            f'font-size="16" fill="{BORDER}" font-weight="600">LEFT</text>'
            f'<text x="{3*W_S//4}" y="{H_S//2+6}" text-anchor="middle" '
            f'font-size="16" fill="{BORDER}" font-weight="600">RIGHT</text>'
        )
        if not show_dot:
            rem = max(0, SACC_FIX_SEC - elapsed)
            p.append(
                f'<line x1="{cx_-22}" y1="{cy_}" x2="{cx_+22}" y2="{cy_}" stroke="{TEXT}" stroke-width="3"/>'
                f'<line x1="{cx_}" y1="{cy_-22}" x2="{cx_}" y2="{cy_+22}" stroke="{TEXT}" stroke-width="3"/>'
                f'<circle cx="{cx_}" cy="{cy_}" r="4" fill="{ACCENT}"/>'
                f'<text x="{cx_}" y="{H_S-12}" text-anchor="middle" font-size="11" '
                f'fill="{MUTED}" font-family="monospace">Hold fixation  -  {rem:.1f}s</text>'
            )
        else:
            side  = st.session_state.sacc_side
            corr  = st.session_state.sacc_correct
            dot_x = W_S//4  if side=="left"  else 3*W_S//4
            cx2   = 3*W_S//4 if side=="left" else W_S//4
            rem   = max(0, SACC_STIM_SEC - elapsed)
            arrow = ">" if side=="left" else "<"
            p.append(
                f'<circle cx="{dot_x}" cy="{cy_}" r="22" fill="{DANGER}22"/>'
                f'<circle cx="{dot_x}" cy="{cy_}" r="15" fill="{DANGER}"/>'
                f'<circle cx="{dot_x+4}" cy="{cy_-4}" r="5" fill="#fff" opacity="0.45"/>'
                f'<text x="{dot_x}" y="{cy_-30}" text-anchor="middle" font-size="9" '
                f'fill="{DANGER}" font-family="monospace" font-weight="700">STIMULUS</text>'
                f'<text x="{cx2}" y="{cy_+10}" text-anchor="middle" font-size="30" '
                f'fill="{SAFE}55" font-weight="700">{arrow}</text>'
                f'<text x="{cx2}" y="{cy_-32}" text-anchor="middle" font-size="10" '
                f'fill="{SAFE}" font-family="monospace" font-weight="700">LOOK HERE</text>'
                f'<text x="{cx_}" y="{H_S-12}" text-anchor="middle" font-size="11" '
                f'fill="{ACCENT}" font-family="monospace" font-weight="600">'
                f'Look OPPOSITE to the dot  ({rem:.1f}s)</text>'
            )
        p.append('</svg>')
        return "".join(p)

    if phase == "fixation":
        rem = max(0, SACC_FIX_SEC - elapsed)
        st.markdown(
            f'<p style="text-align:center;color:{MUTED};font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.84rem;margin-bottom:.45rem;">'
            f'Focus on the central cross  -  stimulus in {rem:.1f}s</p>',
            unsafe_allow_html=True)
        st.markdown(sacc_canvas(False), unsafe_allow_html=True)
        st.session_state.gaze_log.extend(GazeSimulator.simulate([(0.5,0.5)], 0.2, 30, 0.005))
        if rem <= 0:
            st.session_state.sacc_phase       = "stimulus"
            st.session_state.sacc_phase_start = time.time()
            st.rerun()
        else:
            time.sleep(0.18); st.rerun()

    elif phase == "stimulus":
        side = st.session_state.sacc_side
        corr = st.session_state.sacc_correct
        rem  = max(0, SACC_STIM_SEC - elapsed)
        st.markdown(
            f'<div style="text-align:center;margin-bottom:.45rem;">'
            f'<span style="background:{DANGER}12;border:1.5px solid {DANGER}38;'
            f'border-radius:5px;padding:4px 14px;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.88rem;color:{DANGER};font-weight:700;">'
            f'Dot is {side.upper()}  -  Look {corr.upper()}</span></div>',
            unsafe_allow_html=True)
        st.markdown(sacc_canvas(True), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        bl, br_ = st.columns(2, gap="large")
        with bl:
            if st.button("LEFT", key=f"sl_{trial}", use_container_width=True):
                _sacc_record("left"); st.rerun()
            st.markdown(
                f'<div style="text-align:center;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.72rem;color:{SAFE if corr=="left" else DANGER};">'
                f'{"CORRECT" if corr=="left" else "ERROR"}</div>', unsafe_allow_html=True)
        with br_:
            if st.button("RIGHT", key=f"sr_{trial}", use_container_width=True):
                _sacc_record("right"); st.rerun()
            st.markdown(
                f'<div style="text-align:center;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:.72rem;color:{SAFE if corr=="right" else DANGER};">'
                f'{"CORRECT" if corr=="right" else "ERROR"}</div>', unsafe_allow_html=True)
        if rem <= 0:
            _sacc_record("timeout"); st.rerun()
        else:
            time.sleep(0.18); st.rerun()

    elif phase == "feedback":
        resp  = st.session_state.sacc_last
        error = resp.get("error", 1)
        rt    = resp.get("rt", 0)
        rs    = resp.get("response", "timeout")
        fc    = SAFE if not error else DANGER
        bg_fb = "#F0FFF4" if not error else "#FFF5F5"
        if rs == "timeout":
            lbl = "Timeout - no response recorded"
        elif not error:
            lbl = f"Correct - looked {st.session_state.sacc_correct.upper()}"
        else:
            lbl = "Error - reflexive saccade toward the stimulus"

        st.markdown(sacc_canvas(False), unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{bg_fb};border:1.5px solid {fc}44;'
            f'border-radius:7px;padding:.65rem .95rem;text-align:center;'
            f'margin-top:.55rem;max-width:500px;margin-left:auto;margin-right:auto;">'
            f'<span style="color:{fc};font-size:.94rem;font-weight:700;">{lbl}</span>'
            f'<div style="color:{MUTED};font-size:.78rem;margin-top:3px;">'
            f'RT: {rt*1000:.0f}ms  |  Stimulus: {st.session_state.sacc_side.upper()}'
            f'  |  Correct: {st.session_state.sacc_correct.upper()}'
            f'  |  Response: {rs.upper()}</div></div>',
            unsafe_allow_html=True)
        cx_g = 0.15 if rs=="left" else (0.85 if rs=="right" else 0.5)
        st.session_state.gaze_log.extend(GazeSimulator.simulate([(cx_g,0.5)], 0.6, 30, 0.01))
        det: AnomalyDetector = st.session_state.det
        det.process_saccade(
            latency_ms=rt*1000, peak_vel=float(random.gauss(265,55)),
            amplitude=float(random.gauss(8,2)),
            cx=cx_g, cy=0.5, is_anti=True, correct=(not error),
        )
        time.sleep(1.0)
        _sacc_next(); st.rerun()


def _sacc_record(direction: str):
    rt    = time.time() - (st.session_state.sacc_phase_start or time.time())
    error = int(direction != st.session_state.sacc_correct) if direction in ("left","right") else 1
    resp  = {"response": direction, "rt": round(rt,4), "error": error}
    st.session_state.sacc_last = resp
    st.session_state.sacc_trials.append({
        "trial":        st.session_state.sacc_trial+1,
        "stim_side":    st.session_state.sacc_side,
        "correct_side": st.session_state.sacc_correct,
        "response":     direction,
        "error":        error,
        "rt":           round(rt,4),
    })
    st.session_state.sacc_phase = "feedback"


def _sacc_next():
    idx = st.session_state.sacc_trial + 1
    st.session_state.sacc_trial = idx
    if idx >= SACC_TRIALS:
        nav("report")
    else:
        _sacc_gen()


# ── PAGE 5: REPORT ────────────────────────────────────────────────────────────

def page_report():
    hdr(f"Report  |  {st.session_state.participant}  |  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    show_steps(5)

    mem_trials  = st.session_state.mem_trials
    sacc_trials = st.session_state.sacc_trials
    gaze_log    = st.session_state.gaze_log
    det: AnomalyDetector = st.session_state.det

    mem_hr  = sum(t["hit_rate"]     for t in mem_trials)  / max(len(mem_trials),1)
    mem_rt  = sum(t["rt"]           for t in mem_trials)  / max(len(mem_trials),1)
    mem_fa  = sum(t["false_alarms"] for t in mem_trials)  / max(len(mem_trials),1)
    s_err   = sum(t["error"]        for t in sacc_trials) / max(len(sacc_trials),1)
    s_rt    = sum(t["rt"]           for t in sacc_trials) / max(len(sacc_trials),1)

    ext   = FeatureExtractor()
    feats = {
        **ext.extract(gaze_log),
        "reaction_time":          mem_rt,
        "accuracy":               mem_hr,
        "antisaccade_error_rate": s_err,
        "antisaccade_rt":         s_rt,
    }

    clf = load_classifier()
    res = clf.predict(feats)

    mem_risk  = float(np.clip(
        (1.0 - mem_hr) * 0.50 +
        float(np.clip((mem_rt - 5.0) / 15.0, 0, 1)) * 0.25 +
        float(np.clip(mem_fa / 3.0, 0, 1)) * 0.25,
        0, 1))
    sacc_risk = float(np.clip(s_err, 0, 1))
    anom_r    = det.get_risk_score()
    combined  = float(np.clip(
        mem_risk * 0.35 + sacc_risk * 0.35 + anom_r * 0.30, 0, 1))
    pct       = round(combined * 100)

    if combined >= 0.66:
        risk_level = "HIGH";     rc = DANGER; v_cls = "v-high"
    elif combined >= 0.33:
        risk_level = "MODERATE"; rc = WARN;   v_cls = "v-mod"
    else:
        risk_level = "LOW";      rc = SAFE;   v_cls = "v-low"

    n_anom  = len(det.events)
    lbd_n   = sum(1 for e in det.events if e.is_lbd_biomarker)
    sev_cnt = Counter(e.severity    for e in det.events)
    typ_cnt = Counter(e.anomaly_type for e in det.events)

    st.markdown('<div class="sec">CLINICAL RISK ASSESSMENT</div>', unsafe_allow_html=True)
    va, vb = st.columns([5, 2], gap="large")

    with va:
        risk_descriptions = {
            "HIGH":     (f"HIGH RISK — {pct}% combined score.",
                         "Neurological evaluation is strongly recommended."),
            "MODERATE": (f"MODERATE RISK — {pct}% combined score.",
                         "Routine neurological follow-up is advised."),
            "LOW":      (f"LOW RISK — {pct}% combined score.",
                         "Performance within expected range. Routine monitoring advised."),
        }
        tier_txt, rec_txt = risk_descriptions[risk_level]
        mem_score_pct  = round(mem_risk * 100)
        sacc_score_pct = round(sacc_risk * 100)
        anom_score_pct = round(anom_r * 100)
        st.markdown(f"""
        <div class="{v_cls}">
            <div class="sec" style="color:{MUTED};margin-bottom:4px;">CLASSIFICATION OUTCOME</div>
            <div style="font-size:1.85rem;font-weight:800;color:{rc};line-height:1.2;">{risk_level} RISK</div>
            <div style="color:{rc};font-weight:600;font-size:.86rem;margin:3px 0;
                        font-family:'JetBrains Mono',monospace;">{tier_txt}</div>
            <div style="color:{MUTED};font-size:.83rem;line-height:1.65;margin-top:5px;">{rec_txt}</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:10px;">
                <div style="background:{SURFACE};border-radius:5px;padding:.45rem;text-align:center;border:1px solid {BORDER};">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{MUTED};">MEMORY</div>
                    <div style="font-size:1.05rem;font-weight:700;color:{DANGER if mem_score_pct>=66 else (WARN if mem_score_pct>=33 else SAFE)};">{mem_score_pct}%</div>
                </div>
                <div style="background:{SURFACE};border-radius:5px;padding:.45rem;text-align:center;border:1px solid {BORDER};">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{MUTED};">SACCADE</div>
                    <div style="font-size:1.05rem;font-weight:700;color:{DANGER if sacc_score_pct>=66 else (WARN if sacc_score_pct>=33 else SAFE)};">{sacc_score_pct}%</div>
                </div>
                <div style="background:{SURFACE};border-radius:5px;padding:.45rem;text-align:center;border:1px solid {BORDER};">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{MUTED};">EYE ANOMALY</div>
                    <div style="font-size:1.05rem;font-weight:700;color:{DANGER if anom_score_pct>=66 else (WARN if anom_score_pct>=33 else SAFE)};">{anom_score_pct}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with vb:
        fig, ax = plt.subplots(figsize=(2.3,2.3), facecolor=SURFACE)
        th = np.linspace(0, 2*math.pi, 300)
        ax.plot(np.cos(th)*0.72, np.sin(th)*0.72, lw=11, color=LIGHT, solid_capstyle="round")
        ft = np.linspace(math.pi/2, math.pi/2-2*math.pi*pct/100, 300)
        if len(ft) > 1:
            ax.plot(np.cos(ft)*0.72, np.sin(ft)*0.72, lw=11, color=rc, solid_capstyle="round")
        ax.text(0, .10, f"{pct}%", ha="center", va="center",
                fontsize=20, fontweight="bold", color=rc, fontfamily="monospace")
        ax.text(0, -.22, "LBD RISK", ha="center", va="center",
                fontsize=7, color=MUTED, fontfamily="monospace")
        ax.set_xlim(-1,1); ax.set_ylim(-1,1)
        ax.set_aspect("equal"); ax.axis("off")
        fig.tight_layout(pad=0.1)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">THREE-LEVEL RISK CLASSIFICATION</div>', unsafe_allow_html=True)
    three_levels = [
        ("LEVEL 1","LOW",      SAFE,   "#F0FFF4","0–33%",
         "Performance within normal range. No significant LBD indicators."),
        ("LEVEL 2","MODERATE", WARN,   "#FFFAF0","33–66%",
         "Some LBD-associated patterns observed. Follow-up advised."),
        ("LEVEL 3","HIGH",     DANGER, "#FFF5F5","66–100%",
         "Multiple LBD biomarkers detected. Neurological evaluation strongly recommended."),
    ]
    level_html = ""
    for lv_lbl, lv_name, lv_c, lv_bg, lv_range, lv_desc in three_levels:
        is_current = lv_name == risk_level
        bw  = "3px" if is_current else "1.5px"
        op  = "1"   if is_current else "0.38"
        bg_val  = lv_bg  if is_current else SURFACE
        bd_col  = lv_c   if is_current else lv_c+"88"
        cur_badge = (f"<div style='margin-top:8px;font-weight:700;font-size:.75rem;"
                     f"color:{lv_c};font-family:JetBrains Mono,monospace;'>"
                     f"&#9658; CURRENT RESULT</div>") if is_current else ""
        level_html += (
            f'<div style="background:{bg_val};border:{bw} solid {bd_col};'
            f'border-radius:8px;padding:1rem;text-align:center;opacity:{op};">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.62rem;'
            f'color:{lv_c};letter-spacing:.09em;margin-bottom:4px;">{lv_lbl} · {lv_range}</div>'
            f'<div style="font-size:1.15rem;font-weight:800;color:{lv_c};margin-bottom:6px;">{lv_name}</div>'
            f'<div style="font-size:.71rem;color:{MUTED};line-height:1.55;">{lv_desc}</div>'
            f'{cur_badge}</div>'
        )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:.9rem;">'
        + level_html + '</div>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    col_eye, col_mem, col_sac = st.columns(3, gap="large")

    with col_eye:
        st.markdown(f'<div class="sec">EYE ANOMALY ANALYSIS</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("Total",    str(n_anom))
        with m2: st.metric("LBD",      str(lbd_n))
        m3, m4 = st.columns(2)
        with m3: st.metric("Critical", str(sev_cnt.get("CRITICAL",0)))
        with m4: st.metric("High",     str(sev_cnt.get("HIGH",0)))

        sev_labels = ["Critical","High","Medium","Low"]
        sev_vals   = [sev_cnt.get(s,0) for s in ["CRITICAL","HIGH","MEDIUM","LOW"]]
        sev_colors = [DANGER, WARN, ACCENT, SAFE]
        if sum(sev_vals) > 0:
            fig2, ax2 = plt.subplots(figsize=(2.8,2.2), facecolor=SURFACE)
            ax2.set_facecolor(SURFACE)
            ax2.pie(
                sev_vals, labels=sev_labels, colors=sev_colors,
                autopct="%1.0f%%", startangle=90,
                pctdistance=0.78, labeldistance=1.15,
                textprops={"fontsize":6.5,"color":MUTED,"fontfamily":"monospace"},
                wedgeprops={"linewidth":0.4,"edgecolor":"white"},
                radius=0.9,
            )
            centre_circle = plt.Circle((0,0), 0.55, fc=SURFACE)
            ax2.add_artist(centre_circle)
            ax2.text(0, 0, f"{n_anom}\nevents", ha="center", va="center",
                     fontsize=7, color=TEXT, fontfamily="monospace", fontweight="bold")
            ax2.set_title("Severity breakdown", color=MUTED, fontsize=7, fontfamily="monospace", pad=3)
            fig2.tight_layout(pad=0.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        if det.events:
            st.markdown('<div class="sec" style="margin-top:.5rem;">TOP ANOMALIES</div>',
                        unsafe_allow_html=True)
            rows_html = ""
            for atype, cnt in typ_cnt.most_common(6):
                sev  = next((e.severity for e in det.events if e.anomaly_type==atype), "LOW")
                is_l = next((e.is_lbd_biomarker for e in det.events if e.anomaly_type==atype), False)
                sc2  = "b-red" if sev in ("CRITICAL","HIGH") else ("b-amber" if sev=="MEDIUM" else "b-grey")
                lb2  = '<span class="badge b-red" style="margin-left:3px;font-size:.60rem;">LBD</span>' if is_l else ""
                rows_html += (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'padding:4px 0;border-bottom:1px solid {BORDER};">'
                    f'<div style="flex:1;font-size:.74rem;font-weight:500;color:{TEXT};'
                    f'font-family:\'JetBrains Mono\',monospace;">'
                    f'{atype.replace("_"," ")}</div>'
                    f'<div style="display:flex;align-items:center;gap:3px;">{lb2}'
                    f'<span class="badge {sc2}">{sev[:3]}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.70rem;'
                    f'color:{MUTED};">x{cnt}</span></div></div>'
                )
            st.markdown(f'<div class="card" style="padding:.5rem .65rem;">{rows_html}</div>',
                        unsafe_allow_html=True)

    with col_mem:
        st.markdown(f'<div class="sec">MEMORY TASK ANALYSIS</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("Hit Rate", f"{mem_hr*100:.0f}%")
        with m2: st.metric("Mean RT",  f"{mem_rt:.1f}s")
        m3, m4 = st.columns(2)
        with m3: st.metric("False Alarms", f"{mem_fa:.1f}/trial")
        with m4:
            total_misses = sum(t["misses"] for t in mem_trials)
            st.metric("Total Misses", str(total_misses))

        if mem_trials:
            fig3, ax3  = plt.subplots(figsize=(2.8,2.2), facecolor=SURFACE)
            ax3.set_facecolor("#F7FAFC")
            xs3  = [t["trial"]    for t in mem_trials]
            hrs3 = [t["hit_rate"] for t in mem_trials]
            rts3 = [t["rt"]       for t in mem_trials]
            ax3_r = ax3.twinx()
            ax3.bar(xs3, hrs3, color=ACCENT, alpha=0.65, width=0.55, label="Hit rate")
            ax3_r.plot(xs3, rts3, color=DANGER, lw=1.5, marker="o", markersize=4, label="RT (s)")
            ax3.axhline(0.7, color=SAFE, lw=0.9, ls="--", alpha=0.7)
            ax3.set_ylim(0,1.1)
            ax3.set_xlabel("Trial", color=MUTED, fontsize=6)
            ax3.set_ylabel("Hit rate", color=ACCENT, fontsize=6)
            ax3_r.set_ylabel("RT (s)", color=DANGER, fontsize=6)
            ax3.tick_params(colors=MUTED, labelsize=5.5)
            ax3_r.tick_params(colors=DANGER, labelsize=5.5)
            for sp in ax3.spines.values(): sp.set_color(BORDER)
            ax3.set_title("Hit rate & RT per trial",
                          color=MUTED, fontsize=6.5, fontfamily="monospace")
            fig3.tight_layout(pad=0.3)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            rows3 = ""
            for t in mem_trials:
                hr_c = SAFE if t["hit_rate"]>=.70 else (WARN if t["hit_rate"]>=.40 else DANGER)
                rows3 += (
                    f'<div style="display:grid;grid-template-columns:30px 1fr 1fr 1fr 1fr;'
                    f'gap:4px;padding:3px 0;border-bottom:1px solid {BORDER};font-size:.72rem;'
                    f'font-family:\'JetBrains Mono\',monospace;color:{MUTED};">'
                    f'<div>{t["trial"]}</div>'
                    f'<div style="color:{hr_c};">{t["hit_rate"]*100:.0f}%</div>'
                    f'<div>{t["rt"]:.1f}s</div>'
                    f'<div style="color:{SAFE};">{t["hits"]} H</div>'
                    f'<div style="color:{DANGER};">{t["misses"]} M / {t["false_alarms"]} FA</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;'
                f'color:{MUTED};display:grid;grid-template-columns:30px 1fr 1fr 1fr 1fr;'
                f'gap:4px;padding-bottom:3px;border-bottom:1.5px solid {BORDER};">'
                f'<div>#</div><div>HIT%</div><div>RT</div><div>Hits</div><div>M/FA</div></div>'
                f'<div class="card" style="padding:.4rem .55rem;margin-top:.2rem;">{rows3}</div>',
                unsafe_allow_html=True)

    with col_sac:
        st.markdown(f'<div class="sec">SACCADE TASK ANALYSIS</div>', unsafe_allow_html=True)
        n_corr = sum(1 for t in sacc_trials if not t["error"])
        n_err  = sum(t["error"] for t in sacc_trials)
        n_to   = sum(1 for t in sacc_trials if t["response"]=="timeout")
        m1, m2 = st.columns(2)
        with m1: st.metric("Error Rate", f"{s_err*100:.0f}%")
        with m2: st.metric("Mean RT",    f"{s_rt*1000:.0f}ms")
        m3, m4 = st.columns(2)
        with m3: st.metric("Correct", str(n_corr))
        with m4: st.metric("Errors",  str(n_err))

        if sacc_trials:
            fig4, ax4 = plt.subplots(figsize=(2.8,2.2), facecolor=SURFACE)
            ax4.set_facecolor("#F7FAFC")
            xs4  = [t["trial"] for t in sacc_trials]
            rts4 = [t["rt"]*1000 for t in sacc_trials]
            cols4 = [DANGER if t["error"] else SAFE for t in sacc_trials]
            ax4.bar(xs4, rts4, color=cols4, alpha=0.75, width=0.6)
            ax4.axhline(120, color=WARN,   lw=1.0, ls="--", alpha=0.8)
            ax4.axhline(800, color=DANGER, lw=0.8, ls=":",  alpha=0.6)
            ax4.set_xlabel("Trial", color=MUTED, fontsize=6)
            ax4.set_ylabel("RT (ms)", color=MUTED, fontsize=6)
            ax4.tick_params(colors=MUTED, labelsize=5.5)
            for sp in ax4.spines.values(): sp.set_color(BORDER)
            handles = [
                mpatches.Patch(color=SAFE,   label="Correct"),
                mpatches.Patch(color=DANGER, label="Error"),
            ]
            ax4.legend(handles=handles, fontsize=5.5, facecolor=SURFACE,
                       edgecolor=BORDER, loc="upper right")
            ax4.set_title("RT per trial", color=MUTED, fontsize=6.5, fontfamily="monospace")
            fig4.tight_layout(pad=0.3)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

            rows4 = ""
            for t in sacc_trials:
                err_c = DANGER if t["error"] else SAFE
                sym   = "ERR" if t["error"] else "OK"
                rows4 += (
                    f'<div style="display:grid;grid-template-columns:28px 30px 40px 1fr 1fr;'
                    f'gap:4px;padding:3px 0;border-bottom:1px solid {BORDER};font-size:.72rem;'
                    f'font-family:\'JetBrains Mono\',monospace;color:{MUTED};">'
                    f'<div>{t["trial"]}</div>'
                    f'<div style="color:{err_c};font-weight:600;">{sym}</div>'
                    f'<div>{t["rt"]*1000:.0f}ms</div>'
                    f'<div>{t["stim_side"].upper()}</div>'
                    f'<div>{t["response"].upper()}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;'
                f'color:{MUTED};display:grid;grid-template-columns:28px 30px 40px 1fr 1fr;'
                f'gap:4px;padding-bottom:3px;border-bottom:1.5px solid {BORDER};">'
                f'<div>#</div><div>Res</div><div>RT</div><div>Stim</div><div>Resp</div></div>'
                f'<div class="card" style="padding:.4rem .55rem;margin-top:.2rem;">{rows4}</div>',
                unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec">GAZE DENSITY HEATMAP  +  FEATURE PROFILE</div>',
                unsafe_allow_html=True)
    hcol, rcol = st.columns([3, 2], gap="large")

    with hcol:
        if len(gaze_log) > 10:
            fig5, ax5 = plt.subplots(figsize=(5,3.2), facecolor=SURFACE)
            ax5.set_facecolor("#EBF8FF")
            xs5 = [p[0] for p in gaze_log]
            ys5 = [p[1] for p in gaze_log]
            try:
                hmap, _, _ = np.histogram2d(xs5, ys5, bins=30, range=[[0,1],[0,1]])
                ax5.imshow(gaussian_filter(hmap,1.5).T, origin="upper",
                           extent=[0,1,1,0], cmap="plasma", alpha=0.80, aspect="auto")
            except Exception:
                pass
            ax5.set_xlabel("Gaze X (norm)", color=MUTED, fontsize=7)
            ax5.set_ylabel("Gaze Y (norm)", color=MUTED, fontsize=7)
            ax5.tick_params(colors=MUTED, labelsize=6)
            for sp in ax5.spines.values(): sp.set_color(BORDER)
            ax5.set_title(f"Full session gaze density  ({len(gaze_log)} samples)",
                          color=MUTED, fontsize=7.5, fontfamily="monospace")
            fig5.tight_layout(pad=0.3)
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

    with rcol:
        feat_labels = {
            "fixation_duration":     "Fixation dur (s)",
            "saccade_frequency":     "Saccade freq (/s)",
            "scan_path_length":      "Scan path",
            "gaze_variability":      "Gaze variability",
            "reaction_time":         "Reaction time (s)",
            "accuracy":              "Memory accuracy",
            "antisaccade_error_rate":"Antisaccade error",
            "antisaccade_rt":        "Antisaccade RT (s)",
        }
        ref_vals = {
            "fixation_duration":0.24, "saccade_frequency":3.0,
            "scan_path_length":0.35,  "gaze_variability":0.08,
            "reaction_time":1.1,      "accuracy":0.84,
            "antisaccade_error_rate":0.10, "antisaccade_rt":0.36,
        }
        names_ = list(feat_labels.values())
        vals_  = []
        ref_n_ = []
        for k in FEATURE_COLS:
            v    = feats.get(k, 0)
            r    = ref_vals.get(k, 1)
            ceil = max(v, r)*1.5 if max(v, r) > 0 else 1
            vals_.append(float(np.clip(v/ceil, 0, 1)))
            ref_n_.append(float(np.clip(r/ceil, 0, 1)))

        fig6, ax6 = plt.subplots(figsize=(3.5,3.4), facecolor=SURFACE)
        ax6.set_facecolor("#F7FAFC")
        y6 = np.arange(len(names_))
        h6 = 0.36
        ax6.barh(y6+h6/2, vals_,  h6, color=rc,    alpha=0.75, label="Participant")
        ax6.barh(y6-h6/2, ref_n_, h6, color=BORDER, alpha=0.80, label="Normal range")
        ax6.set_yticks(y6)
        ax6.set_yticklabels(names_, fontsize=6, color=MUTED, fontfamily="monospace")
        ax6.set_xlabel("Normalised value", color=MUTED, fontsize=6)
        ax6.tick_params(colors=MUTED, labelsize=5.5)
        for sp in ax6.spines.values(): sp.set_color(BORDER)
        ax6.legend(fontsize=6, facecolor=SURFACE, edgecolor=BORDER, loc="lower right")
        ax6.set_title("Feature profile vs normal range",
                      color=MUTED, fontsize=7, fontfamily="monospace")
        ax6.grid(axis="x", color=BORDER, lw=0.4, alpha=0.55)
        fig6.tight_layout(pad=0.3)
        st.pyplot(fig6, use_container_width=True)
        plt.close(fig6)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec">CLINICAL NOTES</div>', unsafe_allow_html=True)
    high_ev = [e for e in det.events if e.severity in ("HIGH","CRITICAL") and e.clinical_note]
    if high_ev:
        seen      = set()
        note_cols = st.columns(2, gap="large")
        col_idx   = 0
        for ev in high_ev[:6]:
            if ev.anomaly_type in seen:
                continue
            seen.add(ev.anomaly_type)
            sc3    = DANGER if ev.severity == "CRITICAL" else WARN
            bg3    = "#FFF5F5" if ev.severity == "CRITICAL" else "#FFFAF0"
            bd_cls = "b-red" if ev.severity == "CRITICAL" else "b-amber"
            lbd3   = ('<span class="badge b-red" style="margin-left:4px;font-size:.60rem;">LBD</span>'
                      if ev.is_lbd_biomarker else "")
            with note_cols[col_idx % 2]:
                st.markdown(
                    f'<div style="background:{bg3};border:1px solid {sc3}33;'
                    f'border-radius:7px;padding:.55rem .8rem;margin-bottom:.5rem;">'
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                    f'<span class="badge {bd_cls}">{ev.severity}</span>'
                    f'<b style="font-size:.80rem;color:{TEXT};font-family:JetBrains Mono,monospace;">'
                    f'{ev.anomaly_type.replace("_"," ")}</b>{lbd3}</div>'
                    f'<div style="font-size:.76rem;color:{MUTED};line-height:1.6;">{ev.clinical_note}</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:.67rem;'
                    f'color:{MUTED};margin-top:3px;opacity:.8;">'
                    f'val={ev.measured_value:.3f} {ev.unit} · threshold={ev.threshold_value:.3f}</div>'
                    f'</div>', unsafe_allow_html=True)
            col_idx += 1
    else:
        st.markdown(
            f'<div style="background:#F0FFF4;border:1px solid #9AE6B4;border-radius:7px;'
            f'padding:.8rem 1rem;font-size:.84rem;color:{SAFE};">'
            f'✓  No HIGH or CRITICAL anomalies detected during this session.</div>',
            unsafe_allow_html=True)

    payload = {
        "participant_id": st.session_state.participant,
        "age":            st.session_state.age,
        "session_date":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "classification": {
            "risk_level":  risk_level,
            "combined":    round(combined, 4),
            "memory_risk": round(mem_risk, 4),
            "sacc_risk":   round(sacc_risk, 4),
            "anomaly_risk":round(anom_r, 4),
        },
        "features": {k: round(v, 4) for k, v in feats.items()},
        "memory_task": {
            "mean_hit_rate":    round(mem_hr, 4),
            "mean_rt":          round(mem_rt, 4),
            "mean_false_alarms":round(mem_fa, 4),
            "trials":           mem_trials,
        },
        "saccade_task": {
            "error_rate": round(s_err, 4),
            "mean_rt":    round(s_rt, 4),
            "n_correct":  n_corr,
            "n_errors":   n_err,
            "n_timeouts": n_to,
            "trials":     sacc_trials,
        },
        "eye_anomalies": {
            "total":      n_anom,
            "lbd_flags":  lbd_n,
            "by_severity":dict(sev_cnt),
            "by_type":    dict(typ_cnt.most_common()),
            "events":     [asdict(e) for e in det.events],
        },
    }
    fname = os.path.join(RESULTS_DIR,
        f"lbd_{st.session_state.participant}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{SURFACE},{BG});border:1.5px solid {BORDER};
                border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:.8rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
            <div>
                <div class="sec" style="margin-bottom:4px;">SESSION COMPLETE</div>
                <div style="font-size:1.1rem;font-weight:700;color:{TEXT};">
                    {st.session_state.participant}  ·  {datetime.now().strftime('%d %b %Y %H:%M')}
                </div>
                <div style="font-size:.78rem;color:{MUTED};margin-top:3px;">
                    Memory: {mem_hr*100:.0f}% accuracy  ·
                    Saccade: {(1-s_err)*100:.0f}% correct  ·
                    Anomalies: {n_anom} detected  ·
                    Combined risk: <strong style="color:{rc};">{pct}% {risk_level}</strong>
                </div>
            </div>
            <div style="background:{rc}18;border:1.5px solid {rc}44;border-radius:50%;
                        width:52px;height:52px;display:flex;align-items:center;justify-content:center;
                        font-size:1.2rem;font-weight:800;color:{rc};font-family:'JetBrains Mono',monospace;">
                {pct}%
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    sa_col, sb_col, sc_col = st.columns([2, 2, 3], gap="medium")
    with sa_col:
        st.download_button(
            label="⬇  Download JSON Report",
            data=json.dumps(payload, indent=2),
            file_name=f"lbd_{st.session_state.participant}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with sb_col:
        if st.button("↺  New Assessment", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with sc_col:
        st.markdown(
            f'<div style="background:#FFFAF0;border:1px solid {WARN}44;border-radius:6px;'
            f'padding:.55rem .8rem;font-size:.74rem;color:{WARN};line-height:1.6;">'
            f'⚠  Research use only. Not a clinical diagnosis.<br>'
            f'All results require qualified neurologist review.</div>',
            unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    init_state()
    dispatch = {
        "welcome":     page_welcome,
        "calibration": page_calibration,
        "mem_intro":   page_mem_intro,
        "mem_task":    page_mem_task,
        "sacc_intro":  page_sacc_intro,
        "sacc_task":   page_sacc_task,
        "report":      page_report,
    }
    fn = dispatch.get(st.session_state.get("page", "welcome"))
    if fn:
        fn()
    else:
        nav("welcome"); st.rerun()


if __name__ == "__main__":
    main()
