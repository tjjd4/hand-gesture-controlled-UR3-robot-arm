import cv2
import numpy as np
import json
import os

class HandGestureResult:
    def __init__(self, fingertip_count: int, angle: float, mask: np.ndarray, annotated: np.ndarray):
        self.fingertip_count = fingertip_count
        self.angle = angle
        self.mask = mask
        self.annotated = annotated

class HandGestureRecognizer:
    # 初始 HSV 參數
    h_min, h_max = 116, 131
    s_min, s_max = 76, 255
    v_min, v_max = 40, 255
    _cfg_file = "hsv_config.json"

    @classmethod
    def save_config(cls):
        cfg = {
            "h_min": cls.h_min, "h_max": cls.h_max,
            "s_min": cls.s_min, "s_max": cls.s_max,
            "v_min": cls.v_min, "v_max": cls.v_max,
        }
        with open(cls._cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[INFO] Saved HSV config to {cls._cfg_file}")

    @classmethod
    def load_config(cls):
        if not os.path.exists(cls._cfg_file):
            return
        with open(cls._cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls.h_min = cfg.get("h_min", cls.h_min)
        cls.h_max = cfg.get("h_max", cls.h_max)
        cls.s_min = cfg.get("s_min", cls.s_min)
        cls.s_max = cfg.get("s_max", cls.s_max)
        cls.v_min = cfg.get("v_min", cls.v_min)
        cls.v_max = cfg.get("v_max", cls.v_max)
        print(f"[INFO] Loaded HSV config from {cls._cfg_file}")

    @classmethod
    def hsv_tuning(cls, cap: cv2.VideoCapture):
        cv2.namedWindow("HSV Tuning", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("H min", "HSV Tuning", cls.h_min, 179, lambda v: setattr(cls, 'h_min', v))
        cv2.createTrackbar("H max", "HSV Tuning", cls.h_max, 179, lambda v: setattr(cls, 'h_max', v))
        cv2.createTrackbar("S min", "HSV Tuning", cls.s_min, 255, lambda v: setattr(cls, 's_min', v))
        cv2.createTrackbar("S max", "HSV Tuning", cls.s_max, 255, lambda v: setattr(cls, 's_max', v))
        cv2.createTrackbar("V min", "HSV Tuning", cls.v_min, 255, lambda v: setattr(cls, 'v_min', v))
        cv2.createTrackbar("V max", "HSV Tuning", cls.v_max, 255, lambda v: setattr(cls, 'v_max', v))
        print("[INFO] In HSV tuning mode: press 'q' or ESC to exit and save.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                (cls.h_min, cls.s_min, cls.v_min),
                (cls.h_max, cls.s_max, cls.v_max)
            )
            cv2.imshow("HSV Tuning", mask)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
        cv2.destroyWindow("HSV Tuning")
        cls.save_config()

    @staticmethod
    def recognize(frame: np.ndarray) -> HandGestureResult:
        annotated = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (HandGestureRecognizer.h_min, HandGestureRecognizer.s_min, HandGestureRecognizer.v_min),
            (HandGestureRecognizer.h_max, HandGestureRecognizer.s_max, HandGestureRecognizer.v_max)
        )
        # 形態學操作
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, close_k)
        # 找輪廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandGestureResult(0, 0.0, morph, annotated)
        max_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) < 1:
            return HandGestureResult(0, 0.0, morph, annotated)
        cv2.drawContours(annotated, [max_cnt], -1, (0,0,255), 2)
        hull_idx = cv2.convexHull(max_cnt, returnPoints=False)
        if len(hull_idx) <= 3:
            return HandGestureResult(0, 0.0, morph, annotated)
        hull_pts = cv2.convexHull(max_cnt)
        cv2.polylines(annotated, [hull_pts], True, (0,255,0), 2)
        defects = cv2.convexityDefects(max_cnt, hull_idx)
        fingertips = []
        angle = 0.0
        if defects is not None:
            for d in defects.reshape(-1,4):
                s,e,f,depth = d
                depth /= 256.0
                if depth > 60:
                    pt = tuple(max_cnt[s][0])
                    fingertips.append(pt)
                    cv2.circle(annotated, pt, 5, (0,255,255), -1)
                    cv2.line(annotated, tuple(max_cnt[s][0]), tuple(max_cnt[f][0]), (255,255,0), 2)
        if len(fingertips) == 1 and len(max_cnt) >= 5:
            ellipse = cv2.fitEllipse(max_cnt)
            angle = ellipse[2]
            cv2.ellipse(annotated, ellipse, (255,0,0), 2)
        return HandGestureResult(len(fingertips), angle, morph, annotated)

# 在模組載入時自動讀取配置
HandGestureRecognizer.load_config()
