#!/usr/bin/env python3
# hand_gesture_recognizer.py

import cv2
import numpy as np
import json
import os

# ===== 結果結構 =====
class HandGestureResult:
    def __init__(self, fingertip_count: int, angle: float, mask: np.ndarray, annotated: np.ndarray):
        self.fingertip_count = fingertip_count
        self.angle = angle
        self.mask = mask
        self.annotated = annotated

# ===== 主識別類 =====
class HandGestureRecognizer:
    # 靜態 HSV 參數（預設值）
    h_min, h_max = 116, 131
    s_min, s_max = 26, 255
    v_min, v_max = 40, 255
    _cfg_file = "hsv_config.json"

    @classmethod
    def save_config(cls):
        cfg = {
            "h_min": cls.h_min, "h_max": cls.h_max,
            "s_min": cls.s_min, "s_max": cls.s_max,
            "v_min": cls.v_min, "v_max": cls.v_max,
        }
        with open(cls._cfg_file, "w", encoding="utf8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[INFO] Saved HSV config to {cls._cfg_file}")

    @classmethod
    def load_config(cls):
        if not os.path.exists(cls._cfg_file):
            return
        with open(cls._cfg_file, "r", encoding="utf8") as f:
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
        """即時調整 HSV 參數，調整完自動存檔"""
        cv2.namedWindow("HSV Tuning", cv2.WINDOW_AUTOSIZE)
        # 建立 Trackbar
        cv2.createTrackbar("H min", "HSV Tuning", cls.h_min, 179, lambda v: setattr(cls, 'h_min', v))
        cv2.createTrackbar("H max", "HSV Tuning", cls.h_max, 179, lambda v: setattr(cls, 'h_max', v))
        cv2.createTrackbar("S min", "HSV Tuning", cls.s_min, 255, lambda v: setattr(cls, 's_min', v))
        cv2.createTrackbar("S max", "HSV Tuning", cls.s_max, 255, lambda v: setattr(cls, 's_max', v))
        cv2.createTrackbar("V min", "HSV Tuning", cls.v_min, 255, lambda v: setattr(cls, 'v_min', v))
        cv2.createTrackbar("V max", "HSV Tuning", cls.v_max, 255, lambda v: setattr(cls, 'v_max', v))

        print("[INFO] Entering HSV tuning mode. Press 'q' or ESC to finish.")
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
        """偵測手勢：回傳 (fingertip_count, angle, mask, annotated)"""
        # 1. 前處理
        annotated = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (HandGestureRecognizer.h_min, HandGestureRecognizer.s_min, HandGestureRecognizer.v_min),
            (HandGestureRecognizer.h_max, HandGestureRecognizer.s_max, HandGestureRecognizer.v_max)
        )
        # 2. 形態學去噪 + 補洞
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)

        # 3. 找輪廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandGestureResult(0, 0.0, morph, annotated)

        # 4. 最大輪廓
        max_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) < 1:
            return HandGestureResult(0, 0.0, morph, annotated)
        cv2.drawContours(annotated, [max_cnt], -1, (0,0,255), 2)

        # 5. 凸包 + 缺陷
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
                start, end, far, depth = d
                depth /= 256.0
                if depth > 60.0:
                    pt = tuple(max_cnt[start][0])
                    fingertips.append(pt)
                    cv2.circle(annotated, pt, 5, (0,255,255), -1)
                    # 畫分支線
                    cv2.line(annotated, tuple(max_cnt[start][0]), tuple(max_cnt[far][0]), (255,255,0), 2)

        # 6. 單指時計算橢圓角度
        if len(fingertips) == 1 and len(max_cnt) >= 5:
            ellipse = cv2.fitEllipse(max_cnt)
            angle = ellipse[2]
            cv2.ellipse(annotated, ellipse, (255,0,0), 2)

        return HandGestureResult(len(fingertips), angle, morph, annotated)

# ===== 主程式 =====
def main():
    HandGestureRecognizer.load_config()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # 若要做 HSV 調整，就解除下面註解：
    # HandGestureRecognizer.hsv_tuning(cap)

    print("[INFO] Starting hand gesture recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = HandGestureRecognizer.recognize(frame)
        print(f"Fingertips: {result.fingertip_count}", end='')
        if result.fingertip_count >= 1:
            print(f", Angle: {result.angle:.1f}", end='')
        print()

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", result.mask)
        cv2.imshow("Annotated", result.annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
