#!/usr/bin/env python3
"""
UR3 控制 + 手勢辨識整合腳本（獨立檔案）
依賴：
  pip install opencv-python numpy rtde_control rtde_receive rtde_io scipy
"""

import cv2
import numpy as np
import argparse
import time
import threading
import queue
from scipy.spatial.transform import Rotation as Rot
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO

from Hand_Gesture_Recognizer import HandGestureRecognizer, HandGestureResult

# TCP 長度
TCPLENGTH = 0.21

# 回原點的關節值範例
HOME_ORANGE = [0.5,
                -0.3,
                0.5,
                -1.248,
                1.187,
                -1.205]

HOME_APPLE = [0.5,
              0,
              0.5,
            -1.248,
            1.187,
            -1.205]

HOME_BANANA = [0.5,
                -0,
                0.45,
                -1.248,
                1.187,
                -1.205]

HOME_PEAR = [0.5,
                -0.3,
                0.45,
                -1.248,
                1.187,
                -1.205]

def main():
    # 解析命令列
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ur3_ip',
                        type=str,
                        default='192.168.10.204',
                        help='UR3 IP')
    args = parser.parse_args()

    # 初始化 UR3 RTDE
    rtde_r = RTDEReceive(args.ur3_ip)
    rtde_c = RTDEControl(args.ur3_ip)
    rtde_i = RTDEIO(args.ur3_ip)  # 如有需要 I/O
    rtde_c.setTcp([0, 0, TCPLENGTH, 0, 0, 0])
    print("[INFO] Connected to UR3 at", args.ur3_ip)

    # 先回到橘色位置
    print("[INFO] Moving to HOME_ORANGE pose...")
    rtde_c.moveL(HOME_ORANGE)
    time.sleep(1)  # 等待平穩到位

    # 載入手勢辨識設定
    HandGestureRecognizer.load_config()

    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    ret, frame = cap.read()

    gesture_queue = queue.Queue(1)

    # 啟動背景 thread 處理手勢控制
    moving_thread = threading.Thread(
        target=from_frame_to_move,
        args=(gesture_queue, rtde_r, rtde_c, rtde_i),
        daemon=True
    )
    moving_thread.start()

    print("[INFO] Starting gesture recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result: HandGestureResult = HandGestureRecognizer.recognize(frame)
        # 顯示標註影像
        cv2.imshow("Annotated", result.annotated)

        print(f"Fingertips: {result.fingertip_count}, Angle: {result.angle:.1f}")

        while not gesture_queue.empty():
            try:
                gesture_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            gesture_queue.put_nowait(result)
        except queue.Full:
            pass

        # 按 q 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    moving_thread.join()
    rtde_c.disconnect()
    print("[INFO] Done.")


def from_frame_to_move(queue, rtde_r, rtde_c, rtde_i):
    while True:
        result: HandGestureResult = queue.get()
        if result.fingertip_count == 0:
            pass
        elif result.fingertip_count == 1:
            print("[ACTION] Move to ORANGE")
            rtde_c.moveL(HOME_ORANGE)
        elif result.fingertip_count == 2:
            print("[ACTION] Move to APPLE")
            rtde_c.moveL(HOME_APPLE)
        elif result.fingertip_count == 3:
            print("[ACTION] Move to BANANA")
            rtde_c.moveL(HOME_BANANA)
        elif result.fingertip_count == 4:
            print("[ACTION] Move to PEAR")
            rtde_c.moveL(HOME_PEAR)

if __name__ == "__main__":
    main()
