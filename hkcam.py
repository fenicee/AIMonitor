import os
import time

import cv2
import gc
from multiprocessing import Process, Manager
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """·
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)

    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    count = 0
    dir = 0
    pTime = 0
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while True:
            lmList = []
            if len(stack) != 0:
                value = stack.pop()
                image = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style()
                )
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = image.shape
                        # print(id,lm)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # cv2.circle(image, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                        lmList.append([id, cx, cy])
                if len(lmList) != 0:
                    # 右手
                    # angle, image = findAngle(image, 12, 14, 16)
                    x1, y1 = lmList[12][1:]
                    x2, y2 = lmList[14][1:]
                    x3, y3 = lmList[16][1:]
                    # 计算角度
                    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    if angle < 0:
                        angle += 360
                        # 画图
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.line(image, (x2, y2), (x3, y3), (255, 255, 255), 3)
                    cv2.circle(image, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, (x1, y1), 15, (255, 0, 0), 2)
                    cv2.circle(image, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 15, (255, 0, 0), 2)
                    cv2.circle(image, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, (x3, y3), 15, (255, 0, 0), 3)
                    cv2.putText(image, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 50),
                                3)
                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     results.face_landmarks,
                    #     mp_holistic.FACEMESH_TESSELATION,
                    #     connection_drawing_spec=mp_drawing_styles
                    #         .get_default_face_mesh_tesselation_style())
                    per = np.interp(angle, (270, 300), (0, 100))
                    # print(per)
                    # #左手
                    # detector.findAngle(img,11,13,15)
                    # print(lmList[16][2]>lmList[12][2])
                    if lmList[16][2] < lmList[12][2]:
                        if lmList[16][2] < lmList[14][2]:
                            if per == 100:
                                if dir == 0:
                                    count += 0.5
                                    dir = 1
                            if per == 0:
                                if dir == 1:
                                    count += 0.5
                                    dir = 0
                    else:
                        count = 0
                    if count >= 3:
                        cv2.putText(image, "confirmed", (60, 150), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                    # cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
                    cv2.putText(image, str(int(count)), (60, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.imshow("hkcam", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

if __name__ == "__main__":
    q = Manager().list()
    pw = Process(target=write, args=(q, "rtsp://admin:wasdjkl123@192.168.2.63:554/Streaming/Channels/101?transportmode=unicast", 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pr结束:
    pr.join()
    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
