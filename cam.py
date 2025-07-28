import serial
import cv2
import numpy as np

# 串口名称和波特率
port = 'COM4'  # 请根据实际情况修改
baud = 115200

# 初始化串口
ser = serial.Serial(port, baud, timeout=2)

def read_jpeg_frame():
    jpg_data = bytearray()
    start = False
    while True:
        byte = ser.read(1)
        if not byte:
            continue
        jpg_data += byte
        if jpg_data[-2:] == b'\xff\xd8':  # 开始
            jpg_data = b'\xff\xd8'
            start = True
        elif start and jpg_data[-2:] == b'\xff\xd9':  # 结束
            return jpg_data

while True:
    try:
        jpg_frame = read_jpeg_frame()
        np_arr = np.frombuffer(jpg_frame, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("ESP32-CAM", img)
        if cv2.waitKey(1) == 27:  # ESC退出
            break
    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()
ser.close()
