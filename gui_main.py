import sys
import threading
import time
import json
import base64
import ssl
import hashlib
import hmac
import re
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import numpy as np
import cv2
import pyaudio
import websocket
import mysql.connector
from mysql.connector import Error
from openai import OpenAI
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox, QGroupBox,
    QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize

# 导入main.py中的配置和函数以及test模块
from main import PERSON, PERSON_id
import test

client = OpenAI(api_key="sk-db429a52f01f4d5a9feff537972d738d", base_url="https://api.deepseek.com")

# MySQL数据库配置 - 使用远程服务器
DB_CONFIG = {
    'host': 'aws.tianle666.xyz',
    'database': 'factory_management',
    'user': 'root',
    'password': 'Ztl20040720.',
    'charset': 'utf8mb4'
}

# 状态常量
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2
SILENCE_THRESHOLD = 200
MAX_SILENCE_DURATION = 8

# 全局变量初始化
if 'PERSON' not in globals():
    PERSON = 0
if 'PERSON_id' not in globals():
    PERSON_id = 0


class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.iat_params = {
            "domain": "slm",
            "language": "zh_cn",
            "accent": "mulacc",
            "result": {"encoding": "utf8", "compress": "raw", "format": "json"}
        }

    def create_url(self):
        url = 'wss://iat.cn-huabei-1.xf-yun.com/v1'
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))
        signature_origin = f"host: iat.cn-huabei-1.xf-yun.com\ndate: {date}\nGET /v1 HTTP/1.1"
        signature_sha = hmac.new(
            self.APISecret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode('utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        v = {"authorization": authorization, "date": date, "host": "iat.cn-huabei-1.xf-yun.com"}
        return url + '?' + urlencode(v)


def insert_to_database(employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size,
                       start_time=None, process_type=None, self_check_result=None,
                       inspection_passed=None, inspection_details=None):
    """将数据插入到MySQL数据库，增强错误处理"""
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)

        if connection.is_connected():
            cursor = connection.cursor()

            if job_type == '工人':
                sql = """
                INSERT INTO employee_workpiece 
                (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size, 
                 start_time, process_type, self_check_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size,
                          start_time, process_type, self_check_result)
            else:
                sql = """
                INSERT INTO employee_workpiece 
                (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size, 
                 inspection_passed, inspection_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                passed_bool = None
                if inspection_passed == "是":
                    passed_bool = True
                elif inspection_passed == "否":
                    passed_bool = False

                values = (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size,
                          passed_bool, inspection_details)

            cursor.execute(sql, values)
            connection.commit()
            return True

    except Error as e:
        print(f"数据库操作错误: {e}")
        if connection:
            connection.rollback()
        return False
    except Exception as e:
        print(f"数据库操作异常: {e}")
        if connection:
            connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def is_silence(data, threshold=SILENCE_THRESHOLD):
    """VAD静音检测，增强错误处理"""
    try:
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() < threshold
    except Exception as e:
        print(f"静音检测错误: {e}")
        return False


class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.camera_index = 0

    def start_camera(self):
        try:
            # 尝试不同的摄像头索引
            for i in range(3):  # 尝试索引0, 1, 2
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.camera_index = i
                    print(f"摄像头 {i} 连接成功")
                    break
                if self.cap:
                    self.cap.release()
                self.cap = None

            if self.cap is None:
                error_msg = "无法打开任何摄像头设备"
                print(error_msg)
                self.camera_error.emit(error_msg)
                return False

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.running = True
            self.start()
            return True
        except Exception as e:
            error_msg = f"摄像头启动错误: {e}"
            print(error_msg)
            self.camera_error.emit(error_msg)
            return False

    def stop_camera(self):
        self.running = False
        if self.isRunning():
            self.wait(5000)  # 等待最多5秒
            if self.isRunning():
                self.terminate()  # 强制终止
                self.wait(1000)

        try:
            if self.cap:
                self.cap.release()
                self.cap = None
                print("摄像头资源已释放")
        except Exception as e:
            print(f"释放摄像头资源时出错: {e}")

    def run(self):
        consecutive_failures = 0
        max_failures = 10

        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame_ready.emit(frame)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        error_msg = "摄像头读取失败次数过多，停止摄像头"
                        print(error_msg)
                        self.camera_error.emit(error_msg)
                        self.running = False
                        break

                time.sleep(0.03)  # 约30fps
            except Exception as e:
                consecutive_failures += 1
                print(f"摄像头读取错误: {e}")
                if consecutive_failures >= max_failures:
                    error_msg = f"摄像头错误次数过多: {e}"
                    self.camera_error.emit(error_msg)
                    self.running = False
                    break
                time.sleep(0.1)


class SpeechRecognitionThread(QThread):
    """语音识别线程"""
    text_recognized = pyqtSignal(str)
    recognition_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, person_type):
        super().__init__()
        self.person_type = person_type
        self.ws = None
        self.all_recognized_text = []
        self.recording = False
        self.recording_timeout = 30  # 录音超时时间（秒）
        self.start_time = None
        self.audio_stream = None
        self.audio_p = None

        self.wsParam = Ws_Param(
            APPID='12d84f1e',
            APIKey='598d23f9a26a02b52a0a0ec9760298fd',
            APISecret='OGFmYTQ5YWRhNmZkNDM2ZWI4NWJlOGI3'
        )

    def start_recording(self):
        """开始录音，增强错误处理"""
        try:
            self.recording = True
            self.all_recognized_text = []
            self.start_time = time.time()

            # 检查音频设备可用性
            test_p = pyaudio.PyAudio()
            device_count = test_p.get_device_count()
            input_device = None

            for i in range(device_count):
                device_info = test_p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_device = i
                    break

            test_p.terminate()

            if input_device is None:
                raise Exception("未找到可用的音频输入设备")

            print(f"找到音频输入设备: {input_device}")
            self.start()

        except Exception as e:
            error_msg = f"录音启动失败: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            self.cleanup_audio_resources()

    def stop_recording(self):
        """停止录音，增强资源管理"""
        self.recording = False

        if self.ws:
            try:
                # 发送结束帧
                trailer_frame = {
                    "header": {"status": STATUS_LAST_FRAME, "app_id": self.wsParam.APPID},
                    "payload": {"audio": {"status": 2}}
                }
                self.ws.send(json.dumps(trailer_frame))
            except:
                pass

        if self.isRunning():
            self.wait(3000)  # 等待最多3秒
            if self.isRunning():
                self.terminate()  # 强制终止
                self.wait(1000)

        self.cleanup_audio_resources()

    def cleanup_audio_resources(self):
        """清理音频资源"""
        try:
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            if self.audio_p:
                self.audio_p.terminate()
                self.audio_p = None

            print("音频资源已清理")
        except Exception as e:
            print(f"清理音频资源时出错: {e}")

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            code = msg["header"]["code"]
            if code != 0:
                self.error_occurred.emit(f"错误码: {code}")
                ws.close()
            else:
                payload = msg.get("payload", {})
                if "result" in payload:
                    result_base64 = payload["result"]["text"]
                    result_str = base64.b64decode(result_base64).decode("utf-8")
                    result_json = json.loads(result_str)
                    text = "".join(cw["w"] for ws_item in result_json.get("ws", []) for cw in ws_item.get("cw", []))
                    if text.strip():
                        self.text_recognized.emit(text)
                        self.all_recognized_text.append(text)

            if msg["header"]["status"] == 2:
                # 识别结束
                full_text = ''.join(self.all_recognized_text)
                self.recognition_finished.emit(full_text)
                ws.close()
        except Exception as e:
            self.error_occurred.emit(f"解析失败: {str(e)}")

    def on_error(self, ws, error):
        self.error_occurred.emit(f"连接错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        pass

    def on_open(self, ws):
        def audio_capture():
            consecutive_errors = 0
            max_errors = 10

            try:
                self.audio_p = pyaudio.PyAudio()

                # 查找可用的音频输入设备
                device_count = self.audio_p.get_device_count()
                input_device = None

                for i in range(device_count):
                    device_info = self.audio_p.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        input_device = i
                        break

                if input_device is None:
                    raise Exception("未找到可用的音频输入设备")

                self.audio_stream = self.audio_p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=1280
                )

                # 发送头部包
                header_frame = {
                    "header": {"status": STATUS_FIRST_FRAME, "app_id": self.wsParam.APPID},
                    "parameter": {"iat": self.wsParam.iat_params},
                    "payload": {"audio": {"sample_rate": 16000, "encoding": "raw"}}
                }
                ws.send(json.dumps(header_frame))

                silence_start = None

                while self.recording:
                    try:
                        # 检查录音超时
                        if self.start_time and (time.time() - self.start_time) > self.recording_timeout:
                            print("录音超时，自动停止")
                            break

                        if not self.audio_stream or not self.audio_stream.is_active():
                            print("音频流不可用，停止录音")
                            break

                        data = self.audio_stream.read(1280, exception_on_overflow=False)

                        if not data:
                            consecutive_errors += 1
                            if consecutive_errors >= max_errors:
                                break
                            continue

                        consecutive_errors = 0

                        if is_silence(data):
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > MAX_SILENCE_DURATION:
                                print("检测到长时间静音，停止录音")
                                break
                            continue
                        else:
                            silence_start = None

                        audio_base64 = base64.b64encode(data).decode('utf-8')
                        data_frame = {
                            "header": {"status": STATUS_CONTINUE_FRAME, "app_id": self.wsParam.APPID},
                            "payload": {"audio": {"audio": audio_base64}}
                        }

                        if ws.sock and ws.sock.connected:
                            ws.send(json.dumps(data_frame))
                        else:
                            print("WebSocket连接已断开")
                            break

                    except Exception as e:
                        consecutive_errors += 1
                        print(f"录音读取错误: {e}")

                        if consecutive_errors >= max_errors:
                            error_msg = f"录音错误次数过多: {e}"
                            print(error_msg)
                            self.error_occurred.emit(error_msg)
                            break

                        time.sleep(0.1)  # 短暂延迟后重试

            except Exception as e:
                error_msg = f"录音异常: {str(e)}"
                print(error_msg)
                self.error_occurred.emit(error_msg)
            finally:
                self.cleanup_audio_resources()

        threading.Thread(target=audio_capture, daemon=True).start()

    def run(self):
        try:
            ws_url = self.wsParam.create_url()
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.ws.on_open = self.on_open
            self.ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=25,
                ping_timeout=10,
                ping_payload="\x09"
            )
        except Exception as e:
            self.error_occurred.emit(f"WebSocket连接失败: {str(e)}")


class ArduinoControlThread(QThread):
    """Arduino串口控制线程"""
    x_axis_triggered = pyqtSignal()  # X轴限位 - 启动摄像头
    y_axis_triggered = pyqtSignal()  # Y轴限位 - 开始录音
    z_axis_triggered = pyqtSignal()  # Z轴限位 - 确认上传

    def __init__(self):
        super().__init__()
        self.ser = None
        self.running = False
        self.connection_attempts = 0
        self.max_attempts = 3
        self.init_serial_connection()

    def init_serial_connection(self):
        """初始化串口连接，自动检测可用端口，增强错误处理"""
        while self.connection_attempts < self.max_attempts:
            try:
                # 列出所有可用的串口
                ports = serial.tools.list_ports.comports()
                available_ports = [port.device for port in ports]

                print(f"可用串口: {available_ports}")

                # 尝试连接常见的串口
                common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']

                for port in common_ports:
                    if port in available_ports:
                        try:
                            if self.ser:
                                self.ser.close()
                            self.ser = serial.Serial(port, 9600, timeout=1)
                            print(f"成功连接到Arduino串口: {port}")
                            time.sleep(2)  # 等待连接建立
                            return
                        except serial.SerialException as e:
                            print(f"无法连接到 {port}: {e}")
                            continue

                # 如果常见端口都无法连接，尝试所有可用端口
                for port in available_ports:
                    try:
                        if self.ser:
                            self.ser.close()
                        self.ser = serial.Serial(port, 9600, timeout=1)
                        print(f"成功连接到Arduino串口: {port}")
                        time.sleep(2)  # 等待连接建立
                        return
                    except serial.SerialException as e:
                        print(f"无法连接到 {port}: {e}")
                        continue

                self.connection_attempts += 1
                print(f"Arduino连接失败 (尝试 {self.connection_attempts}/{self.max_attempts})")
                if self.connection_attempts < self.max_attempts:
                    time.sleep(1)

            except Exception as e:
                self.connection_attempts += 1
                print(f"Arduino串口连接失败 (尝试 {self.connection_attempts}/{self.max_attempts}): {e}")
                if self.connection_attempts < self.max_attempts:
                    time.sleep(1)

        print("警告: 无法找到可用的Arduino串口连接，已达到最大尝试次数")
        self.ser = None

    def start_monitoring(self):
        """开始监听Arduino信号"""
        if self.ser is not None:
            self.running = True
            self.start()
        else:
            print("Arduino串口未连接，无法开始监听")

    def stop_monitoring(self):
        """停止监听Arduino信号"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def run(self):
        """监听Arduino串口信号的主循环"""
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    print(f"收到Arduino消息: {line}")

                    if line == "X_AXIS_LIMIT_TRIGGERED":
                        print("检测到X轴限位开关触发 - 启动摄像头")
                        self.x_axis_triggered.emit()
                    elif line == "Y_AXIS_LIMIT_TRIGGERED":
                        print("检测到Y轴限位开关触发 - 开始录音")
                        self.y_axis_triggered.emit()
                    elif line == "Z_AXIS_LIMIT_TRIGGERED":
                        print("检测到Z轴限位开关触发 - 确认上传")
                        self.z_axis_triggered.emit()

                time.sleep(0.1)  # 避免过度占用CPU

            except serial.SerialException as e:
                print(f"Arduino串口通信错误: {e}")
                break
            except Exception as e:
                print(f"Arduino监听线程错误: {e}")
                break


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工厂语音识别系统")

        # 获取屏幕尺寸并设置窗口最大化
        screen_geometry = QApplication.desktop().screenGeometry()
        self.setGeometry(screen_geometry)
        # 或者使用以下代码设置窗口最大化
        # self.showMaximized()

        # 初始化变量 - 根据main.py中的PERSON值自动设置
        self.person_type = PERSON  # 从main.py导入的PERSON值：1=工人, 0=质检员
        self.person_id = PERSON_id if PERSON_id > 0 else 1  # 从main.py导入的PERSON_id值
        self.camera_thread = None
        self.speech_thread = None
        self.extracted_data = None
        self.recording = False

        # 初始化Arduino控制线程
        self.arduino_thread = ArduinoControlThread()
        self.arduino_thread.x_axis_triggered.connect(self.on_arduino_start_camera)
        self.arduino_thread.y_axis_triggered.connect(self.on_arduino_start_recording)
        self.arduino_thread.z_axis_triggered.connect(self.on_arduino_upload)

        self.init_ui()

        # 启动Arduino监听
        self.arduino_thread.start_monitoring()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 设置未来科技感样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2f;
            }
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #4fc3f7;
                border: 2px solid #4fc3f7;
                border-radius: 10px;
                margin-top: 1.5em;
                padding-top: 15px;
                background-color: rgba(30, 30, 47, 150);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                font-size: 20px;
            }
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, 
                                                  stop: 0 #2196F3, stop: 1 #0c2e70);
                border: none;
                color: white;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, 
                                                  stop: 0 #42a5f5, stop: 1 #1565c0);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, 
                                                  stop: 0 #1976d2, stop: 1 #0d47a1);
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #2d2d44;
                border: 2px solid #4a4a6a;
                border-radius: 8px;
                color: #ffffff;
                font-size: 16px;
                padding: 8px;
            }
            QTextEdit:disabled {
                background-color: #3d3d54;
                color: #aaaaaa;
            }
            QComboBox, QSpinBox {
                background-color: #2d2d44;
                border: 2px solid #4a4a6a;
                border-radius: 6px;
                color: #ffffff;
                padding: 6px;
                font-size: 14px;
            }
        """)

        # 设置样式主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 左侧布局（摄像头和控制）
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)

        # 摄像头显示
        camera_group = QGroupBox("摄像头监控")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(10)

        self.camera_label = QLabel()
        self.camera_label.setStyleSheet("border: 2px solid #4fc3f7; background-color: #000000; border-radius: 8px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("摄像头未启动")
        self.camera_label.setFont(QFont("Arial", 16))
        camera_layout.addWidget(self.camera_label)

        left_layout.addWidget(camera_group)

        # 创建控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(12)

        # 员工信息显示（只读）
        info_display_layout = QHBoxLayout()
        job_type = "工人" if self.person_type == 1 else "质检员"
        prefix = "W" if self.person_type == 1 else "Q"
        employee_id = f"{prefix}{self.person_id:03d}"
        self.employee_info_label = QLabel(f"当前员工: {job_type} - {employee_id}")
        self.employee_info_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.employee_info_label.setStyleSheet(
            "padding: 15px; background-color: rgba(33, 150, 243, 0.2); "
            "border: 2px solid #2196F3; border-radius: 8px; color: #4fc3f7;")
        self.employee_info_label.setAlignment(Qt.AlignCenter)
        info_display_layout.addWidget(self.employee_info_label)

        control_layout.addLayout(info_display_layout)

        # 按钮布局
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setMinimumHeight(45)
        control_layout.addWidget(self.start_camera_btn)

        self.record_btn = QPushButton("开始录音")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setMinimumHeight(45)
        control_layout.addWidget(self.record_btn)

        self.retry_btn = QPushButton("重新录音")
        self.retry_btn.clicked.connect(self.retry_recording)
        self.retry_btn.setEnabled(False)
        self.retry_btn.setMinimumHeight(45)
        control_layout.addWidget(self.retry_btn)

        self.upload_btn = QPushButton("确认上传")
        self.upload_btn.clicked.connect(self.upload_to_database)
        self.upload_btn.setEnabled(False)
        self.upload_btn.setMinimumHeight(45)
        control_layout.addWidget(self.upload_btn)

        # Arduino控制说明
        arduino_info_label = QLabel("Arduino实体按键控制:")
        arduino_info_label.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(arduino_info_label)

        arduino_mapping_label = QLabel(
            "• X轴限位开关 → 启动摄像头\n"
            "• Y轴限位开关 → 开始/停止录音\n"
            "• Z轴限位开关 → 确认上传"
        )
        arduino_mapping_label.setStyleSheet(
            "padding: 12px; background-color: rgba(64, 64, 92, 0.7); "
            "border: 2px solid #5c5c7a; border-radius: 8px; font-size: 11pt; color: #e0e0e0;")
        arduino_mapping_label.setWordWrap(True)
        control_layout.addWidget(arduino_mapping_label)

        left_layout.addWidget(control_group)

        # 右侧布局（文本显示和结果）
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # 实时识别文本
        text_group = QGroupBox("语音识别结果")
        text_layout = QVBoxLayout(text_group)
        text_layout.setSpacing(10)

        realtime_label = QLabel("实时识别:")
        realtime_label.setFont(QFont("Arial", 24, QFont.Bold))  # 确保这里设置生效
        text_layout.addWidget(realtime_label)

        self.realtime_text = QTextEdit()
        self.realtime_text.setPlaceholderText("实时识别的语音内容将显示在这里...")
        self.realtime_text.setMaximumHeight(150)
        self.realtime_text.setStyleSheet("font-size: 20px;")  # 添加行内样式表
        text_layout.addWidget(self.realtime_text)

        # 完整文本
        full_text_label = QLabel("完整文本:")
        full_text_label.setFont(QFont("Arial", 24, QFont.Bold))
        text_layout.addWidget(full_text_label)

        self.full_text = QTextEdit()
        self.full_text.setPlaceholderText("完整的语音识别结果将显示在这里...")
        self.full_text.setMaximumHeight(150)
        self.full_text.setStyleSheet("font-size: 20px;")  # 添加行内样式表
        text_layout.addWidget(self.full_text)

        right_layout.addWidget(text_group)

        # AI提取结果
        result_group = QGroupBox("AI信息提取结果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)

        result_label = QLabel("提取结果:")
        result_label.setFont(QFont("Arial", 24, QFont.Bold))
        result_layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("AI提取的结构化信息将显示在这里...")
        self.result_text.setStyleSheet("font-size: 20px;")
        result_layout.addWidget(self.result_text)

        right_layout.addWidget(result_group)

        # 状态显示
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet(
            "padding: 15px; background-color: rgba(76, 175, 80, 0.2); "
            "border: 2px solid #4CAF50; border-radius: 8px; color: #a5d6a7;")
        right_layout.addWidget(self.status_label)

        # Arduino连接状态显示
        self.arduino_status_label = QLabel("Arduino: 检测中...")
        self.arduino_status_label.setFont(QFont("Arial", 12))
        self.arduino_status_label.setStyleSheet(
            "padding: 10px; background-color: rgba(255, 193, 7, 0.2); "
            "border: 2px solid #FFC107; border-radius: 6px; color: #FFE082;")
        right_layout.addWidget(self.arduino_status_label)

        # 设置定时器检查Arduino连接状态
        self.arduino_status_timer = QTimer()
        self.arduino_status_timer.timeout.connect(self.update_arduino_status)
        self.arduino_status_timer.start(2000)  # 每2秒检查一次

        # 添加到主布局
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

    def start_camera(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_camera_frame)
            self.camera_thread.camera_error.connect(self.on_camera_error)

        if self.camera_thread.start_camera():
            self.start_camera_btn.setText("摄像头已启动")
            self.start_camera_btn.setEnabled(False)
            self.record_btn.setEnabled(True)
            self.status_label.setText("状态: 摄像头已启动")
        else:
            QMessageBox.warning(self, "错误", "无法启动摄像头")

    def update_camera_frame(self, frame):
        # 转换OpenCV格式到Qt格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 获取摄像头监控区域的尺寸，并设置为固定尺寸以防止变化
        target_size = self.camera_label.contentsRect().size()

        # 缩放图像以填充整个摄像头标签区域，保持纵横比
        scaled_image = qt_image.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 创建居中显示的QPixmap
        final_pixmap = QPixmap(target_size)
        final_pixmap.fill(Qt.black)  # 用黑色填充背景

        # 计算居中位置
        x = (target_size.width() - scaled_image.width()) // 2
        y = (target_size.height() - scaled_image.height()) // 2

        # 在黑色背景上绘制居中图像
        painter = QPainter(final_pixmap)
        painter.drawImage(x, y, scaled_image)
        painter.end()

        # 更新摄像头画面
        self.camera_label.setPixmap(final_pixmap)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # 在录音前进行人脸识别
        self.status_label.setText("状态: 正在进行人脸识别...")

        try:
            # 调用face_run()进行人脸识别
            from test import face_run
            result = face_run()

            if result is None or len(result) != 3:
                # 人脸识别失败
                QMessageBox.warning(self, "人脸识别失败", "需要录入人脸信息")
                self.status_label.setText("状态: 人脸识别失败，请录入人脸信息")
                return

            # 人脸识别成功，更新PERSON和PERSON_id
            name, person_type, person_id = result
            if name and person_type is not None and person_id is not None:
                global PERSON, PERSON_id
                PERSON = int(person_type)
                PERSON_id = int(person_id)
            else:
                QMessageBox.warning(self, "人脸识别失败", "识别结果无效")
                self.status_label.setText("状态: 人脸识别结果无效")
                return

            # 更新界面显示的员工信息
            self.person_type = PERSON
            self.person_id = PERSON_id
            job_type = "工人" if self.person_type == 1 else "质检员"
            prefix = "W" if self.person_type == 1 else "Q"
            employee_id = f"{prefix}{self.person_id:03d}"
            self.employee_info_label.setText(f"当前员工: {job_type} - {employee_id} ({name})")

            # 开始录音
            self.recording = True
            self.record_btn.setText("停止录音")
            self.retry_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
            self.status_label.setText("状态: 正在录音...")

            # 清空之前的内容
            self.realtime_text.clear()
            self.full_text.clear()
            self.result_text.clear()

            # 启动语音识别线程
            self.speech_thread = SpeechRecognitionThread(self.person_type)
            self.speech_thread.text_recognized.connect(self.on_text_recognized)
            self.speech_thread.recognition_finished.connect(self.on_recognition_finished)
            self.speech_thread.error_occurred.connect(self.on_speech_error)
            self.speech_thread.start_recording()

        except Exception as e:
            QMessageBox.warning(self, "人脸识别错误", f"人脸识别过程中出现错误: {str(e)}")
            self.status_label.setText("状态: 人脸识别出错")

    def stop_recording(self):
        if self.speech_thread:
            self.speech_thread.stop_recording()

        self.recording = False
        self.record_btn.setText("开始录音")
        self.status_label.setText("状态: 录音已停止，等待处理...")

    def retry_recording(self):
        self.extracted_data = None
        self.start_recording()

    def on_text_recognized(self, text):
        # 显示实时识别的文本
        current_text = self.realtime_text.toPlainText()
        self.realtime_text.setPlainText(current_text + text + " ")

        # 自动滚动到底部
        cursor = self.realtime_text.textCursor()
        cursor.movePosition(cursor.End)
        self.realtime_text.setTextCursor(cursor)

    def on_recognition_finished(self, full_text):
        self.full_text.setPlainText(full_text)
        self.status_label.setText("状态: 语音识别完成，正在进行AI信息提取...")

        # 启动AI信息提取
        self.extract_information(full_text)

    def on_speech_error(self, error_msg):
        QMessageBox.warning(self, "语音识别错误", error_msg)
        self.recording = False
        self.record_btn.setText("开始录音")
        self.status_label.setText("状态: 语音识别出错")

    def extract_information(self, text_content):
        """使用AI提取信息，增强错误处理"""
        if not text_content or text_content.strip() == "":
            QMessageBox.warning(self, "错误", "没有可提取的文本内容")
            self.status_label.setText("状态: 没有可提取的文本内容")
            return

        try:
            if self.person_type == 1:  # 工人
                prompt = f"""你是一位熟悉工业制造流程和工厂现场语言风格的信息提取专家。
                        内容: {text_content}。
                        以上内容来源于工人现场的语音输入，可能包含口语表达、不规范说法、同音字错误（如"图好"可能是"图号"，"尺码"是"尺寸"）等情况。
                        请你尽可能理解语义并从中提取以下关键信息（若信息缺失，请填写"未提及"）：
                        1.Workpiece_name（工件名称）
                        2.Workpiece_DN（工件图号）
                        3.Workpiece_size（工件尺寸）
                        4.Start_time（开工时间）
                        5.Process_type（加工类型，如车削、铣削、钻孔等）
                        6.Self_check_result（工人自检结果，是否有缺陷、是否合格等）
                        请严格按照以下格式返回（不要添加任何解释或额外内容）：
                        Workpiece_name: [工件名称]
                        Workpiece_DN: [工件图号]
                        Workpiece_size: [工件尺寸]
                        Start_time: [提取的开工时间]
                        Process_type: [提取的加工类型]
                        Self_check_result: [提取的工人自检结果]
                    """
            else:  # 质检员
                prompt = f"""你是一位熟悉工业制造流程和工厂现场语言风格的信息提取专家。
                        内容: {text_content}。
                        以上内容来源于质检员现场的语音输入，可能包含口语表达、不规范说法、同音字错误（如"图好"可能是"图号"，"尺码"是"尺寸"）等情况。
                        请你尽可能理解语义并从中提取以下关键信息（若信息缺失，请填写"未提及"）：
                        1.Workpiece_name（工件名称）
                        2.Workpiece_DN（工件图号）
                        3.Workpiece_size（工件尺寸）
                        4.Inspection_passed（是否检验合格，仅提取"是"或"否"）
                        5.Inspection_details（检验时的具体情况，包括是否有问题、缺陷描述等）
                        请严格按照以下格式返回（不要添加任何其他内容）：
                        Workpiece_name: [提取的工件名称]
                        Workpiece_DN: [提取的工件图号]
                        Workpiece_size: [提取的工件尺寸]
                        Inspection_passed: [是/否/未提及]
                        Inspection_details: [提取的检验状况]
                    """

            # 设置超时和重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system",
                             "content": "你是一个信息提取专家，根据以下的内容，进行信息提取。如果没有，不要试图编造答案。"},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False
                    )

                    result = response.choices[0].message.content
                    if result and result.strip():
                        self.result_text.setPlainText(result)

                        # 解析结果
                        lines = result.strip().split('\n')
                        extracted_data = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                extracted_data[key.strip()] = value.strip()

                        self.extracted_data = extracted_data
                        self.retry_btn.setEnabled(True)
                        self.upload_btn.setEnabled(True)
                        self.status_label.setText("状态: AI信息提取完成，可以上传数据")
                        print(f"AI提取结果: {result}")
                        return
                    else:
                        raise Exception("AI返回空结果")

                except Exception as e:
                    print(f"AI提取尝试 {attempt + 1}/{max_retries} 失败: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1)  # 重试前等待1秒

        except Exception as e:
            error_msg = f"信息提取失败: {str(e)}"
            print(error_msg)
            QMessageBox.warning(self, "AI处理错误", error_msg)
            self.status_label.setText("状态: AI信息提取失败")

    def upload_to_database(self):
        """上传到数据库，增强错误处理"""
        if not hasattr(self, 'extracted_data') or not self.extracted_data:
            QMessageBox.warning(self, "错误", "没有可上传的数据")
            return

        # 验证必要的用户信息
        if not hasattr(self, 'person_type') or not hasattr(self, 'person_id'):
            QMessageBox.warning(self, "错误", "用户身份信息不完整")
            return

        try:
            job_type = "工人" if self.person_type == 1 else "质检员"
            prefix = "W" if self.person_type == 1 else "Q"
            employee_id = f"{prefix}{self.person_id:03d}"

            workpiece_name = self.extracted_data.get('Workpiece_name', '未提及')
            workpiece_dn = self.extracted_data.get('Workpiece_DN', '未提及')
            workpiece_size = self.extracted_data.get('Workpiece_size', '未提及')

            print(f"准备上传数据: 员工ID={employee_id}, 类型={job_type}, 工件={workpiece_name}")

            if self.person_type == 1:  # 工人
                start_time = self.extracted_data.get('Start_time', '未提及')
                process_type = self.extracted_data.get('Process_type', '未提及')
                self_check_result = self.extracted_data.get('Self_check_result', '未提及')

                success = insert_to_database(
                    employee_id=employee_id,
                    job_type=job_type,
                    workpiece_name=workpiece_name,
                    workpiece_dn=workpiece_dn,
                    workpiece_size=workpiece_size,
                    start_time=start_time if start_time != '未提及' else None,
                    process_type=process_type if process_type != '未提及' else None,
                    self_check_result=self_check_result if self_check_result != '未提及' else None
                )
            else:  # 质检员
                inspection_passed = self.extracted_data.get('Inspection_passed', '未提及')
                inspection_details = self.extracted_data.get('Inspection_details', '未提及')

                success = insert_to_database(
                    employee_id=employee_id,
                    job_type=job_type,
                    workpiece_name=workpiece_name,
                    workpiece_dn=workpiece_dn,
                    workpiece_size=workpiece_size,
                    inspection_passed=inspection_passed if inspection_passed != '未提及' else None,
                    inspection_details=inspection_details if inspection_details != '未提及' else None
                )

            if success:
                QMessageBox.information(self, "成功", "数据已成功上传到数据库！")
                self.status_label.setText("状态: 数据上传完成")
                self.upload_btn.setEnabled(False)
                # 清空数据，准备下一次录音
                self.extracted_data = None
                self.result_text.clear()
                print("数据上传成功")
            else:
                QMessageBox.warning(self, "错误", "数据上传失败，请检查数据库连接")
                self.status_label.setText("状态: 数据上传失败")
                print("数据上传失败")

        except Exception as e:
            error_msg = f"上传数据时出错: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "数据库错误", error_msg)
            self.status_label.setText(f"状态: {error_msg}")

    # Arduino控制回调函数
    def on_arduino_start_camera(self):
        """Arduino X轴限位触发 - 启动摄像头"""
        if self.start_camera_btn.isEnabled():
            print("Arduino触发: 启动摄像头")
            self.start_camera()
        else:
            print("Arduino触发: 摄像头已启动，忽略信号")

    def on_arduino_start_recording(self):
        """Arduino Y轴限位触发 - 开始/停止录音"""
        if self.record_btn.isEnabled():
            print("Arduino触发: 开始/停止录音")
            self.toggle_recording()
        else:
            print("Arduino触发: 录音按钮不可用，忽略信号")

    def on_arduino_upload(self):
        """Arduino Z轴限位触发 - 确认上传"""
        if self.upload_btn.isEnabled():
            print("Arduino触发: 确认上传")
            self.upload_to_database()
        else:
            print("Arduino触发: 上传按钮不可用，忽略信号")

    def on_camera_error(self, error_msg):
        """处理摄像头错误"""
        QMessageBox.warning(self, "摄像头错误", error_msg)
        self.status_label.setText(f"状态: 摄像头错误 - {error_msg}")
        # 重置摄像头按钮状态
        self.start_camera_btn.setText("启动摄像头")
        self.start_camera_btn.setEnabled(True)
        self.record_btn.setEnabled(False)

    def update_arduino_status(self):
        """更新Arduino连接状态显示"""
        if self.arduino_thread and self.arduino_thread.ser and self.arduino_thread.ser.is_open:
            self.arduino_status_label.setText("Arduino: 已连接 ✓")
            self.arduino_status_label.setStyleSheet(
                "padding: 5px; background-color: lightgreen; border: 1px solid gray; border-radius: 3px;")
        else:
            self.arduino_status_label.setText("Arduino: 未连接 ✗")
            self.arduino_status_label.setStyleSheet(
                "padding: 5px; background-color: lightcoral; border: 1px solid gray; border-radius: 3px;")

    def closeEvent(self, event):
        """关闭事件处理，增强资源管理"""
        try:
            print("正在关闭应用程序...")

            # 停止录音
            if hasattr(self, 'recording') and self.recording:
                self.stop_recording()

            # 停止摄像头线程
            if hasattr(self, 'camera_thread') and self.camera_thread:
                print("停止摄像头线程...")
                self.camera_thread.stop_camera()
                if self.camera_thread.isRunning():
                    self.camera_thread.wait(3000)
                    if self.camera_thread.isRunning():
                        self.camera_thread.terminate()
                        self.camera_thread.wait(1000)

            # 停止语音识别线程
            if hasattr(self, 'speech_thread') and self.speech_thread:
                print("停止语音识别线程...")
                self.speech_thread.stop_recording()
                if self.speech_thread.isRunning():
                    self.speech_thread.wait(3000)
                    if self.speech_thread.isRunning():
                        self.speech_thread.terminate()
                        self.speech_thread.wait(1000)

            # 停止Arduino线程
            if hasattr(self, 'arduino_thread') and self.arduino_thread:
                print("停止Arduino线程...")
                self.arduino_thread.stop_monitoring()
                if self.arduino_thread.isRunning():
                    self.arduino_thread.wait(3000)
                    if self.arduino_thread.isRunning():
                        self.arduino_thread.terminate()
                        self.arduino_thread.wait(1000)

            print("应用程序关闭完成")
            event.accept()

        except Exception as e:
            print(f"关闭应用程序时出错: {e}")
            event.accept()  # 即使出错也要关闭


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

