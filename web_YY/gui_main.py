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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox, QGroupBox,
    QMessageBox, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from face import face_run

# 导入main.py中的配置和函数
from main import PERSON, PERSON_id
client = OpenAI(api_key="sk-db429a52f01f4d5a9feff537972d738d", base_url="https://api.deepseek.com")

# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'database': 'factory_management',
    'user': 'root',
    'password': 'Caozx616',
    'charset': 'utf8mb4'
}

# 状态常量
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2
SILENCE_THRESHOLD = 200
MAX_SILENCE_DURATION = 8

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
    """将数据插入到MySQL数据库"""
    connection = None
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
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def is_silence(data, threshold=SILENCE_THRESHOLD):
    """VAD静音检测"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() < threshold

class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.running = True
        self.start()
        return True
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
    
    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(0.03)  # 约30fps

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
        
        self.wsParam = Ws_Param(
            APPID='12d84f1e',
            APIKey='598d23f9a26a02b52a0a0ec9760298fd',
            APISecret='OGFmYTQ5YWRhNmZkNDM2ZWI4NWJlOGI3'
        )
    
    def start_recording(self):
        self.recording = True
        self.all_recognized_text = []
        self.start()
    
    def stop_recording(self):
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
            p = pyaudio.PyAudio()
            name = face_run()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
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
            
            try:
                while self.recording:
                    data = stream.read(1280)
                    
                    if is_silence(data):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > MAX_SILENCE_DURATION:
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
                        break
            
            except Exception as e:
                self.error_occurred.emit(f"录音异常: {str(e)}")
            finally:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                    p.terminate()
                except:
                    pass
        
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工厂语音识别系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量 - 根据main.py中的PERSON值自动设置
        self.person_type = PERSON  # 从main.py导入的PERSON值：1=工人, 0=质检员
        self.person_id = PERSON_id if PERSON_id > 0 else 1  # 从main.py导入的PERSON_id值
        self.camera_thread = None
        self.speech_thread = None
        self.extracted_data = None
        self.recording = False
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧布局（摄像头和控制）
        left_layout = QVBoxLayout()
        
        # 摄像头显示
        camera_group = QGroupBox("摄像头监控")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("摄像头未启动")
        camera_layout.addWidget(self.camera_label)
        

        
        left_layout.addWidget(camera_group)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        
        # 员工信息显示（只读）
        info_display_layout = QHBoxLayout()
        job_type = "工人" if self.person_type == 1 else "质检员"
        prefix = "W" if self.person_type == 1 else "Q"
        employee_id = f"{prefix}{self.person_id:03d}"
        self.employee_info_label = QLabel(f"当前员工: {job_type} - {employee_id}")
        self.employee_info_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.employee_info_label.setStyleSheet("padding: 10px; background-color: lightblue; border: 1px solid gray; border-radius: 5px;")
        info_display_layout.addWidget(self.employee_info_label)
        
        control_layout.addLayout(info_display_layout)
        
        # 按钮
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        control_layout.addWidget(self.start_camera_btn)
        
        self.record_btn = QPushButton("开始录音")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        control_layout.addWidget(self.record_btn)
        
        self.retry_btn = QPushButton("重新录音")
        self.retry_btn.clicked.connect(self.retry_recording)
        self.retry_btn.setEnabled(False)
        control_layout.addWidget(self.retry_btn)
        
        self.upload_btn = QPushButton("确认上传")
        self.upload_btn.clicked.connect(self.upload_to_database)
        self.upload_btn.setEnabled(False)
        control_layout.addWidget(self.upload_btn)
        
        left_layout.addWidget(control_group)
        
        # 右侧布局（文本显示和结果）
        right_layout = QVBoxLayout()
        
        # 实时识别文本
        text_group = QGroupBox("语音识别结果")
        text_layout = QVBoxLayout(text_group)
        
        self.realtime_text = QTextEdit()
        self.realtime_text.setPlaceholderText("实时识别的语音内容将显示在这里...")
        self.realtime_text.setMaximumHeight(150)
        text_layout.addWidget(QLabel("实时识别:"))
        text_layout.addWidget(self.realtime_text)
        
        self.full_text = QTextEdit()
        self.full_text.setPlaceholderText("完整的语音识别结果将显示在这里...")
        self.full_text.setMaximumHeight(150)
        text_layout.addWidget(QLabel("完整文本:"))
        text_layout.addWidget(self.full_text)
        
        right_layout.addWidget(text_group)
        
        # AI提取结果
        result_group = QGroupBox("AI信息提取结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("AI提取的结构化信息将显示在这里...")
        result_layout.addWidget(self.result_text)
        
        right_layout.addWidget(result_group)
        
        # 状态显示
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("padding: 10px; background-color: lightgray; border: 1px solid gray;")
        right_layout.addWidget(self.status_label)
        
        # 添加到主布局
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)
    


    
    def start_camera(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.update_camera_frame)
            
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
        
        # 缩放到标签大小
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
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
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个信息提取专家，根据以下的内容，进行信息提取。如果没有，不要试图编造答案。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            
            result = response.choices[0].message.content
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
            
        except Exception as e:
            QMessageBox.warning(self, "AI处理错误", f"信息提取失败: {str(e)}")
            self.status_label.setText("状态: AI信息提取失败")
    
    def upload_to_database(self):
        if not self.extracted_data:
            QMessageBox.warning(self, "错误", "没有可上传的数据")
            return
        
        try:
            job_type = "工人" if self.person_type == 1 else "质检员"
            prefix = "W" if self.person_type == 1 else "Q"
            employee_id = f"{prefix}{self.person_id:03d}"
            
            workpiece_name = self.extracted_data.get('Workpiece_name', '未提及')
            workpiece_dn = self.extracted_data.get('Workpiece_DN', '未提及')
            workpiece_size = self.extracted_data.get('Workpiece_size', '未提及')
            
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
            else:
                QMessageBox.warning(self, "错误", "数据上传失败，请检查数据库连接")
                
        except Exception as e:
            QMessageBox.critical(self, "数据库错误", f"上传数据时出错: {str(e)}")
    
    def closeEvent(self, event):
        # 关闭摄像头线程
        if self.camera_thread:
            self.camera_thread.stop_camera()
            self.camera_thread.wait()
        
        # 停止语音识别
        if self.speech_thread and self.speech_thread.isRunning():
            self.speech_thread.stop_recording()
            self.speech_thread.wait()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()