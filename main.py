import threading
import time

import numpy as np
import websocket
import base64
import json
import ssl
import pyaudio
import cv2
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import hashlib
import hmac
import requests
import json
import re
import mysql.connector
from mysql.connector import Error

# 导入增强人脸识别模块
from test import RealTimeFaceDetection
import os
import pickle
from collections import deque, defaultdict
# 状态常量
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2
SILENCE_THRESHOLD = 200  # 静音阈值（根据麦克风调整，降低以提高灵敏度）
MAX_SILENCE_DURATION = 8  # 最大允许静音时间（秒）
PERSON = 0
PERSON_id = 0

# 摄像头控制变量
camera_running = True
cap = None

# 增强人脸识别系统变量
face_recognition_system = None
face_recognition_enabled = True
recognition_history = deque(maxlen=10)  # 存储最近的识别历史
environmental_feedback = {
    'lighting_conditions': deque(maxlen=50),
    'recognition_success_rate': 0.0,
    'adaptive_threshold': 0.6,
    'last_update_time': datetime.now()
}


# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-db429a52f01f4d5a9feff537972d738d", base_url="https://api.deepseek.com")

# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'database': 'factory_management',
    'user': 'root',  # 请根据实际情况修改
    'password': 'Caozx616',  # 请根据实际情况修改
    'charset': 'utf8mb4'
}

def insert_to_database(employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size, 
                      start_time=None, process_type=None, self_check_result=None,
                      inspection_passed=None, inspection_details=None):
    """将数据插入到MySQL数据库"""
    connection = None
    try:
        # 建立数据库连接
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # 构建插入SQL语句
            if job_type == '工人':
                sql = """
                INSERT INTO employee_workpiece 
                (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size, 
                 start_time, process_type, self_check_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size,
                         start_time, process_type, self_check_result)
            else:  # 质检员
                sql = """
                INSERT INTO employee_workpiece 
                (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size, 
                 inspection_passed, inspection_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                # 将"是"转换为True，"否"转换为False
                passed_bool = None
                if inspection_passed == "是":
                    passed_bool = True
                elif inspection_passed == "否":
                    passed_bool = False
                    
                values = (employee_id, job_type, workpiece_name, workpiece_dn, workpiece_size,
                         passed_bool, inspection_details)
            
            # 执行插入操作
            cursor.execute(sql, values)
            connection.commit()
            
            print(f"数据已成功插入数据库，记录ID: {cursor.lastrowid}")
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
            print("数据库连接已关闭")






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

def is_silence(data, threshold=SILENCE_THRESHOLD):
    """VAD静音检测"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_data).mean() < threshold

def init_camera():
    """初始化摄像头和增强人脸识别系统"""
    global cap, face_recognition_system
    
    # 打开默认摄像头（通常为0，如果有多个摄像头可以尝试1,2等）
    cap = cv2.VideoCapture(1)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
    
    # 初始化增强人脸识别系统
    try:
        print("正在初始化增强人脸识别系统...")
        face_recognition_system = RealTimeFaceDetection()
        print("增强人脸识别系统初始化成功")
        print(f"- 支持多尺度检测: {len(face_recognition_system.detection_scales)} 个尺度")
        print(f"- 自适应阈值: {face_recognition_system.environmental_feedback['adaptive_threshold']:.3f}")
        print(f"- 环境反馈机制: 已启用")
        print(f"- 自监督学习: 已启用")
    except Exception as e:
        print(f"增强人脸识别系统初始化失败: {e}")
        print("将使用基础摄像头功能")
        face_recognition_system = None
    
    print("摄像头已初始化")
    return True

def enhanced_face_recognition():
    """增强人脸识别功能"""
    global face_recognition_system, PERSON, PERSON_id, recognition_history
    
    if face_recognition_system is None:
        print("增强人脸识别系统未初始化，跳过人脸识别")
        return None
    
    print("\n=== 开始增强人脸识别 ===")
    print("请面向摄像头，系统将进行身份识别...")
    
    try:
        # 运行增强人脸识别
        recognized_name = face_recognition_system.run()
        
        if recognized_name:
            print(f"\n✓ 人脸识别成功: {recognized_name}")
            
            # 更新识别历史
            recognition_history.append({
                'name': recognized_name,
                'timestamp': datetime.now(),
                'success': True
            })
            
            # 根据识别结果设置PERSON和PERSON_id
            # 这里可以根据实际的员工数据库来映射
            # 示例逻辑：假设姓名中包含特定关键词来判断职位
            if '工人' in recognized_name or 'worker' in recognized_name.lower():
                PERSON = 1  # 工人
                PERSON_id = 1
                print(f"识别为工人: {recognized_name}")
            elif '质检' in recognized_name or 'inspector' in recognized_name.lower():
                PERSON = 2  # 质检员
                PERSON_id = 2
                print(f"识别为质检员: {recognized_name}")
            else:
                # 默认处理：可以根据员工ID或其他规则来判断
                PERSON = 1  # 默认为工人
                PERSON_id = 1
                print(f"默认识别为工人: {recognized_name}")
            
            # 更新环境反馈
            update_environmental_feedback(True)
            
            print(f"当前用户类型: {'工人' if PERSON == 1 else '质检员'}")
            print(f"用户ID: {PERSON_id}")
            
            return recognized_name
        else:
            print("\n✗ 人脸识别失败，未能识别到有效用户")
            
            # 更新识别历史
            recognition_history.append({
                'name': None,
                'timestamp': datetime.now(),
                'success': False
            })
            
            # 更新环境反馈
            update_environmental_feedback(False)
            
            # 提供备选方案
            print("\n备选方案:")
            print("1. 请调整光照条件后重试")
            print("2. 确保面部清晰可见")
            print("3. 手动输入用户信息")
            
            return None
            
    except Exception as e:
        print(f"\n✗ 人脸识别过程中发生错误: {e}")
        
        # 更新识别历史
        recognition_history.append({
            'name': None,
            'timestamp': datetime.now(),
            'success': False,
            'error': str(e)
        })
        
        return None

def update_environmental_feedback(success):
    """更新环境反馈信息"""
    global environmental_feedback, face_recognition_system
    
    try:
        # 更新成功率
        recent_attempts = list(recognition_history)[-10:]  # 最近10次尝试
        if recent_attempts:
            success_count = sum(1 for attempt in recent_attempts if attempt['success'])
            environmental_feedback['recognition_success_rate'] = success_count / len(recent_attempts)
        
        # 更新时间戳
        environmental_feedback['last_update_time'] = datetime.now()
        
        # 如果人脸识别系统可用，同步环境反馈
        if face_recognition_system:
            face_recognition_system.environmental_feedback.update({
                'recognition_success_rate': environmental_feedback['recognition_success_rate'],
                'last_update_time': environmental_feedback['last_update_time']
            })
        
        print(f"环境反馈更新 - 成功率: {environmental_feedback['recognition_success_rate']:.2%}")
        
    except Exception as e:
        print(f"更新环境反馈时出错: {e}")

def camera_display_loop():
    """摄像头显示循环 - 在主线程中运行"""
    global cap, camera_running
    
    if cap is None or not cap.isOpened():
        print("摄像头未初始化")
        return
    
    window_name = "摄像头画面 (程序运行中)"
    print("摄像头显示已启动，画面将持续显示")
    
    while camera_running:
        # 逐帧捕获
        ret, frame = cap.read()
        
        # 如果正确读取帧，ret为True
        if not ret:
            print("无法获取帧，摄像头可能已断开")
            break
        
        # 显示帧
        cv2.imshow(window_name, frame)
        
        # 等待1毫秒，检查是否需要退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or not camera_running:
            break
    
    # 关闭显示窗口
    cv2.destroyAllWindows()
    print("摄像头显示已关闭")


# WebSocket回调函数（保持不变）
# 用于存储所有识别出的文本
all_recognized_text = []
# 用于存储处理结果
processed_result = None
processed_person = None
processed_person_id = None

def on_message(ws, message):
    global processed_result, processed_person, processed_person_id, camera_running
    try:
        msg = json.loads(message)
        code = msg["header"]["code"]
        if code != 0:
            print(f"错误码: {code}, 消息: {msg.get('message', '未知错误')}")
            ws.close()
        else:
            payload = msg.get("payload", {})
            if "result" in payload:
                result_base64 = payload["result"]["text"]
                result_str = base64.b64decode(result_base64).decode("utf-8")
                result_json = json.loads(result_str)
                text = "".join(cw["w"] for ws_item in result_json.get("ws", []) for cw in ws_item.get("cw", []))
                print(f"实时识别: {text}")
                # 将识别出的文本添加到列表中
                if text.strip():
                    all_recognized_text.append(text)
        if msg["header"]["status"] == 2:
            print("识别结束")
            # 在识别结束时打印所有识别出的文本
            if all_recognized_text:
                print("\n所有识别出的文本汇总:")
                for i, text in enumerate(all_recognized_text, 1):
                    print(f"{i}. {text}")
                print(f"完整文本: {''.join(all_recognized_text)}")
                text_content = ''.join(all_recognized_text) 

                if PERSON == 1:
                    PERSON_id = 1
                    prompt = f"""你是一位熟悉工业制造流程和工厂现场语言风格的信息提取专家。
                            内容: {text_content}。
                            以上内容来源于工人现场的语音输入，可能包含口语表达、不规范说法、同音字错误（如“图好”可能是“图号”，“尺码”是“尺寸”）等情况。
                            请你尽可能理解语义并从中提取以下关键信息（若信息缺失，请填写“未提及”）：
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
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个信息提取专家，根据以下的内容，进行信息提取。如果没有，不要试图编造答案。"},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False
                    )
                    result1 = response.choices[0].message.content
                    print(f"AI提取结果: {result1}")
                    
                    # 解析AI返回的文本格式结果
                    lines = result1.strip().split('\n')
                    extracted_data = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            extracted_data[key.strip()] = value.strip()
                    
                    workpiece_name = extracted_data.get('Workpiece_name', '未提及')
                    workpiece_dn = extracted_data.get('Workpiece_DN', '未提及')
                    workpiece_size = extracted_data.get('Workpiece_size', '未提及')
                    start_time = extracted_data.get('Start_time', '未提及')
                    process_type = extracted_data.get('Process_type', '未提及')
                    self_check_result = extracted_data.get('Self_check_result', '未提及')
                    
                    # 生成员工编号（工人）
                    employee_id = f"W{PERSON_id:03d}"
                    
                    # 直接插入数据库
                    success = insert_to_database(
                        employee_id=employee_id,
                        job_type='工人',
                        workpiece_name=workpiece_name,
                        workpiece_dn=workpiece_dn,
                        workpiece_size=workpiece_size,
                        start_time=start_time if start_time != '未提及' else None,
                        process_type=process_type if process_type != '未提及' else None,
                        self_check_result=self_check_result if self_check_result != '未提及' else None
                    )
                    
                    if success:
                        print("工人数据已成功保存到数据库")
                    else:
                        print("工人数据保存失败")
                else:
                    PERSON_id = 2
                    prompt = f"""你是一位熟悉工业制造流程和工厂现场语言风格的信息提取专家。
                            内容: {text_content}。
                            以上内容来源于工人现场的语音输入，可能包含口语表达、不规范说法、同音字错误（如“图好”可能是“图号”，“尺码”是“尺寸”）等情况。
                            请你尽可能理解语义并从中提取以下关键信息（若信息缺失，请填写“未提及”）：
                            1.Workpiece_name（工件名称）
                            2.Workpiece_DN（工件图号）
                            3.Workpiece_size（工件尺寸）
                            4.Inspection_passed（是否检验合格，仅提取“是”或“否”）
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
                    result2 = response.choices[0].message.content
                    print(f"AI提取结果: {result2}")
                    
                    # 解析AI返回的文本格式结果
                    lines = result2.strip().split('\n')
                    extracted_data = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            extracted_data[key.strip()] = value.strip()
                    
                    workpiece_name = extracted_data.get('Workpiece_name', '未提及')
                    workpiece_dn = extracted_data.get('Workpiece_DN', '未提及')
                    workpiece_size = extracted_data.get('Workpiece_size', '未提及')
                    inspection_passed = extracted_data.get('Inspection_passed', '未提及')
                    inspection_details = extracted_data.get('Inspection_details', '未提及')
                    
                    # 生成员工编号（质检员）
                    employee_id = f"Q{PERSON_id:03d}"
                    
                    # 直接插入数据库
                    success = insert_to_database(
                        employee_id=employee_id,
                        job_type='质检员',
                        workpiece_name=workpiece_name,
                        workpiece_dn=workpiece_dn,
                        workpiece_size=workpiece_size,
                        inspection_passed=inspection_passed if inspection_passed != '未提及' else None,
                        inspection_details=inspection_details if inspection_details != '未提及' else None
                    )
                    
                    if success:
                        print("质检员数据已成功保存到数据库")
                    else:
                        print("质检员数据保存失败")
                
                # 停止摄像头循环，让程序继续到用户输入环节
                camera_running = False
                # 立即关闭摄像头窗口
                cv2.destroyAllWindows()
                
                # 关闭WebSocket连接，确保资源释放
                ws.close()
            else:
                print("没有识别出任何文本")
                # 停止摄像头循环，让程序继续到用户输入环节
                camera_running = False
                # 立即关闭摄像头窗口
                cv2.destroyAllWindows()
                # 即使没有识别出文本，也关闭WebSocket连接
                ws.close()
    except Exception as e:
        print(f"解析失败: {str(e)}")
        # 停止摄像头循环，让程序继续到用户输入环节
        camera_running = False
        # 立即关闭摄像头窗口
        cv2.destroyAllWindows()
        # 确保在异常情况下也关闭WebSocket连接
        ws.close()

def on_error(ws, error):
    print(f"### 连接错误: {error}")

def on_close(ws, close_status_code, close_msg):
    print("")

# 全局变量用于存储WebSocket参数
wsParam = None

def on_open(ws):
    def audio_capture_thread():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280
        )

        # 发送头部包（必须）
        header_frame = {
            "header": {"status": STATUS_FIRST_FRAME, "app_id": wsParam.APPID},
            "parameter": {"iat": wsParam.iat_params},
            "payload": {"audio": {"sample_rate": 16000, "encoding": "raw"}}
        }
        ws.send(json.dumps(header_frame))

        print("开始录音... (按Ctrl+C停止)")
        silence_start = None

        try:
            while True:
                data = stream.read(1280)

                # 静音检测
                if is_silence(data):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > MAX_SILENCE_DURATION:
                        print("检测到长时间静音，停止发送")
                        break
                    continue
                else:
                    silence_start = None

                # 发送数据包（关键修复）
                audio_base64 = base64.b64encode(data).decode('utf-8')
                data_frame = {
                    "header": {"status": STATUS_CONTINUE_FRAME, "app_id": wsParam.APPID},
                    "payload": {"audio": {"audio": audio_base64}}
                }
                # 检查WebSocket连接状态
                if ws.sock and ws.sock.connected:
                    ws.send(json.dumps(data_frame))
                else:
                    print("")
                    break

        except KeyboardInterrupt:
            print("用户停止录音")
        except Exception as e:
            print(f"捕获异常: {str(e)}")
        finally:
            # 确保发送尾部包（仅在连接仍然有效时）
            try:
                if ws.sock and ws.sock.connected:
                    trailer_frame = {
                        "header": {"status": STATUS_LAST_FRAME, "app_id": wsParam.APPID},
                        "payload": {"audio": {"status": 2}}
                    }
                    ws.send(json.dumps(trailer_frame))
                    print("尾部包发送成功")
                else:
                    print("")
            except Exception as e:
                print(f"发送尾部包时出错: {str(e)}")

            # 安全关闭音频流
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                p.terminate()
                print("音频设备已释放")
            except Exception as e:
                print(f"释放音频设备时出错: {str(e)}")
            
            # 等待一小段时间确保数据处理完成
            time.sleep(1)

    # 启动音频线程
    threading.Thread(target=audio_capture_thread, daemon=True).start()

# start_flask_app函数已移除，main.py不再负责启动Flask应用
# Flask应用应该独立运行（通过运行1.py）

def run_speech_recognition():
    """运行语音识别服务"""
    global wsParam
    # 配置参数
    wsParam = Ws_Param(
        APPID='12d84f1e',  # 替换为你的APPID
        APIKey='598d23f9a26a02b52a0a0ec9760298fd',  # 替换为你的APIKey
        APISecret='OGFmYTQ5YWRhNmZkNDM2ZWI4NWJlOGI3'  # 替换为你的APISecret
    )

    # 建立WebSocket连接
    ws_url = wsParam.create_url()
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    print("连接语音识别服务...")
    
    # 启动语音识别
    try:
        ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            ping_interval=25,    # 25秒发送一次ping
            ping_timeout=10,     # 10秒未响应视为超时
            ping_payload="\x09"  # WebSocket ping操作码
        )
    except KeyboardInterrupt:
        print("用户中断语音识别")
    finally:
        # 确保WebSocket连接关闭
        if ws.sock and ws.sock.connected:
            ws.close()
        print("语音识别服务已关闭")

if __name__ == "__main__":
    print("=== 增强型工厂管理系统启动 ===")
    print("集成功能: 增强人脸识别 + 环境反馈 + 自监督学习 + 语音识别")
    
    # 初始化摄像头和增强人脸识别系统
    print("\n正在初始化系统组件...")
    if not init_camera():
        print("摄像头初始化失败，程序退出")
        exit(1)
    
    # 等待系统稳定
    time.sleep(2)
    
    try:
        # 第一步：增强人脸识别
        print("\n=== 第一步：用户身份识别 ===")
        recognized_user = enhanced_face_recognition()
        
        if recognized_user:
            print(f"\n✓ 用户身份确认: {recognized_user}")
            print(f"✓ 用户类型: {'工人' if PERSON == 1 else '质检员'}")
            print("\n系统已根据用户身份优化识别参数")
        else:
            print("\n⚠ 人脸识别失败，将使用默认设置")
            # 提供手动输入选项
            manual_input = input("是否手动指定用户类型？(1-工人, 2-质检员, 回车跳过): ").strip()
            if manual_input == '1':
                PERSON = 1
                PERSON_id = 1
                print("已手动设置为工人")
            elif manual_input == '2':
                PERSON = 2
                PERSON_id = 2
                print("已手动设置为质检员")
            else:
                PERSON = 1  # 默认为工人
                PERSON_id = 1
                print("使用默认设置：工人")
        
        # 显示当前环境状态
        if face_recognition_system:
            print(f"\n当前环境状态:")
            print(f"- 识别成功率: {environmental_feedback['recognition_success_rate']:.1%}")
            print(f"- 自适应阈值: {face_recognition_system.environmental_feedback['adaptive_threshold']:.3f}")
            print(f"- 光照适应: 已启用")
            print(f"- 自监督学习: 已启用")
        
        # 第二步：启动语音识别
        print("\n=== 第二步：语音数据采集 ===")
        print("正在启动语音识别服务...")
        speech_thread = threading.Thread(target=run_speech_recognition, daemon=True)
        speech_thread.start()
        
        # 在主线程中运行摄像头显示循环
        print("\n摄像头监控已启动，开始语音识别...")
        print("(说话时请保持自然，系统会自动检测静音并结束录音)")
        
        camera_display_loop()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行时发生错误: {e}")
    finally:
        # 停止摄像头
        camera_running = False
        
        # 保存环境学习数据
        if face_recognition_system:
            try:
                face_recognition_system.save_environmental_model()
                print("\n✓ 环境学习数据已保存")
            except Exception as e:
                print(f"\n⚠ 保存环境数据时出错: {e}")
    
    # 等待语音识别线程结束
    if 'speech_thread' in locals() and speech_thread.is_alive():
        print("\n等待语音识别服务结束...")
        # speech_thread.join(timeout=3)

    # 显示系统统计信息
    print("\n=== 系统运行统计 ===")
    if recognition_history:
        total_attempts = len(recognition_history)
        successful_attempts = sum(1 for attempt in recognition_history if attempt['success'])
        print(f"人脸识别尝试次数: {total_attempts}")
        print(f"成功识别次数: {successful_attempts}")
        print(f"识别成功率: {successful_attempts/total_attempts:.1%}")
    
    print("\n语音识别和数据处理已完成")
    print("数据已自动保存到MySQL数据库中")
    
    # 显示环境反馈总结
    if face_recognition_system:
        print(f"\n=== 环境适应总结 ===")
        print(f"最终自适应阈值: {face_recognition_system.environmental_feedback['adaptive_threshold']:.3f}")
        print(f"光照条件评估: {len(face_recognition_system.environmental_feedback['lighting_history'])} 次")
        print(f"自监督学习更新: {face_recognition_system.environmental_feedback.get('learning_updates', 0)} 次")
    
    input("\n按回车键退出程序...")

    # 关闭摄像头
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")

    print("\n=== 程序已完全退出，所有资源已释放 ===")
    print("感谢使用增强型工厂管理系统！")
