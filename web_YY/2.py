import threading
import time

import numpy as np
import websocket
import base64
import json
import ssl
import pyaudio
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import hashlib
import hmac
import requests
import json
import re
# 状态常量
STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2
SILENCE_THRESHOLD = 200  # 静音阈值（根据麦克风调整，降低以提高灵敏度）
MAX_SILENCE_DURATION = 8  # 最大允许静音时间（秒）
PERSON = 0
PERSON_id = 0


# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-db429a52f01f4d5a9feff537972d738d", base_url="https://api.deepseek.com")






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
# WebSocket回调函数（保持不变）
# 用于存储所有识别出的文本
all_recognized_text = []

def on_message(ws, message):
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
                            请严格按照以下格式返回（不要添加任何解释或额外内容）,请按照以下键值将答案格式化为 JSON 格式： Workpiece_name, Workpiece_DN, Workpiece_size, Start_time, Process_type, Self_check_result.
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
                    result1 = re.sub(r'```json|```', '', result1)
                    print(result1)
                    # Send result to Flask API
                    try:
                        response = requests.post(
                            'http://localhost:5001/api/result',
                            json={
                                'result': result1,
                                'person': PERSON,
                                'person_id': PERSON_id  # 添加person_id确保兼容
                            },
                            timeout=5
                        )
                        if response.status_code == 200:
                            print("结果已发送到Flask应用")
                        else:
                            print(f"发送结果失败，状态码: {response.status_code}")
                    except Exception as e:
                        print(f"发送结果到Flask应用出错: {str(e)}")
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
                            {"role": "user", "content": prompt,'id':PERSON_id},
                        ],
                        stream=False
                    )
                    result2 = response.choices[0].message.content
                    result2 = re.sub(r'```json|```', '', result2)
                    print(result2)
                    # Send result to Flask API
                    try:
                        response = requests.post(
                            'http://localhost:5001/api/result',
                            json={
                                'result': result2, 
                                'person': PERSON,
                                'person_id': PERSON_id  # 添加person_id确保兼容
                            },
                            timeout=5
                        )
                        if response.status_code == 200:
                            print("结果已发送到Flask应用")
                        else:
                            print(f"发送结果失败，状态码: {response.status_code}")
                    except Exception as e:
                        print(f"发送结果到Flask应用出错: {str(e)}")
                
                # 关闭WebSocket连接，确保资源释放
                ws.close()
            else:
                print("没有识别出任何文本")
                # 即使没有识别出文本，也关闭WebSocket连接
                ws.close()
    except Exception as e:
        print(f"解析失败: {str(e)}")
        # 确保在异常情况下也关闭WebSocket连接
        ws.close()

def on_error(ws, error):
    print(f"### 连接错误: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### 连接关闭 ###")

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
                ws.send(json.dumps(data_frame))

        except KeyboardInterrupt:
            print("用户停止录音")
        except Exception as e:
            print(f"捕获异常: {str(e)}")
        finally:
            # 确保发送尾部包
            trailer_frame = {
                "header": {"status": STATUS_LAST_FRAME, "app_id": wsParam.APPID},
                "payload": {"audio": {"status": 2}}
            }
            ws.send(json.dumps(trailer_frame))

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

def start_flask_app():
    """启动Flask应用"""
    import subprocess
    import sys
    try:
        print("正在启动Flask应用...")
        # 启动1.py Flask应用
        subprocess.run([sys.executable, "1.py"], cwd="/Users/caozhixuan/Desktop/web_YY")
    except Exception as e:
        print(f"启动Flask应用失败: {str(e)}")

if __name__ == "__main__":
    # 配置参数（替换为你的凭证）
    # 请在讯飞开放平台(https://www.xfyun.cn/)注册账号
    # 创建应用并添加语音听写流式服务，获取APPID、APIKey和APISecret
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

    # 关键修复：添加心跳维持连接和超时机制
    try:
        ws.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            ping_interval=25,    # 25秒发送一次ping
            ping_timeout=10,     # 10秒未响应视为超时
            ping_payload="\x09"  # WebSocket ping操作码
        )
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        # 确保WebSocket连接关闭
        if ws.sock and ws.sock.connected:
            ws.close()
        print("麦克风已关闭，WebSocket连接已断开")
        
        # 等待用户输入q来上传到1.py
        print("\n请输入 'q' 来启动上传功能并查看结果，或输入其他任意键直接退出程序:")
        user_input = input().strip().lower()
        
        if user_input == 'q':
            print("正在启动上传功能...")
            start_flask_app()
        else:
            print("程序已退出")
        
        print("程序已完全退出，所有资源已释放")
