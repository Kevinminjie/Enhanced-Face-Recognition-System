from flask import Flask, render_template, request
import time
import os
import json
from threading import Lock
app = Flask(__name__)

# 全局变量存储结果
current_result = None
current_person = 1  # 默认人员类型
current_person = 1
result_lock = Lock()

@app.route('/')
def index():
    global current_result, current_person
    with result_lock:
        try:
            if not current_result:
                return render_template('index.html')

            data = current_result
            
            if current_person == 1:
                return render_template('index.html',
                    person_type=current_person,
                    workpiece_name=data.get('Workpiece_name', '未提及'),
                    workpiece_dn=data.get('Workpiece_DN', '未提及'),
                    workpiece_size=data.get('Workpiece_size', '未提及'),
                    start_time=data.get('Start_time', '未提及'),
                    process_type=data.get('Process_type', '未提及'),
                    self_check=data.get('Self_check_result', '未提及'),
                    id=PERSON_id)
            else:
                return render_template('index.html',
                    person_type=current_person,
                    workpiece_name=data.get('Workpiece_name', '未提及'),
                    workpiece_dn=data.get('Workpiece_DN', '未提及'),
                    workpiece_size=data.get('Workpiece_size', '未提及'),
                    Inspection_passed=data.get('Inspection_passed', '未提及'),
                    Inspection_details=data.get('Inspection_details', '未提及'),
                    id=PERSON_id)

        except Exception as e:
            print(f"解析JSON出错: {str(e)}")
            return render_template('index.html', error=str(e))

@app.route('/api/result', methods=['POST'])
def receive_result():
    global current_result, current_person, PERSON_id
    with result_lock:
        try:
            current_result = request.json.get('result')
            current_person = request.json.get('person')  # 默认工人
            PERSON_id = request.json.get('person_id')  # 默认工人
            print(PERSON_id)
            print(current_person)
            
            print(f"接收到原始数据: {current_result}")
            if isinstance(current_result, str):
                # 尝试解析字符串为字典
                try:
                    current_result = json.loads(current_result)
                except json.JSONDecodeError:
                    # 处理键值对字符串
                    data = {}
                    for line in current_result.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            data[key.strip()] = value.strip()
                    current_result = data
            
            print(f"处理后数据(PERSON={current_person},{PERSON_id}): {current_result}")
            return {'status': 'success'}
        except Exception as e:
            print(f"处理结果出错: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
