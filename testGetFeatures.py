import csv
import os
import cv2
import dlib
import numpy as np

# 人脸数据目录
path_face_dir = "./data/database_faces/"
person_list = os.listdir(path_face_dir)

# Dlib 初始化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1(
    "./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
)

# 提取单张图像的人脸特征
def extract_features(path_img):
    img_rd = cv2.imdecode(np.fromfile(path_img, dtype=np.uint8), -1)
    faces = detector(img_rd, 1)
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
    return face_descriptor

# 写入CSV
with open("./data/features_all.csv", "w", newline="", encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    header = ["name", "id", "type"] + [f"f{i}" for i in range(128)]
    writer.writerow(header)

    for person in person_list:
        features_list = []
        photos_list = os.listdir(os.path.join(path_face_dir, person))

        # 提取人脸特征
        if photos_list:
            for photo in photos_list:
                img_path = os.path.join(path_face_dir, person, photo)
                features_128D = extract_features(img_path)
                print("图片" + photo + "已录入！")
                if features_128D != 0:
                    features_list.append(features_128D)

        # 计算平均特征
        if features_list:
            features_mean = np.array(features_list).mean(axis=0)
        else:
            features_mean = np.zeros(128, dtype=int, order='C')

        # 假设文件夹命名格式是：张三_001_操作工
        try:
            name, id_str, type_str = person.split("_")
        except ValueError:
            print(f"命名格式错误：{person}，请使用 '姓名_工号_工种' 格式")
            continue

        row = [name, id_str, type_str] + list(features_mean)
        writer.writerow(row)
        print(f"{name} 的人脸录入完成！")
