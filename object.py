import smtplib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from email.mime.text import MIMEText

import cv2
import time  # 시간 관련 라이브러리 추가

# 웹캠 캡처
cap = cv2.VideoCapture(1)

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# 초기값 설정
fall_detection = 0

# 녹화 관련 설정
record = False
out = None
frame_count = 0
output_directory = 'D:/Users/gnmun/Documents/saved/'  # 저장할 경로
video_filename = 'video'  # 파일 이름의 시작 부분

# 메일 관련 설정
smtpName = "smtp.naver.com"
smtpPort = 587

sendEmail = "mun990202@naver.com"
password = "munsj6203"
recvEmail = "rudxo1276@naver.com"



# 웹캠 캡처 루프
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # 사람 감지 후 처리
            person_detected = False
            for i, box in enumerate(np.squeeze(boxes)):
                if np.squeeze(classes).astype(np.int32)[i] == 1:  # 클래스 1은 '사람'을 나타냅니다.
                    ymin, xmin, ymax, xmax = box
                    if xmax > ymax:  # x가 y보다 길면
                        fall_detection = 1
                        person_detected = True
            if not person_detected:
                fall_detection = 0

            # fall_detection 값에 따라 화면에 메시지 표시
            if fall_detection == 1:
                cv2.putText(image_np, 'Person Detected!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                title = "낙상 감지"
                content = "저장된 영상을 확인하세요"

                msg = MIMEText(content)
                msg['From'] = sendEmail
                msg['To'] = recvEmail
                msg['Subject'] = title

                s = smtplib.SMTP(smtpName, smtpPort)
                s.starttls()
                s.login(sendEmail, password)
                s.sendmail(sendEmail, recvEmail, msg.as_string())
                s.close()

                # 녹화 시작
                if not record:
                    video_filename = 'video'
                    frame_count = 0
                    while os.path.exists(output_directory + video_filename + '.avi'):
                        frame_count += 1
                        video_filename = 'video' + str(frame_count)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_directory + video_filename + '.avi', fourcc, 20.0, (640, 480))
                    record = True
            else:
                # 녹화 중지
                if record:
                    record = False
                    out.release()

            # 녹화 중인 경우 프레임 저장
            if record:
                out.write(image_np)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                if record:
                    record = False
                    out.release()
                break