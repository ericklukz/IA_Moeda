import cv2 as cv
import mediapipe as mp
import math
from flask import Flask, jsonify
import numpy as np
from urllib.request import urlopen
import requests
from ultralytics import YOLO

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE = [4, 5, 51, 275, 45]

moeda = 2.5


def process_image(caminho, scale_percent=40):
    # Carrega a imagem
    global r
    image = cv.imread(caminho)

    # Redimensiona a imagem para diminuir o zoom (ajuste conforme necessário)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    # Aplica equalização de histograma para melhorar o contraste
    gray_eq = cv.equalizeHist(gray)

    blurred = cv.medianBlur(gray_eq, 5)

    # Ajuste os valores de param1, param2, minRadius e maxRadius conforme necessário
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=50,
        param2=30,
        minRadius=35,
        maxRadius=50,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(resized_image, (x, y), r, (255, 0, 255), 3)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        frame = cv.flip(resized_image, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            (x_cx, x_cy), c_radius = cv.minEnclosingCircle(mesh_points[NOSE])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            center = np.array([x_cx, x_cy], dtype=np.int32)

            cv.circle(frame, center_left, 1, (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, 1, (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center, 1, (255, 0, 255), 1, cv.LINE_AA)

            cv.line(frame, center_right, center_left, (255, 0, 255), 3, cv.LINE_AA)
            cv.line(frame, center_right, center, (255, 0, 255), 3, cv.LINE_AA)
            cv.line(frame, center_left, center, (255, 0, 255), 3, cv.LINE_AA)

            distancia_pupilas = math.sqrt(
                (center_right[0] - center_left[0]) ** 2 + (center_left[1] - center_right[1]) ** 2)
            direito_naso = math.sqrt((center_right[0] - center[0]) ** 2 + (center[1] - center_right[1]) ** 2)
            esquerdo_naso = math.sqrt((center[0] - center_left[0]) ** 2 + (center_left[1] - center[1]) ** 2)
            media = (direito_naso + esquerdo_naso) / 2
            diametro = r * 2
            global distancia_pupilas_conversao, distancia_naso_direita_conversao, distancia_naso_esquerda_conversao
            distancia_pupilas_conversao = round((distancia_pupilas * moeda) / diametro , 3)
            distancia_naso_direita_conversao = round((direito_naso * moeda) / diametro , 3)
            distancia_naso_esquerda_conversao = round((esquerdo_naso * moeda) / diametro , 3)
            media_conversao = (media * moeda) / diametro

            print("Distância entre pupilas: ", distancia_pupilas_conversao)
            print("Distância entre a pupila direita e o nariz: ", distancia_naso_direita_conversao)
            print("Distância entre a pupila esquerda e o nariz: ", distancia_naso_esquerda_conversao)
            print("Diâmetro da moeda: ", r * 2)

            cv.putText(frame, str("Distancia entre pupilas: " + str(round(distancia_pupilas_conversao, 3)) + " cm"),
                       (5, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 0, 255)
            cv.putText(frame, str("Distancia entre a pupila direita e o nariz: " + str(
                round(distancia_naso_direita_conversao, 3)) + " cm"), (5, 60), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 0, 255)
            cv.putText(frame, str("Distancia entre a pupila esquerda e o nariz: " + str(
                round(distancia_naso_esquerda_conversao, 3)) + " cm"), (5, 90), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 0, 255)
            cv.putText(frame, str("Media das distancias: " + str(round(media_conversao, 3)) + " cm"), (5, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 1, 255, 0, 255)
            cv.putText(frame, str("Diametro da moeda: " + str(r * 2) + " px"), (5, 150), cv.FONT_HERSHEY_SIMPLEX, 1,
                       255, 0, 255)

        @app.route('/api/distancia', methods=['GET'])
        def obter_distancias():
            return jsonify(DP=distancia_pupilas_conversao ,
            DNPLeft=distancia_naso_esquerda_conversao ,
            DNPRight=distancia_naso_direita_conversao)

        cv.imshow('img', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        app.run(host = '0.0.0.0')


if __name__ == "__main__":
    image_url = "https://bucketimage.blob.core.windows.net/images/1709673500532image%2Fjpeg"
    response = requests.get(image_url)
    with open("imagem.jpg", "wb") as f:
        f.write(response.content)
    resp = urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    process_image('imagem.jpg', scale_percent=40)
