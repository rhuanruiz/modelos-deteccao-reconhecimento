import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import ast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detectar(imagem, bboxesVerdadeiras):
    image = cv2.imread(imagem)
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tempoInicio = time.time()
    faces = face_detection.process(imageRgb)
    tempoDeteccao = time.time() - tempoInicio

    bboxes = []
    if faces.detections:
        for detection in faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = imageRgb.shape
            bboxes.append([int(bboxC.xmin * w), int(bboxC.ymin * h),
                          int(bboxC.width * w + bboxC.xmin * w), int(bboxC.height * h + bboxC.ymin * h)])

    nFacesDetectadas = len(bboxes)
    nBboxesVerdadeiras = len(bboxesVerdadeiras)

    FP = nFacesDetectadas - nBboxesVerdadeiras if nFacesDetectadas > nBboxesVerdadeiras else 0

    verdadeiros = 1 if nBboxesVerdadeiras > 0 else 0
    preditos = 1 if nFacesDetectadas > 0 else 0

    return verdadeiros, preditos, FP, tempoDeteccao

meta = 'fddb_dataset.csv'
df = pd.read_csv(meta)

df['face_location'] = df['face_location'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

rotulosVerdadeiros = []
rotulosPreditos = []
tempoDeteccao = 0
imagens = 0
nFP= 0

for index, row in df.iterrows():
    imagem = row['path']
    bboxesVerdadeiras = row['face_location']
    verdadeiros, preditos, FP, tempoDeteccao = detectar(imagem, bboxesVerdadeiras)
    rotulosVerdadeiros.append(verdadeiros)
    rotulosPreditos.append(preditos)
    nFP+= FP
    tempoDeteccao += tempoDeteccao
    imagens += 1
    if (imagens) % 100 == 0:
        print(f'{imagens} imagens processadas.')

precision, recall, f1, _ = precision_recall_fscore_support(rotulosVerdadeiros, rotulosPreditos, average='binary')
accuracy = accuracy_score(rotulosVerdadeiros, rotulosPreditos)
tempoDeteccaoMedio = tempoDeteccao / imagens

print(f'Precisão (sklearn): {precision}')
print(f'Recall (sklearn): {recall}')
print(f'F1-Score: {f1}')
print(f'Acurácia: {accuracy}')
print(f'Tempo Médio de Detecção: {tempoDeteccaoMedio} segundos')