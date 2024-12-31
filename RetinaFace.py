import cv2
import numpy as np
import pandas as pd
import time
import ast
from retinaface import RetinaFace
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def detectar(imagem, bboxesVerdadeiras, iouThreshold=0.5):
    image = cv2.imread(imagem)
    tempoInicio = time.time()
    faces = RetinaFace.detect_faces(image)
    tempoDeteccao = time.time() - tempoInicio

    bboxDetectada = []
    if isinstance(faces, dict):  # Se houver faces detectadas
        bboxDetectada = [face['facial_area'] for face in faces.values()]

    verdadeiros = []
    preditos = []
    iouRes = []

    for bbox in bboxesVerdadeiras:
        bboxConvertida = []
        for coord in bbox:
            bboxConvertida.append(float(coord))
        bbox = bboxConvertida
        match = False
        iouMax = 0
        for detBbox in bboxDetectada:
            iou = calcularIou(bbox, detBbox)
            iouMax = max(iouMax, iou)
            if iou >= iouThreshold:
                match = True
                break
        verdadeiros.append(1)
        if match:
            preditos.append(1)
        else:
            preditos.append(0)
        iouRes.append(iouMax)
    if not bboxesVerdadeiras:
        verdadeiros.append(0)
        preditos.append(0)
        iouRes.append(0)
    
    return verdadeiros, preditos, iouRes, tempoDeteccao

def calcularIou(bboxA, bboxB):
    xA, yA, xA2, yA2 = bboxA
    xB, yB, xB2, yB2 = bboxB

    xI = max(xA, xB)
    yI = max(yA, yB)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)
    
    areaInter = max(0, xI2 - xI + 1) * max(0, yI2 - yI + 1)
    areaBboxA = (xA2 - xA + 1) * (yA2 - yA + 1)
    areaBboxB = (xB2 - xB + 1) * (yB2 - yB + 1)
    iou = areaInter / float(areaBboxA + areaBboxB - areaInter)

    return iou

meta = 'fddb_dataset.csv'
df = pd.read_csv(meta)

df['face_location'] = df['face_location'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

rotuladosVerdadeiros = []
rotuladosPreditos = []
resIou = []
tempoDeteccao = 0
imagens = 0

for index, row in df.iterrows():
    imagem = row['path']
    bboxesVerdadeiras = [row['face_location']]  # Ajustar se houver mais de uma face por imagem
    verdadeiros, preditos, iouRes, tempoDeteccao = detectar(imagem, bboxesVerdadeiras)
    rotuladosVerdadeiros.extend(verdadeiros)
    rotuladosPreditos.extend(preditos)
    resIou.extend(iouRes)
    tempoDeteccao += tempoDeteccao
    imagens += 1
    if (imagens) % 100 == 0:
        print(f'{imagens} imagens processadas.')

precision, recall, f1, _ = precision_recall_fscore_support(rotuladosVerdadeiros, rotuladosPreditos, average='binary')
accuracy = accuracy_score(rotuladosVerdadeiros, rotuladosPreditos)
tempoDeteccaoMedio = tempoDeteccao / imagens

print(f'Precisão: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Acurácia: {accuracy}')
print(f'Tempo Médio de Detecção: {tempoDeteccaoMedio} segundos')