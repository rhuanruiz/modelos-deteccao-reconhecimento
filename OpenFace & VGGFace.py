import os
import time
import csv
from deepface import DeepFace
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def meta(pairs, lfw):
    pares = []
    with open(pairs, "r") as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            row = [item for item in row if item] 
            if len(row) == 3:
                nome, idx1, idx2 = row
                img1 = os.path.join(lfw, nome, f"{nome}_{int(idx1):04d}.jpg")
                img2 = os.path.join(lfw, nome, f"{nome}_{int(idx2):04d}.jpg")
                pares.append((img1, img2, 1))
            else:
                print(f"row ignorada: {row}")
    return pares

lfw = "LFW/lfw-deepfunneled/lfw-deepfunneled"  
pairs = "LFW/pairs.csv" 

pares = meta(pairs, lfw)

verdadeiros = []
preditos = []
tempo = []

for idx, (img1, img2, rotulo) in enumerate(pares):
    tempoInicio = time.time()
    res = DeepFace.verify(img1, img2, model_name="OpenFace", distance_metric="cosine", enforce_detection=False) # Alterar model_name para modelo almejado para teste.
    tempoFinal = time.time()
    verdadeiros.append(rotulo)
    preditos.append(1 if res["verified"] else 0)
    tempo.append(tempoFinal - tempoInicio)
    if (idx + 1) % 100 == 0:
        print(f'{idx + 1}/{len(pares)} imagens processadas.')

precision = precision_score(verdadeiros, preditos, average="binary")
recall = recall_score(verdadeiros, preditos, average="binary")
f1 = f1_score(verdadeiros, preditos, average="binary")
accuracy = accuracy_score(verdadeiros, preditos)
tempoMedio = sum(tempo) / len(tempo)

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Tempo Médio: {tempoMedio:.4f} segundos.")
