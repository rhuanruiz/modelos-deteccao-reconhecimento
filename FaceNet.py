import time
import numpy as np
from sklearn.datasets import fetch_lfw_people
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


lfw = fetch_lfw_people(resize=0.4)
modelo = InceptionResnetV1(pretrained='vggface2').eval()

preProcessamento = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

times = []
embeddings = []
rotulos = lfw.target 
tamanhoBatch = 32 
nBatches = len(lfw.images) // tamanhoBatch + 1

for batchIdx in range(nBatches):
    idxInicial = batchIdx * tamanhoBatch
    idxFinal = min((batchIdx + 1) * tamanhoBatch, len(lfw.images))
    batchImagens = lfw.images[idxInicial:idxFinal]
    batchTensors = []
    for img in batchImagens:
        imgs = Image.fromarray((img * 255).astype(np.uint8))
        imgs = imgs.convert('RGB')
        imgTensor = preProcessamento(imgs).unsqueeze(0)
        batchTensors.append(imgTensor)
    batchTensors = torch.cat(batchTensors)
    tempoInicio = time.time()
    with torch.no_grad():
        batchEmbeddings = modelo(batchTensors)
    tempoFinal = time.time()
    times.append(tempoFinal - tempoInicio)
    embeddings.append(batchEmbeddings)
    if idxFinal % 100 == 0:
        print(f'{idxFinal} imagens processadas.')

tempoMedio = sum(times) / len(times)
embeddings = torch.cat(embeddings).numpy()

rotulosPreditos = []
for i in range(len(embeddings)):
    dists = np.linalg.norm(embeddings[i] - embeddings, axis=1)
    rotulosPreditos.append(np.argmin(dists)) 

accuracy = accuracy_score(labels, rotulosPreditos)
precision = precision_score(labels, rotulosPreditos, average='weighted', zero_division=0)
recall = recall_score(labels, rotulosPreditos, average='weighted', zero_division=0)
f1 = f1_score(labels, rotulosPreditos, average='weighted', zero_division=0)

print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Tempo Médio: {tempoMedio} segundos")
