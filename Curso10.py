import cv2
print(cv2.__version__)
import dlib
print(dlib.__version__)
import matplotlib.pyplot as plt
imagem = cv2.imread("imagens/px-people.jpg")
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.imshow(imagem)
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
classificador = cv2.CascadeClassifier("classificadores/haarcascade_frontalface_default.xml")
faces = classificador.detectMultiScale(imagem_gray, 1.3, 5)
faces[0]
imagem_anotada = imagem.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(imagem_anotada, (x,y), (x+w, y+h), (255, 255, 0), 2)

plt.figure(figsize=(20,10))
plt.imshow(imagem_anotada)

face_imagem = 0
for (x,y,w,h) in faces:
    face_imagem += 1
    imagem_roi = imagem[y:y+h, x:x+w]
    imagem_roi = cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite("face_" + str(face_imagem) + ".png", imagem_roi)