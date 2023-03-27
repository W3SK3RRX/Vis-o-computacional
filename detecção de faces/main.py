import cv2 #OpenCV

imagem = cv2.imread('img/imagem.png') # Atribui o caminho da imagem a uma variável

imagem = cv2.resize(imagem, (800, 600)) # redimencionar a imagem

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Muda a escala de cores da imagem

# print(imagem_cinza.shape) # mostra o tamanho da imagem
janela = 'image' # janela para a imagem ser exibida
# cv2.imshow(janela, imagem_cinza) # exibe a imagem
cv2.waitKey(0) # evita que a janela seja fechada
cv2.destroyAllWindows() #fecha a janela

detector_facial = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml') # Detector de faces já treinado
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor = 1.09) # Função que detecta as faces

# parâmetros detectMultiScale:
# scaleFactor - ajusta a escala da imagem para melhorar a detecção
# minNeighbors - ajusta o número mínimo de vizinhos para se considerar uma face
# minSize - tamanho mínimo da face
# maxSize - tamanho máximo da face

#print(deteccoes)

for x, y, w, h in deteccoes:
    # print(x, y, w, h)
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow(janela, imagem)
cv2.waitKey(0) # evita que a janela seja fechada
cv2.destroyAllWindows() #fecha a janela

