import cv2

imagem = cv2.imread('img/people2.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
janela = 'image'
detector_facial = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32),
                                             maxSize=(100, 100))
for x, y, w, h in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow(janela, imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()