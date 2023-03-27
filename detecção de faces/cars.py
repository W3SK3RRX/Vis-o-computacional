import cv2
imagem = cv2.imread("img/car.jpg")
imagem_cinza= cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
janela = 'image'
detector_carros = cv2.CascadeClassifier('cascades/cars.xml')
deteccoes = detector_carros.detectMultiScale(imagem_cinza, scaleFactor=1.1)
for x, y, w, h in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow(janela, imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()