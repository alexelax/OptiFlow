import cv2
import pytesseract
import CVfunction as fun

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Lettura l'immagine
img = cv2.imread('Targa3.png')      #fare prima mogrify -format png *.png

# Conversione immagine in scala di grigio
gray = fun.get_grayscale(img)

# Assegnazione 0/1 all'immagine
edged = fun.canny(gray)

# Identificazione dei contorni
crop = fun.edge_detect(gray, edged, img)

# Autorotate dell'immagine
#rotate = fun.auto_rotate(crop)

# Lettura dei caratteri
fun.charaters_detect(img, crop)

# Lettura parole
#fun.words_detect(img, rotate)

# Visualizzazione finale
cv2.imshow('Risultato', crop)
cv2.waitKey(0)
