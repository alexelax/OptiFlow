import cv2
import pytesseract
import CVfunction as fun
import transform as trans
import numpy as np




        

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
imgName='Auto_3.png'
# Lettura l'immagine
img = cv2.imread(imgName)      #fare prima mogrify -format png *.png

# Conversione immagine in scala di grigio
gray = fun.get_grayscale(img)

# Assegnazione 0/1 all'immagine
edged = fun.canny(gray)

# Identificazione dei contorni
crop,spigoli = fun.edge_detect(gray, edged, img)

print(spigoli)
# Autorotate dell'immagine
#rotate = fun.auto_rotate(crop)

#edges = fun.getEdges(crop)

#[(73, 239), (356, 117), (475, 265), (187, 443)]
#[[[X,Y]],[[X,Y]]]

spigoli  = [ (spigolo[0][0],spigolo[0][1])  for spigolo in spigoli  ]
spigoli = np.array(spigoli, dtype = "float32")
print(spigoli)


warped=trans.four_point_transform(gray,spigoli)


#livello selezione colore
#warped=fun.thresholding(warped)

# Lettura dei caratteri
#fun.charaters_detect(warped, crop)

# Lettura parole
#fun.words_detect(warped)

# Visualizzazione finale
cv2.imwrite(imgName+"out.png", warped)
cv2.imshow('Risultato', warped)
cv2.waitKey(0)



