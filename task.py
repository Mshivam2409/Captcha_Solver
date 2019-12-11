import time as tm
##################
#Add libraries to import here
import os
import numpy as np
import cv2
##################

def function():
    #In this function you have to implement the method to import all the images 
    #present in the same directory apply processing and return individual alphabets
    #in the same directory after the processing.
    #If there are multiple generated images of the same alphabet then name 
    #the file as Alphabet_name_count.jpg/png
    pwd = os.getcwd()
    path = pwd + '/images'
    os.chdir(path)
    files = []
    for r, d, f in os.walk(path):
         for file in f:
            if '.png' in file:
                  files.append(os.path.join(r, file))
    ewe=[]
    for i in range (0,len(files)):
         ttt=i+1
         letters = list((os.path.splitext(os.path.basename(files[i]))[0]))
         image_name = os.path.basename(files[i])
         t=cv2.imread(image_name)
         img= t.copy()
         captcha_image = image_name
         counts={}
         height = np.size(img, 0)
         width = np.size(img, 1)
         for i in range(1,height):
             for j in range(1,width): 
                 b1=img[i,j,0]
                 r1=img[i,j,2]
                 g1=img[i,j,1]
                 b=img[0,0,0]
                 g=img[0,0,1]
                 r=img[0,0,2]        
                 if(b1==b and g1==g and r1==r ):
                     img.itemset((i, j, 0), 0)
                     img.itemset((i, j, 1), 0)
                     img.itemset((i, j, 2), 0)

         img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

         kernel1 = np.ones((3,3), np.uint8)
         w2 = cv2.erode(img, kernel1, iterations=2)

         ret,w2 = cv2.threshold(w2,150,255,cv2.THRESH_BINARY_INV)
         w2 = cv2.dilate(w2, kernel1, iterations=2)
         w2 = cv2.erode(w2, kernel1, iterations=4)
         edged = cv2.Canny(w2, 30, 200)
         cv2.waitKey(0)
         letter_image_regions = []
         contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
         for contour in contours:
             (x, y, w, h) = cv2.boundingRect(contour)
             if w / h > 1.25:
                 half_width = int(w / 2)
                 letter_image_regions.append((x, y, half_width, h))
                 letter_image_regions.append((x + half_width, y, half_width, h))
             else:
                 letter_image_regions.append((x, y, w, h))
                 letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

         for letter_bounding_box, letter_text in zip(letter_image_regions,captcha_image):
            (x, y, w, h) = letter_bounding_box
            letter_image = w2[y - 10:y + h + 12, x - 12:x + w + 12]
            # cv2.imshow(letter_text, letter_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            ewe.append(letter_text)
            wwe=0
            for eq in range(len(ewe)):
                 if(ewe[eq]==letter_text):
                     wwe=wwe+1
            qw = str(wwe)
            letter_image=cv2.resize(letter_image,(140,140))
            cv2.imwrite(pwd+'/data/'+letter_text+qw+'.jpg',letter_image )
            count = counts.get(letter_text, 1)
            counts[letter_text] = count + 1
pass


##################
#Dont change the code below
##################
tic = tm.perf_counter()
function()
toc = tm.perf_counter()

print(toc-tic)