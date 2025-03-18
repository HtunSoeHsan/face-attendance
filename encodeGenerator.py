import os

import cv2
import pickle
import face_recognition

# importing student images
folderPath = "images"
pathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    print(path)
    print(os.path.splitext(path)[0])
print(studentIds)

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return  encodeList
print("Encoding started...")
encodeListKnow = findEncodings(imgList)
encodeListKnowWithIds = [encodeListKnow, studentIds]
print("Encoding end...")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnowWithIds, file)
file.close()
