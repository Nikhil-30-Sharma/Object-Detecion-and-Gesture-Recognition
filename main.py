import cv2
import time
import os
import HandTrackingModule as htm
thres = 0.5 # Threshold to detect object
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
# cap.set(10,70)

folderPath = "Finger_Images"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0
detector = htm.handDetector(detectionCon=0.6)
tipIds = [4,8,12,16,20]


classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox,lmList)

    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(0)
        else:
            fingers.append(1)

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            
        totalFingers = fingers.count(1)
        print(totalFingers)

        h,w,c=overlayList[1].shape
        # img[h,w] = overlayList[totalFingers-1]

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)


    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,0),3)
    cv2.imshow("Output",img)
    cv2.waitKey(1)
