
import json
import boto3
import cv2
import math
import io

def analyzeVideo():
    videoFile = "cavity.mp4"
    projectVersionArn = "arn:aws:rekognition:us-east-2:695748283885:project/testTeeth/version/testTeeth.2021-06-01T14.55.16/1622577316603"

    rekognition = boto3.client('rekognition', region_name = 'us-east-2')        
    customLabels = ['hello']    
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        print("Processing frame id: {}".format(frameId))
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            hasFrame, imageBytes = cv2.imencode(".jpg", frame)

            if(hasFrame):
                response = rekognition.detect_custom_labels(
                    Image={
                        'Bytes': imageBytes.tobytes(),
                    },
                    ProjectVersionArn = projectVersionArn
                )
            
            for elabel in response["CustomLabels"]:
                elabel["Timestamp"] = (frameId/frameRate)*1000
                customLabels.append(elabel)
    
    print(customLabels)

    with open(videoFile + ".json", "w") as f:
        f.write(json.dumps(customLabels)) 

    cap.release()

analyzeVideo()