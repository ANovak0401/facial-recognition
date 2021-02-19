import os
import time
from datetime import datetime
from playsound import playsound
import pyttsx3
import cv2 as cv

engine = pyttsx3.init()  # initialise audio player
engine.setProperty('rate', 145)  # set playback speed
engine.setProperty('volume', 1.0)  # set volume

root_dir = os.path.abspath('.')  # root directory for sound effects
gunfire_path = os.path.join(root_dir, 'gunfire.wav')  # gunfire sound variable
tone_path = os.path.join(root_dir, 'tone.wav')  # all clear tone sound variable

path = 'C:/Users/austi/anaconda3/Lib/site-packages/cv2/data/'  # root directory pathway
face_cascade = cv.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')  # path for haarscade files
eye_cascade = cv.CascadeClassifier(path + 'haarcascade_eye.xml')  # path for eye detection haarscade files

os.chdir('corridor_5')  # corridor 5 directory
contents = sorted(os.listdir())  # sort contents

for image in contents:
    print(f'nMotion detected...{datetime.now()}')  # status update text
    discharge_weapon = True  # prepare weapon
    engine.say("You have entered an active fire zone. \
               Stop and face the gun immediately. \
               When you hear the tone, you have 5 seconds to comply.")  # warning message to user
    engine.runAndWait()  # wait for response
    time.sleep(3)  # wait 3 seconds
    img_gray = cv.imread(image, cv.IMREAD_GRAYSCALE)  # create grayscale image from "camera"
    height, width = img_gray.shape  # height and width for window
    cv.imshow(f'Motion detected {image}', img_gray)  # show image and name window
    cv.waitKey(2000)  # wait for 2 seconds
    cv.destroyWindow(f'Motion detected {image}')  # close window
    face_rect_list = [face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.1,
                                                    minNeighbors=5)]  # empty list for face rectangles
    # function call for facial recognition

    print(f"Searching {image} for eyes.")  # print status to terminal
    for rect in face_rect_list:
        for (x, y, w, h) in rect:  # tuple coords in image
            rect_for_eyes = img_gray[y:y+h, x:x+w]  # create rectangle size for each image
            eyes = eye_cascade.detectMultiScale(image=rect_for_eyes, scaleFactor=1.05, minNeighbors=2)  # eye cascade
            for (xe, ye, we, he) in eyes:  # go through tuple coords in eyes list
                print("Eyes detected.")  # status to terminal
                center = (int(xe + 0.5 * we), int(ye + 0.5 * he))  # center of circle over eyes
                radius = int((we + he) / 3)  # radius of circle
                cv.circle(rect_for_eyes, center, radius, 255, 2)  # draw circle over eyes detected
                cv.rectangle(img_gray, (x, y), (x+w, y+h), (255, 255, 255), 2)  # draw rectangle around eye circles
                discharge_weapon = False  # turn on weapon safety
                break

    if discharge_weapon == False:
        playsound(tone_path, block=False)  # play all clear sound
        cv.imshow('Detected faces', img_gray)  # show detected faces image
        cv.waitKey(2000)  # wait 2 seconds
        cv.destroyWindow('Detected faces')  # close window
        time.sleep(5)  # sleep 5 seconds

    else:
        print(f"No face in {image}. Discharging weapon!")  # print message to terminal
        cv.putText(img_gray, 'Fire!', (int(width / 2) - 20, int(height / 2)), cv.FONT_HERSHEY_PLAIN, 3, 255, 3)
        # print fire to screen
        playsound(gunfire_path, block=False)  # play gunfire sound
        cv.imshow('Mutant', img_gray)  # show mutant image
        cv.waitKey(2000)  # wait 2 seconds
        cv.destroyWindow('Mutant')  # close window
        time.sleep(3)  # wait 3 seconds

engine.stop()  # close the sound player engine
