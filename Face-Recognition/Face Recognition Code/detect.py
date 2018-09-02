import os
import cv2

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_alt2.xml')
train_dir = 'Ruturaj/testing_data'

# use haarcascade_frontalface_alt2 and scaleFactor=1.01 for AT_T

def detectFaces(face_cascade, color_image, scaleFactor=1.1):
    img_copy = color_image.copy()
    res_img = cv2.resize(img_copy,None,fx=1/12, fy=1/12, interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighbors=5)


    if len(faces) == 0:
        cv2.imshow('img', img_copy)
        cv2.waitKey(10)
        return False

    (x, y, w, h) = faces[0]
    cv2.rectangle(res_img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('img', res_img)
    cv2.waitKey(10)
    return True

def detection(dir_path):
    total_det = 0
    total_img = 0
    subjects = os.listdir(dir_path)

    for sub in subjects:
        if not sub[0] == 's':
            continue

        label = sub[1:]

        sub_dir_path = dir_path + '/' + sub

        sub_images = os.listdir(sub_dir_path)

        for images in sub_images:
            if images[0] == '.':
                continue

            img_path = sub_dir_path + '/' + images
            training_img = cv2.imread(img_path)

            if detectFaces(face_cascade, training_img):
                total_det+=1
            else:
                print(img_path)
                cv2.imshow('no det', training_img)
                cv2.waitKey(100)
            total_img += 1
    return total_det, total_img

(x, y) = detection(train_dir)

print(x)
print(y)