import cv2
import LDRP
import facedetect
import csv
import numpy as np
import os

subject = [""]
number_of_subjects = 5
csv_filename = "data_sheet_stud3.csv"
identity = [""]
wrong_det = 0
haar_face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_alt2.xml')

def create_sub(number_of_sub):

    for i in range(1, number_of_sub+1):
        subject.append("sub" + str(i))


def prepare_training_data(data_folder_path):
    create_sub(number_of_subjects)
    dirs = os.listdir(data_folder_path)
    data_file = open(csv_filename, 'w', newline='')

    for dir_name in dirs:
        if not dir_name[0] == 's':
            continue

        label = dir_name[1:]
        subject_dir_path = data_folder_path + '/' + dir_name

        subject_images = os.listdir(subject_dir_path)

        for images in subject_images:
            if images[0] == '.':
                continue

            image_path = subject_dir_path + '/' + images
            training_img = cv2.imread(image_path)
            res_img = cv2.resize(training_img, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
            cv2.imshow("training images....", res_img)
            cv2.waitKey(100)

            #gray_img = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)  # cv2 reads and converts image to BGR format

            roi, rec = facedetect.detectFaces(haar_face_cascade, res_img)

            if roi.any() is None:
                continue

            feature_vector = LDRP.ldrp(roi)
            feature_vector = np.append(feature_vector, [int(label)])

            csv_writer = csv.writer(data_file)
            csv_writer.writerow(feature_vector)
            identity.append(subject[int(label)])
    print(identity)
    data_file.close()


prepare_training_data('Ruturaj/training_data')