import cv2
import os
import csv
import numpy as np
import LDRP
import facedetect
from operator import itemgetter

csv_filename = "data_sheet_stud.csv"
wrong_det = 0
haar_face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_alt2.xml')


def euclidian_distance(first_instance, second_instance):
    distance = 0.0
    for index in range(1024):
        distance += np.sqrt(pow(float(first_instance[index]) - float(second_instance[index]), 2))

    return distance


def find_neighbors(feature_vector, K):
    search_file = open(csv_filename)
    distances = []
    csv_reader = csv.reader(search_file)
    i = 1
    for row in list(csv_reader):
        tup = (euclidian_distance(row, feature_vector), row[1024])
        i += 1
        distances.append(tup)

    distances = sorted(distances, key=itemgetter(0))

    return distances[:K]


def find_votes(final_list):
    dic = {}
    for item in final_list:
        if item[1] in dic:
            dic[item[1]] += 1
        else:
            dic[item[1]] = 1

    max_vote = 0
    result = " "
    for key in dic:
        if dic[key] > max_vote:
            max_vote = dic[key]
            result = key

    return result


def main(testing_dir_path):
    pos = 0
    tot = 0
    subject_list = os.listdir(testing_dir_path)

    for sub in subject_list:
        if sub[0] != 's':
            continue

        subject_dir_path = testing_dir_path + '/' + sub
        sub_images = os.listdir(subject_dir_path)

        for image in sub_images:
            if image.startswith('.'):
                continue

            img_path = subject_dir_path + '/' + image
            in_img = cv2.imread(img_path)
            input_img = cv2.resize(in_img, None, fx=1/12, fy=1/12, interpolation=cv2.INTER_AREA)
            face, rec = facedetect.detectFaces(haar_face_cascade, input_img)

            if face.all() == None:
                print("Face not detected : ", img_path)
                continue

            (x, y, w, h) = rec
            #gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
            input_feature = LDRP.ldrp(face)

            finalk_list = find_neighbors(input_feature, 3)
            print(finalk_list)
            ans = find_votes(finalk_list)
            print("Hey there!, sub: " + ans + " ...!")
            if float(ans) == float(sub[1:]):
                pos += 1
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                if ans == '2.0':
                    facedetect.draw_text(input_img, 'Harshul', x, y)
                elif ans == '3.0':
                    facedetect.draw_text(input_img, 'Sohail', x, y)
                elif ans == '1.0':
                    facedetect.draw_text(input_img, 'Ruturaj', x, y)
                elif ans == '4.0':
                    facedetect.draw_text(input_img, 'Sashank', x, y)
                else:
                    facedetect.draw_text(input_img, 'Lakshay', x, y)

            else:
                print(img_path)
                cv2.imshow('wrong match', input_img)
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            tot += 1
            cv2.imshow("input_image", input_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
    print(pos)
    print(tot)
    print(pos / tot * 100)


main('IITJ_students/testing_data')


