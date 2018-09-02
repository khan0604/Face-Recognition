import cv2


def detectFaces(face_cascade, color_image, scaleFactor = 1.1):

    img_copy = color_image.copy()

    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return gray_img[y:y+w, x:x+h], faces[0]


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255, 0), 2)

