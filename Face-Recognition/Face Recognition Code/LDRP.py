import numpy as np


N = 8
B = 8


def ldrp_helper(img_part, M, x, y):

    mat = []
    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x][y+i] <= img_part[x][y+j]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x-i][y+i] <= img_part[x-j][y+j]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x-i][y] <= img_part[x-j][y]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x-i][y-i] <= img_part[x-j][y-j]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x][y-i] <= img_part[x][y-j]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x+i][y-i] <= img_part[x+j][y-j]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x+i][y] <= img_part[x+j][y]:
                temp += con
            con *= 2
    mat.append(temp)

    con = 1
    temp = 0
    for i in range(1, M):
        for j in range(i+1, M+1):
            if img_part[x+i][y+i] <= img_part[x+j][y+j]:
                temp += con
            con *= 2
    mat.append(temp)

    mew = M*(M-1)/2

    central_val = img_part[x][y]*(pow(2, mew)-1)/(pow(2, B)-1)

    con = 1
    temp = 0
    for i in range(N):
        if central_val <= mat[i]:
            temp += con
        con *= 2

    return temp


def ldrp(face_img):

    height, width = face_img.shape[:2]
    ldrp_vec = []
    img_copy1 = face_img.copy()

    for M in range(3, 7):
        new_ldrp_vec = [0]*256

        for i in range(M, width-M):

            for j in range(M, height-M):
                new_ldrp_vec[ldrp_helper(img_copy1, M, j, i)] += 1  # important change ( interchanging i and j)

        ldrp_vec.extend(new_ldrp_vec)
    mat_sum = np.sum(ldrp_vec)
    ldrp_vec = ldrp_vec/mat_sum

    return ldrp_vec

