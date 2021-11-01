import cv2 as cv
import numpy as np

def detectPlate(path):
    img = cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    threshold = 20

    while True:
        ret, thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

        erode = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations = 1 if threshold < 40 else  3 )

        contours, hierarchy = cv.findContours(erode, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        parent_index = np.bincount(np.array([element[3] for element in hierarchy[0] if element[3] >= 0])).argmax()

        symbol_contours = []
        for idx, contour in enumerate(contours):
            if hierarchy[0][idx][3] == parent_index:
                symbol_contours.append(contour)

        if len(symbol_contours) <= 11 and len(symbol_contours) > 6:
            break
        
        threshold += 3

    detection = img.copy()

    heights = [cv.boundingRect(contour)[3] for contour in symbol_contours]
    widhts = [cv.boundingRect(contour)[2] for contour in symbol_contours]
    mean_height = np.mean(heights)
    mean_width = np.mean(widhts)

    letters = []

    for contour in symbol_contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if mean_height / h > 1.2 or w / mean_width > 2:
            continue

        cv.rectangle(detection, (x, y), (x + w, y + h), (0, 255, 0), 2)

        letter_crop = gray[y:y + h, x:x + w]

        size_max = max(w, h)

        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

        if w > h:
            y_pos = size_max//2 - h//2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            x_pos = size_max//2 - w//2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop

        letters.append([x, cv.resize(letter_square, (28, 28), interpolation=cv.INTER_AREA)])

    letters.sort(key=lambda x: x[0], reverse=False)

    cv.imshow('detection', detection)

    return letters
