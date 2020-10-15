#!/bin/python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from solver import solve, print_board

def show(img):
    cv2.imshow('Image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('sudoku_puzzle.jpg')


def find_puzzle(image):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


    # loop over the contours
    puzzleContours = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            puzzleContours = approx
            break

    puzzle = four_point_transform(image, puzzleContours.reshape(4, 2))
    warped = four_point_transform(gray, puzzleContours.reshape(4, 2))

    return (puzzle, warped)


def get_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None

    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    return digit


(puzzle, warped) = find_puzzle(image)
model = load_model('digit_reconition.h5')

# show(warped)

# initialize our 9x9 sudoku board
board = np.zeros((9, 9), dtype="int")

# a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped imageprint(initial_board)

# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []

# loop over the grid locations
for y in range(0, 9):
    # initialize the current list of cell locations
    row = []

    for x in range(0, 9):
        # compute the starting and ending (x, y)-coordinates of the
        # current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then
        # extract the digit from the cell
        cell = warped[startY:endY, startX:endX]

        digit = get_digit(cell)

        if digit is not None:
            digit_preprocessed = cv2.resize(digit, (28, 28))
            digit_preprocessed = digit_preprocessed.astype("float") / 255.0
            digit_preprocessed = img_to_array(digit_preprocessed)
            digit_preprocessed = np.expand_dims(digit_preprocessed, axis=0)

            pred = model.predict(digit_preprocessed).argmax()
            board[y, x] = pred

    # add the row to our cell locations
    cellLocs.append(row)

for i in board:
    for y in i:
        if y == 0:
            print(y)

solve(board)


# loop over the cell locations and board
for (cellRow, boardRow) in zip(cellLocs, board):
	# loop over individual cell in the row
	for (box, digit) in zip(cellRow, boardRow):
		# unpack the cell coordinates
		startX, startY, endX, endY = box

		# compute the coordinates of where the digit will be drawn
		# on the output puzzle image
		textX = int((endX - startX) * 0.33)
		textY = int((endY - startY) * -0.3)
		textX += startX
		textY += endY

		# draw the result digit on the sudoku puzzle image
		cv2.putText(puzzle, str(digit), (textX, textY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# show the output image
cv2.imshow("Sudoku Result", puzzle)
cv2.waitKey(0)