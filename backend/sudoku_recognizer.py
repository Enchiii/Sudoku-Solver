import cv2
import operator
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import Net, InvertColors, RemoveAlphaChannel
from tk_vis import TkSudoku


def prepare_images(images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
    ])

    data = torch.stack([transform(img) for img in images])

    labels = torch.zeros(len(images), dtype=torch.long)

    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


def sudoku_recognizer(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    maxArea = 0

    margin = 10
    case = 28 + 2 * margin
    perspective_size = 9 * case

    flag = 0
    ans = 0

    # Find the largest contour(Sudoku Grid)
    for c in contours_:
        area = cv2.contourArea(c)

        peri = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, 0.01 * peri, True)

        if area > maxArea and len(polygon) == 4:
            contour = polygon
            maxArea = area

    if contour is not None:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        points = np.vstack(contour).squeeze()
        points = sorted(points, key=operator.itemgetter(1))

        if points[0][0] < points[1][0]:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1 = np.float32([points[1], points[0], points[2], points[3]])

        pts2 = np.float32([[0, 0], [perspective_size, 0], [0, perspective_size], [perspective_size, perspective_size]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_window = cv2.warpPerspective(image, matrix, (perspective_size, perspective_size))
        result = perspective_window.copy()

    p_window = cv2.cvtColor(perspective_window, cv2.COLOR_BGR2GRAY)
    p_window = cv2.GaussianBlur(p_window, (5, 5), 0)
    p_window = cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p_window = cv2.morphologyEx(p_window, cv2.MORPH_CLOSE, vertical_kernel)
    lines = cv2.HoughLinesP(p_window, 1, np.pi / 180, 120, minLineLength=40, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

    images = []
    for y in range(9):
        predicted_line = []
        for x in range(9):
            y2min = y * case + margin
            y2max = (y + 1) * case - margin
            x2min = x * case + margin
            x2max = (x + 1) * case - margin

            image = p_window[y2min:y2max, x2min:x2max]

            img = cv2.resize(image, (28, 28))
            images.append(img)

            pixel_sum = np.sum(img)
            pixels_sum = [pixel_sum]

    return images


PATH = 'models/m1.pth'
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

if __name__ == '__main__':
    image = cv2.imread("../test_images/sudoku1.png")
    images = sudoku_recognizer(image)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#')
    preds = []
    dLoader = prepare_images(images)
    with torch.no_grad():
        for i, _data in enumerate(dLoader, 0):
            inputs, _ = _data
            outputs = net(inputs)
            # print(i, " ", outputs)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted)


    board = []
    for y in range(9):
        row = []
        for x in range(9):
            row.append(classes[preds[y * 9 + x]])
        board.append(row)

    tk = TkSudoku()
    tk.generate_board(board)
    tk.main_loop()
