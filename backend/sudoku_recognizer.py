import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from backend.model import Net
import operator
import torch
from tk_vis import TkSudoku


def prepare_imgages(images):
    data = images
    data = np.expand_dims(data, axis=1)

    transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float32),  # Convert NumPy array to tensor
        transforms.Resize((16, 16))
    ])

    # Apply transform to dataset
    data = torch.stack([transform(img) for img in data])  # Convert each image
    labels = torch.tensor([0 for _ in range(len(images))], dtype=torch.long)
    dataset = TensorDataset(data, labels)

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


image = cv2.imread(r"images/sudoku1.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = None
maxArea = 0

margin = 10
case = 28 + 2*margin
perspective_size = 9*case

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
lines = cv2.HoughLinesP(p_window, 1, np.pi/180, 120, minLineLength=40, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

images = []
for y in range(9):
    predicted_line = []
    for x in range(9):
        y2min = y*case+margin
        y2max = (y+1)*case-margin
        x2min = x*case+margin
        x2max = (x+1)*case-margin

        image = p_window[y2min:y2max, x2min:x2max]

        images.append(image)

        img = cv2.resize(image, (16, 16))
        img = img.reshape((1, 16, 16, 1))

        pixel_sum = np.sum(img)
        pixels_sum = [pixel_sum]

PATH = 'models/m1.pth'
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

if __name__ == '__main__':
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#')
    preds = []
    dLoader = prepare_imgages(images)
    with torch.no_grad():
        for i, _data in enumerate(dLoader, 0):
            inputs, _ = _data
            outputs = net(inputs)
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
