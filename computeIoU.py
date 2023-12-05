import numpy as np
import torch
import math


def computeIntersection(rect1, rect2):
    x_inter1 = max(rect1[0], rect2[0])
    y_inter1 = max(rect1[1], rect2[1])
    x_inter2 = min(rect1[2], rect2[2])
    y_inter2 = min(rect1[3], rect2[3])

    return (x_inter2 - x_inter1) * (y_inter2 - y_inter1)


def computeUion(rect1, rect2):
    return (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) \
           - computeIntersection(rect1, rect2)


def computeIoU(rect1, rect2):
    intersect_area = computeIntersection(rect1, rect2)
    uion_area = computeUion(rect1, rect2)

    IoU = intersect_area / (uion_area + 1e-5)

    return IoU


def computeGIoU(rect1, rect2):
    IoU = computeIoU(rect1, rect2)
    x1_min = min(rect1[0], rect2[0])
    y1_min = min(rect1[1], rect2[0])

    x2_max = max(rect1[2], rect2[2])
    y2_max = max(rect1[3], rect2[3])

    all_area = abs(x2_max - x1_min) * abs(y2_max - y1_min)

    return IoU - (abs(all_area - computeUion(rect1, rect2)) / all_area)


def compute_distance(point1, point2):
    return (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2  # no sqrt


def centerpoint_distance(rect1, rect2):
    centerpoint1 = [int((rect1[0] + rect1[2])) / 2, int((rect1[1] + rect1[3])) / 2]
    centerpoint2 = [int((rect2[0] + rect2[2])) / 2, int((rect2[1] + rect2[3])) / 2]
    return compute_distance(centerpoint1, centerpoint2)


def diagonal_distance(point1, point2):
    return compute_distance(point1, point2)


def computeDIoU(rect1, rect2):
    IoU = computeIoU(rect1, rect2)
    d = centerpoint_distance(rect1, rect2)

    x1_min = min(rect1[0], rect2[0])
    y1_min = min(rect1[1], rect2[0])

    x2_max = max(rect1[2], rect2[2])
    y2_max = max(rect1[3], rect2[3])

    c = diagonal_distance([x1_min, y1_min], [x2_max, y2_max])

    return IoU - d / c


def computeCIoU(rect1, rect2):
    IoU = computeIoU(rect1, rect2)

    w1 = abs(rect1[2] - rect1[0])
    h1 = abs(rect1[3] - rect1[1])

    w2 = abs(rect2[2] - rect2[0])
    h2 = abs(rect2[3] - rect2[1])

    v = (4 / math.pi) * np.power(math.atan(w1 / h1) - math.atan(w2 / h2))

    alpha = v / ((1 - IoU) + v)
    return computeDIoU(rect1, rect2) - alpha * v


def computeEIoU(rect1, rect2):
    DIoU = computeDIoU(rect1, rect2)

    x1_min = min(rect1[0], rect2[0])
    y1_min = min(rect1[1], rect2[0])

    x2_max = max(rect1[2], rect2[2])
    y2_max = max(rect1[3], rect2[3])

    cw = x2_max - x1_min
    ch = y2_max - y1_min

    w1 = abs(rect1[2] - rect1[0])
    h1 = abs(rect1[3] - rect1[1])

    w2 = abs(rect2[2] - rect2[0])
    h2 = abs(rect2[3] - rect2[1])

    w = (w2 - w1) ** 2
    h = (h2 - h1) ** 2

    return DIoU - w / (cw ** 2) - h / (ch ** 2)


if __name__ == '__main__':
    r2 = [0, 0, 6, 8]  # x1,y1,  x2,y2
    r1 = [3, 2, 9, 10]  # x3,y3,  x4, y4
    print(computeIoU(r1, r2))
    print(computeGIoU(r1, r2))
