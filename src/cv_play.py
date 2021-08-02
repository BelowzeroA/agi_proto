from typing import List, Tuple
import copy

import cv2 as cv
import numpy as np
from skimage.measure import approximate_polygon
from sympy import Segment, Point, Line, N, acos
import shapely
from shapely import geometry


ALPHA = 180 / np.arccos(-1)


def does_point_inside_polygon(list_of_obj_points, point):
    """
    :param list_of_obj_points: список точек полигона
    :param point: точка на плоскости
    :return: bool, содержится ли точка в полигоне
    """
    polygon = shapely.geometry.polygon.Polygon(list_of_obj_points)
    point = shapely.geometry.Point(point)
    return polygon.contains(point)


def mean(lst: list) -> float:
    return sum(lst) / len(lst)


def segments_have_common_point(s1: Segment, s2: Segment) -> bool:
    return s1.p1 == s2.p1 or s1.p1 == s2.p2 or s1.p2 == s2.p1 or s1.p2 == s2.p2


def get_segment_midpoint(segment: Segment) -> Point:
    x = (segment.p1[0] + segment.p2[0]) // 2
    y = (segment.p1[1] + segment.p2[1]) // 2
    return Point(x, y)


def segments_in_group(s1: Segment, s2: Segment, group: List[Segment]) -> Segment:
    segments = [s1, s2]
    if s1 in group:
        segments.remove(s1)
    if s2 in group:
        segments.remove(s2)
    if len(segments) < 2:
        return segments
    return None


def collect_adjacent_segment_groups(segments: List[Segment], threshold: float) -> List[Tuple[Segment, Segment]]:
    adjacent_segments_groups = []
    short_segments = [segment for segment in segments if N(segment.length) < threshold]
    singles = list(short_segments)
    for i in range(len(short_segments)):
        for j in range(len(short_segments)):
            if i == j:
                continue
            s1 = short_segments[i]
            s2 = short_segments[j]
            if segments_have_common_point(s1, s2):
                if s1 in singles:
                    singles.remove(s1)
                if s2 in singles:
                    singles.remove(s2)
                found = False
                for group in adjacent_segments_groups:
                    segment_to_add = segments_in_group(s1, s2, group)
                    if segment_to_add is not None:
                        group.extend(segment_to_add)
                        found = True
                if not found:
                    adjacent_segments_groups.append([s1, s2])
    return adjacent_segments_groups


def collect_segments_lengths(contour):
    segments = []
    lengths = []
    points = []
    for i in range(len(contour)):
        point1 = contour[i]
        point2 = contour[i + 1] if i < len(contour) - 1 else contour[0]
        p1 = Point(point1[0], point1[1])
        p2 = Point(point2[0], point2[1])
        segment = Segment(p1, p2)
        if p1 not in points:
            points.append(p1)
        if p2 not in points:
            points.append(p2)
        segments.append(segment)
        lengths.append(N(segment.length))
    return segments, points, lengths


def find_splitting_threshold(lengths: List[float]) -> float:
    lengths = sorted(lengths)
    threshold = 0
    for i in range(len(lengths) - 1):
        length = lengths[i]
        if length < 3:
            continue
        next_length = lengths[i + 1]
        if next_length / length > 4:
            threshold_index = i
            threshold = next_length
            break
    return threshold


def compute_roi_center(group: List[Segment]) -> List[Point]:
    buffer = list(group)
    while True:
        if len(buffer) == 1:
            midpoint = get_segment_midpoint(buffer[0])
            return midpoint
        points = []
        for segment in buffer:
            midpoint = get_segment_midpoint(segment)
            points.append(midpoint)

        buffer = []
        while len(points) > 1:
            point1 = points.pop(0)
            point2 = points.pop(0)
            buffer.append(Segment(point1, point2))

        if len(points) == 1:
            buffer.append(Segment(point2, points[0]))


def find_roi(contour):
    roi_centers = []
    # Extract point, segments and their lengths
    segments, points, lengths = collect_segments_lengths(contour)

    # find a threshold which distinguishes short segments from long ones
    threshold = find_splitting_threshold(lengths)

    if threshold > 0:
        # collect groups of short segments
        adjacent_segments_groups = collect_adjacent_segment_groups(segments, threshold)

        # remove the points that don't belong to the groups extracted above
        for group in adjacent_segments_groups:
            for segment in group:
                if segment.p1 in points:
                    del points[points.index(segment.p1)]
                if segment.p2 in points:
                    del points[points.index(segment.p2)]

        # compute a ROI center for each segment group
        for group in adjacent_segments_groups:
            roi_center = compute_roi_center(group)
            roi_centers.append(roi_center)

    # alternatively, the ROI centers are the points that don't belong to any group (standalone points)
    roi_centers.extend(points)

    return roi_centers


def find_nozero_ind(our_screen):
    print(type(our_screen))
    res = []
    for i in range(our_screen.shape[0]):
        for j in range(our_screen.shape[1]):
            if our_screen[i][j] != 0:
                res.append((i, j))
    return res


def point_orientation(vec_point_1, vec_point_2, point):
    '''
    :param vec_point_1: начальная точка вектора
    :param vec_point_2: конечная точка вектора
    :param point: некоторая точка
    :return: скаляр. Если > 0, то лежит справа от прямой (если смотреть от начальной точки
    вектора в сторону конечной точки вектора)
    Если < 0, то слева от прямой
    Если = 0, то на прямой.
    '''
    return ((point[0] - vec_point_1[0]) * (vec_point_2[1] - point[1]) -
            (point[1] - vec_point_1[1]) * (vec_point_2[0] - point[0])
    )


def what_quadrant(roi, p1, p2):
    dx_1 = roi[0] - p1[0]
    dy_1 = roi[1] - p1[1]
    dx_2 = roi[0] - p2[0]
    dy_2 = roi[1] - p2[1]
    if dx_1 + dx_2 == 1 or dy_1 + dy_2 == 1:
        # точки в разных квадрантах
        return 5
    elif dx_1 > 0:
        if dy_1 > 0:
            return 1
        else:
            return 2
    elif dy_1 > 0:
        return 0
    else:
        return 3


def get_mass(value):
    if value >= 0.4:
        return 2
    return 1


def get_corner(p1, p2):
    if p1[1] > p2[1]:
        p1, p2 = p2, p1
    a = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
    b = np.array([5, 0])
    return round(ALPHA * np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def quadrant_roi_analysis(roi, approx_contour, contour, quadrant_size, img):
    quadrant_norm = 1.42 * quadrant_size
    result = {}
    result['roi'] = roi
    result['quadrants'] = [[], [], [], []]
    polygon = geometry.polygon.Polygon(approx_contour)
    x_left = roi[0] - quadrant_size
    x_right = roi[0] + quadrant_size - 1
    y_top = roi[1] + quadrant_size - 1
    y_bottom = roi[1] - quadrant_size
    approx_contour_points_into_square = []
    res = []
    segmetnts = []
    for point in approx_contour:
        if (x_left <= point[0] and point[0] <= x_right and y_bottom <= point[1] and point[1] <= y_top):
            approx_contour_points_into_square.append(point)
    for point in contour:
        if (x_left <= point[0][0] and point[0][0] <= x_right and y_bottom <= point[0][1] and point[0][1] <= y_top):
            if point[0][0] == x_left or point[0][0] == x_right or point[0][1] == y_top or point[0][1] == y_bottom:
                res.append(point[0])
                segmetnts.append(point[0])
            elif (point[0][0] == roi[0] or point[0][0] == roi[0] - 1 or point[0][1] == roi[1] or point[0][1] == roi[1] - 1):
                if len(res) != 0:
                    if point[0][0] != res[-1][0] and point[0][1] != res[-1][1]:
                        res.append(point[0])
                else:
                    res.append(point[0])
            elif point in approx_contour_points_into_square:
                res.append(point[0])

    for ind in range(1, len(res)):
        quadrant_number = what_quadrant(roi, res[ind - 1], res[ind])
        if quadrant_number == 5:
            continue
        our_line = {}
        our_line['angle'] = get_corner(res[ind - 1], res[ind])
        our_line['mass'] = get_mass(np.linalg.norm([res[ind][0] - res[ind - 1][0],
                                                    res[ind][1] - res[ind - 1][1]]) / quadrant_norm)
        result['quadrants'][quadrant_number].append(our_line)

    # find color
    for ind in range(1, len(segmetnts)):
        u = segmetnts[ind - 1]
        v = segmetnts[ind]
        if u[0] == v[0]:
            try_point = shapely.geometry.Point(u[0], round((u[1] + v[1]) / 2))
            if polygon.contains(try_point):
                our_color_point = u[0], round((u[1] + v[1]) / 2)
                break
        elif u[1] == v[1]:
            try_point = shapely.geometry.Point(round((u[0] + v[0]) / 2), u[1])
            if polygon.contains(try_point):
                our_color_point = round((u[0] + v[0]) / 2), u[1]
                break
        else:
            if polygon.contains(shapely.geometry.Point(u[0], v[1])):
                our_color_point = u[0], v[1]
                break
            elif polygon.contains(shapely.geometry.Point(v[0], u[1])):
                our_color_point = v[0], u[1]
                break
    result['color'] = img[our_color_point[1], our_color_point[0]]
    return result


# def filter(points_list, threshold):
#    res = []
#    len_lp = len(points_list)
#    for i in range(len_lp):
#        for j in range(i + 1, len_lp):
#            res.append(((points_list[i][0] - points_list[j][0]) ** 2 +
#                       (points_list[i][1] - points_list[j][1]) ** 2) ** (1/2))
#    min_dist = min(res)
#    for ind in range(len(res)):
#        if ind / min_dist > threshold:

def into_segment(list_x, list_y, p1, p2):
    x = p1[0]
    y = p2[0]
    if x > y:
        x, y = y, x
    res_x = []
    res_y = []
    res_x.append(x)
    for elem in list_x:
        if x < elem and elem < y:
            res_x.append(elem)
    res_x.append(y)

    x = p1[1]
    y = p2[1]
    if x > y:
        x, y = y, x
    res_y.append(x)
    for elem in list_y:
        if x < elem and elem < y:
            res_y.append(elem)
    res_y.append(y)
    return res_x, res_y


def into_roi_square(point, x_left, x_right, y_bottom, y_top):
    return x_left <= point[0] and point[0] <= x_right and y_bottom <= point[1] and point[1] <= y_top


def is_three_points_belong_to_the_same_shape(p1, p2, p3, img):
    x_mean = (p1 + p2 + p3) / 3
    if int(x_mean[1]) == 440:
        return False
    elif np.any(img[int(x_mean[1]), int(x_mean[0])] != np.array([0, 0, 0], dtype=np.int8)):
        return img[int(x_mean[1]), int(x_mean[0])]
    else:
        return False


def is_color(all_shapes, color):
    for ind in range(len(all_shapes)):
        if np.all(all_shapes[ind]['color'] == color):
            return ind
    return -1


def split_shapes(approx_contour, img):
    all_shapes = []
    quantity_points = len(approx_contour[:-1])
    for ind in range(len(approx_contour[:-1])):
        p1 = approx_contour[ind]
        p2 = approx_contour[(ind + 1) % quantity_points]
        p3 = approx_contour[(ind + 2) % quantity_points]
        triple_res = is_three_points_belong_to_the_same_shape(p1, p2, p3, img)
        if np.any(triple_res != False):
            ind_color = is_color(all_shapes, triple_res)
            if ind_color != -1:
                for point in (p1, p2, p3):
                    if list(point) not in [list(i) for i in all_shapes[ind_color]['points']]:
                        all_shapes[ind_color]['points'].append(point)
            else:
                temp_dict = {}
                temp_dict['color'] = triple_res
                temp_dict['points'] = [p1, p2, p3]
                all_shapes.append(temp_dict)
    return all_shapes


def quadrant_roi_analysis(roi, approx_contour, contour, quadrant_size, img):
    quadrant_norm = 1.42 * quadrant_size
    result = {}
    result['roi'] = roi
    result['quadrants'] = [[], [], [], []]
    polygon = geometry.polygon.Polygon(approx_contour)
    x_left = roi[0] - quadrant_size
    x_right = roi[0] + quadrant_size - 1
    y_top = roi[1] + quadrant_size - 1
    y_bottom = roi[1] - quadrant_size
    approx_contour_points_into_square = []
    list_x = sorted([x_left, x_right, roi[0] - 1, roi[0]])
    list_y = sorted([y_bottom, y_top, roi[1] - 1, roi[1]])
    res = []
    #segmetnts = []
    #for point in approx_contour[:-1]:
    #    if (x_left <= point[0] and point[0] <= x_right and y_bottom <= point[1] and point[1] <= y_top):
    #        approx_contour_points_into_square.append(point)
    a = list(approx_contour[0])
    #approx_contour = list(approx_contour[:-1])
    approx_contour = list(approx_contour)
    approx_contour.append(a)
    for ind in range(1, len(approx_contour)):
        min_res = []
        p1_flag = into_roi_square(approx_contour[ind - 1], x_left, x_right, y_bottom, y_top)
        p2_flag = into_roi_square(approx_contour[ind], x_left, x_right, y_bottom, y_top)
        if p1_flag or p2_flag:
            if p1_flag:
                min_res.append(tuple(approx_contour[ind - 1]))
            if p2_flag:
                min_res.append(tuple(approx_contour[ind]))
            approx_list_x, approx_list_y = into_segment(list_x, list_y, approx_contour[ind - 1], approx_contour[ind])
            if approx_list_x[0] == approx_contour[ind][0]:
                start_point = approx_contour[ind]
                end_point = approx_contour[ind - 1]
            else:
                start_point = approx_contour[ind - 1]
                end_point = approx_contour[ind]
            for x in approx_list_x[1:-1]:
                dx = x - start_point[0]
                y = (dx / (end_point[0] - start_point[0])) * (end_point[1] - start_point[1]) + start_point[1]
                min_res.append((x, round(y)))
            for y in approx_list_y[1:-1]:
                dy = y - start_point[1]
                x = (dy / (end_point[1] - start_point[1])) * (end_point[0] - start_point[0]) + start_point[0]
                min_res.append((round(x), y))
            min_res.sort()
            res.extend(min_res)
            res.append(0)
    for ind in range(1, len(res)):
        if res[ind] == 0 or res[ind - 1] == 0:
            continue
        quadrant_number = what_quadrant(roi, res[ind - 1], res[ind])
        if quadrant_number == 5:
            continue
        if res[ind] == res[ind - 1]:
            continue
        our_line = {}
        our_line['angle'] = get_corner(res[ind - 1], res[ind])
        our_line['mass'] = get_mass(np.linalg.norm(np.array([res[ind][0] - res[ind - 1][0],
                                                    res[ind][1] - res[ind - 1][1]], dtype=np.float32)) / quadrant_norm)
        result['quadrants'][quadrant_number].append(our_line)
    # for color
    for ind in range(1, len(res)):
        if res[ind] == 0 or res[ind - 1] == 0:
            continue
        if True:
            x_mean = int((res[ind][0] + res[ind - 1][0]) / 2)
            y_mean = round((res[ind][1] + res[ind - 1][1]) / 2)
            try_point = shapely.geometry.Point(x_mean + 3, y_mean)
            if polygon.contains(try_point):
                result['color'] = img[y_mean, x_mean + 3]
                break
            try_point = shapely.geometry.Point(x_mean - 3, y_mean)
            if polygon.contains(try_point):
                result['color'] = img[y_mean, x_mean - 3]
                break
    return result


def main():
    filename = 'pics/ScreenShot2.png'
    img = cv.imread(filename)
    img_gray = cv.imread(filename, 0)

    ret, thresh = cv.threshold(img_gray, 0, 200, 0)

    # Compute contours of the shapes
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0
    for i in range(len(contours)):
        arrow_contour = np.squeeze(contours[i])
        # polygonize the rounded square
        approx_contour = approximate_polygon(arrow_contour, tolerance=2.5)
        #print(approx_contour)

        # compute the ROI centers
        roi = find_roi(approx_contour)

        # for point in roi:
        # Big red dots - centers of ROI
        #    cv.circle(img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)

        for point in approx_contour:
            # small black dots - points after polygonizing
            cv.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)
        # cv.drawContours(img, [approx_arrow_contour], -1, (0, 255, 0), 3)

    cv.imwrite('contoured.jpg', img)

    # cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def main_3(filename=None, agent=None):
    if not filename:
        filename = 'pics/test_picture_3.JPG'
    img = cv.imread(filename)
    # print(find_nozero_ind(img))
    img_gray = cv.imread(filename, 0)
    # cv.imshow('Image3', img_gray)

    # ret, thresh = cv.threshold(img_gray, 230, 200, 0)
    ret, thresh = cv.threshold(img_gray, 0, 200, 0)
    # ret, thresh = cv.threshold(img_gray, 0, 0, 0)
    # Compute contours of the shapes
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    data = []
    if len(contours) != 0:
        for ind in range(len(contours)):
            obj_data = []
            #if ind in [1, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]:
            #    continue
            arrow_contour = np.squeeze(contours[ind])
            # polygonize the rounded square
            approx_contours = approximate_polygon(arrow_contour, tolerance=2.5)
            line = split_shapes(approx_contours, img)
            for approx_contour in line:
                obj_data = []
                approx_contour = approx_contour['points']
                # print('approx_contour', approx_contour)
                # compute the ROI centers
                if len(approx_contour) <= 2:
                    continue
                roi = find_roi(approx_contour)
                """
                cv.imshow('roi0', cv.resize(img_gray[(roi[0][1]) - 20:roi[0][1] + 20,
                                              roi[0][0] - 20: roi[0][0] + 20,
                                              ], (120, 120), cv.INTER_NEAREST))
                cv.imshow('roi1', cv.resize(img_gray[(roi[1][1]) - 20:roi[1][1] + 20,
                                              roi[1][0] - 20: roi[1][0] + 20,
                                              ], (120, 120), cv.INTER_NEAREST))
                cv.imshow('roi2', cv.resize(img_gray[(roi[2][1]) - 20:roi[2][1] + 20,
                                              roi[2][0] - 20: roi[2][0] + 20,
                                              ], (120, 120), cv.INTER_NEAREST))
                cv.imshow('roi3', cv.resize(img_gray[(roi[3][1]) - 20:roi[3][1] + 20,
                                              roi[3][0] - 20: roi[3][0] + 20,
                                              ], (120, 120), cv.INTER_NEAREST))
                """
                x_mean = 0
                y_mean = 0
                for point in roi:
                    x_mean += point.args[0]
                    y_mean += point.args[1]
                    obj_data.append(
                        quadrant_roi_analysis(point, approx_contour, contours[1], 20, img)
                    )
                    cv.circle(img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)

                for point in approx_contour:
                    # small black dots - points after polygonizing
                    cv.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)
                cv.imshow('Image', img)
                temp_dict = {}
                temp_dict['center'] = (int(round(x_mean / len(roi))), int(round(y_mean / len(roi))))
                obj_data.append(temp_dict)
                data.append(obj_data)
    agent.env_step(data)

    cv.waitKey(0)
    cv.destroyAllWindows()

def main_2(img):
    # a = find_nozero_ind(img)
    # print(a)
    # print(len(a))
    print(img.shape)
    print(type(img))
    # img = cv.imread(filename)
    # print(type(img))
    # print(img)
    # img_gray = cv.imread(filename, 0)
    # img = np.asarray(img, dtype="int64")
    img = np.asarray(img, dtype="uint8")
    img_gray = copy.deepcopy(img)
    print(img_gray.shape)
    print(type(img_gray))
    print(type(img_gray[0][0]))
    print(img_gray[0][0])
    # img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('1.jpg', img)

    ret, thresh = cv.threshold(img_gray, 230, 200, 0)
    # ret, thresh = cv.threshold(img_gray, 0, 0, 0)
    # Compute contours of the shapes
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    arrow_contour = np.squeeze(contours[8])

    # polygonize the rounded square
    approx_contour = approximate_polygon(arrow_contour, tolerance=2.5)

    # compute the ROI centers
    roi = find_roi(approx_contour)

    # for point in roi:
    #    # Big red dots - centers of ROI
    #    print(img)
    #    print(img.shape)
    #    print(type(img[0][0]))
    #    cv.circle(np.float32(img), (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)

    # for point in approx_contour:
    #    # small black dots - points after polygonizing
    #    cv.circle(img, (point[0], point[1]), 2, (0, 0, 0), -1)
    # cv.drawContours(img, [approx_arrow_contour], -1, (0, 255, 0), 3)

    # cv.imwrite('contoured.jpg', img)

    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()