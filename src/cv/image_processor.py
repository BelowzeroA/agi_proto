from typing import List, Tuple

import cv2 as cv
import numpy as np
from skimage.measure import approximate_polygon
from sympy import Segment, Point, Line, N, acos
import shapely
from shapely import geometry
import Box2D

ALPHA_LIST = [22.5 + a for a in [0, 45, 90, 135, 180, 225, 270, 315]]


def our_zoom(vertices):
    x = vertices[0]
    y = vertices[1]
    return round(x * 10) + 320, round((20 - y) * 10) + 240


class ImageProcessor():
    def __init__(self, world, filename='pics/test_picture_3.JPG', arm_size=(0, 0, 0, 0)):
        self.ALPHA = 180 / np.arccos(-1)
        #self.img = cv.imread(filename)
        #self.img_gray = cv.imread(filename, 0)
        #self.img = cv.imread('rrrrr.png')
        #print(self.img)
        self.my_world = world
        self.img = filename
        self.img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.arm_size = arm_size
        self.pi_alpha = np.pi / 180
        self.hand_polygon = np.array([[12,  0],
                                   [10, 14],
                                   [ 6,  3],
                                   [ 8, 14],
                                   [ 6, 16],
                                   [ 0, 10],
                                   [ 9, 31],
                                   [17, 31],
                                   [29, 23],
                                   [18, 20],
                                   [21,  4],
                                   [15, 14]])

    def mean(self, lst: list) -> float:
        return sum(lst) / len(lst)

    def distance(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.sqrt(np.sum((p2 - p1) ** 2))

    def get_side(self, p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def segments_have_common_point(self, s1: Segment, s2: Segment) -> bool:
        return s1.p1 == s2.p1 or s1.p1 == s2.p2 or s1.p2 == s2.p1 or s1.p2 == s2.p2

    def get_segment_midpoint(self, segment: Segment) -> Point:
        x = (segment.p1[0] + segment.p2[0]) // 2
        y = (segment.p1[1] + segment.p2[1]) // 2
        return Point(x, y)

    def segments_in_group(self, s1: Segment, s2: Segment, group: List[Segment]) -> Segment:
        segments = [s1, s2]
        if s1 in group:
            segments.remove(s1)
        if s2 in group:
            segments.remove(s2)
        if len(segments) < 2:
            return segments
        return None

    def collect_adjacent_segment_groups(self, segments: List[Segment], threshold: float) -> List[Tuple[Segment, Segment]]:
        adjacent_segments_groups = []
        short_segments = [segment for segment in segments if N(segment.length) < threshold]
        singles = list(short_segments)
        for i in range(len(short_segments)):
            for j in range(len(short_segments)):
                if i == j:
                    continue
                s1 = short_segments[i]
                s2 = short_segments[j]
                if self.segments_have_common_point(s1, s2):
                    if s1 in singles:
                        singles.remove(s1)
                    if s2 in singles:
                        singles.remove(s2)
                    found = False
                    for group in adjacent_segments_groups:
                        segment_to_add = self.segments_in_group(s1, s2, group)
                        if segment_to_add is not None:
                            group.extend(segment_to_add)
                            found = True
                    if not found:
                        adjacent_segments_groups.append([s1, s2])
        return adjacent_segments_groups

    def collect_segments_lengths(self, contour):
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

    def find_splitting_threshold(self, lengths: List[float]) -> float:
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

    def compute_roi_center(self, group: List[Segment]) -> List[Point]:
        buffer = list(group)
        while True:
            if len(buffer) == 1:
                midpoint = self.get_segment_midpoint(buffer[0])
                return midpoint
            points = []
            for segment in buffer:
                midpoint = self.get_segment_midpoint(segment)
                points.append(midpoint)

            buffer = []
            while len(points) > 1:
                point1 = points.pop(0)
                point2 = points.pop(0)
                buffer.append(Segment(point1, point2))

            if len(points) == 1:
                buffer.append(Segment(point2, points[0]))

    def find_roi(self, contour):
        roi_centers = []
        # Extract point, segments and their lengths
        segments, points, lengths = self.collect_segments_lengths(contour)

        # find a threshold which distinguishes short segments from long ones
        threshold = self.find_splitting_threshold(lengths)

        if threshold > 0:
            # collect groups of short segments
            adjacent_segments_groups = self.collect_adjacent_segment_groups(segments, threshold)

            # remove the points that don't belong to the groups extracted above
            for group in adjacent_segments_groups:
                for segment in group:
                    if segment.p1 in points:
                        del points[points.index(segment.p1)]
                    if segment.p2 in points:
                        del points[points.index(segment.p2)]

            # compute a ROI center for each segment group
            for group in adjacent_segments_groups:
                roi_center = self.compute_roi_center(group)
                roi_centers.append(roi_center)

        # alternatively, the ROI centers are the points that don't belong to any group (standalone points)
        roi_centers.extend(points)
        return roi_centers

    def point_orientation(self, vec_point_1, vec_point_2, point):
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

    def is_point_inside_polygon(self, list_of_obj_points, point):
        """
        :param list_of_obj_points: список точек полигона
        :param point: точка на плоскости
        :return: bool, содержится ли точка в полигоне
        """
        polygon = shapely.geometry.polygon.Polygon(list_of_obj_points)
        point = shapely.geometry.Point(point)
        return polygon.contains(point)

    def while_quadrant(self, roi, p1, p2):
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

    def get_mass(self, value):
        if value >= 0.4:
            return 2
        return 1

    def get_corner(self, p1, p2):
        if p1[1] > p2[1]:
            p1, p2 = p2, p1
        a = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        b = np.array([5, 0])
        return round(self.ALPHA * np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

    def into_segment(self, list_x, list_y, p1, p2):
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

    def into_roi_square(self, point, x_left, x_right, y_bottom, y_top):
        return x_left <= point[0] and point[0] <= x_right and y_bottom <= point[1] and point[1] <= y_top

    def is_three_points_belong_to_the_same_shape(self, p1, p2, p3, img):
        point_list = [list(p1), list(p2), list(p3)]
        for point in point_list:
            if point in [[0, 440], [639, 440]]:
                return False
        x_mean = (p1 + p2 + p3) / 3
        if int(x_mean[1]) == 440:
            return False
        elif np.any(img[int(x_mean[1]), int(x_mean[0])] != np.array([0, 0, 0], dtype=np.int8)):
            return img[int(x_mean[1]), int(x_mean[0])]
        else:
            return False

    def is_two_points_belong_to_the_same_shape(self, p1, p2, img):
        x_mean = (p1 + p2) / 2
        res = []
        if np.any(img[int(x_mean[1]), int(x_mean[0])] != np.array([0, 0, 0], dtype=np.int8)):
            return img[int(x_mean[1]), int(x_mean[0])]
        else:
            return False

    def does_current_shapes_contains_this_color(self, all_shapes, color):
        for ind in range(len(all_shapes)):
            if np.all(all_shapes[ind]['color'] == color):
                return ind
        return -1

    def split_shapes(self, approx_contour, img):
        all_shapes = []
        approx_contour = [point for point in approx_contour if not (point[0] == 0 and point[1] == 440)
                          and not (point[0] == 639 and point[1] == 440)
                          ]
        quantity_points = len(approx_contour[:-1])
        for ind in range(len(approx_contour[:-1])):
            p1 = approx_contour[ind]
            p2 = approx_contour[(ind + 1) % quantity_points]
            p3 = approx_contour[(ind + 2) % quantity_points]
            triple_res = self.is_three_points_belong_to_the_same_shape(p1, p2, p3, img)
            if np.any(triple_res != False):
                ind_color = self.does_current_shapes_contains_this_color(all_shapes, triple_res)
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

    def quadrant_roi_analysis(self, roi, approx_contour, quadrant_size, img):
        quadrant_norm = 1.42 * quadrant_size
        result = {}
        try:
            result['roi'] = roi.args
        except AttributeError:
            pass
        result['quadrants'] = [[], [], [], []]
        try:
            polygon = geometry.polygon.Polygon(approx_contour)
            x_left = roi[0] - quadrant_size
            x_right = roi[0] + quadrant_size - 1
            y_top = roi[1] + quadrant_size - 1
            y_bottom = roi[1] - quadrant_size
            list_x = sorted([x_left, x_right, roi[0] - 1, roi[0]])
            list_y = sorted([y_bottom, y_top, roi[1] - 1, roi[1]])
            res = []

            a = list(approx_contour[0])
            approx_contour = list(approx_contour)
            approx_contour.append(a)
            for ind in range(1, len(approx_contour)):
                min_res = []
                p1_flag = self.into_roi_square(approx_contour[ind - 1], x_left, x_right, y_bottom, y_top)
                p2_flag = self.into_roi_square(approx_contour[ind], x_left, x_right, y_bottom, y_top)
                if p1_flag or p2_flag:
                    if p1_flag:
                        min_res.append(tuple(approx_contour[ind - 1]))
                    if p2_flag:
                        min_res.append(tuple(approx_contour[ind]))
                    approx_list_x, approx_list_y = self.into_segment(list_x, list_y, approx_contour[ind - 1],
                                                                     approx_contour[ind])
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
                quadrant_number = self.while_quadrant(roi, res[ind - 1], res[ind])
                if quadrant_number == 5:
                    continue
                if res[ind] == res[ind - 1]:
                    continue
                our_line = {}
                our_line['angle'] = self.get_corner(res[ind - 1], res[ind])
                our_line['mass'] = self.get_mass(np.linalg.norm(np.array([res[ind][0] - res[ind - 1][0],
                                                                          res[ind][1] - res[ind - 1][1]],
                                                                         dtype=np.float32)) / quadrant_norm)
                result['quadrants'][quadrant_number].append(our_line)
            # for color
            for ind in range(1, len(res)):
                if res[ind] == 0 or res[ind - 1] == 0:
                    continue
                if True:
                    x_mean = int((res[ind][0] + res[ind - 1][0]) / 2)
                    y_mean = int((res[ind][1] + res[ind - 1][1]) / 2)
                    try_point = shapely.geometry.Point(x_mean + 3, y_mean)
                    if polygon.contains(try_point):
                        result['color'] = img[y_mean, x_mean + 3]
                        break
                    try_point = shapely.geometry.Point(x_mean - 3, y_mean)
                    if polygon.contains(try_point):
                        result['color'] = img[y_mean, x_mean - 3]
                        break
        except ValueError:
            pass
        return result

    def is_it_in_arm_size(self, obj_points):
        point_max = np.max(obj_points, axis=0)
        point_min = np.min(obj_points, axis=0)
        return (self.arm_size[0] <= point_min[0] and self.arm_size[1] <= point_min[1] and
            point_max[0] <= self.arm_size[0] + self.arm_size[2] and
            point_max[1] <= self.arm_size[1] + self.arm_size[3])

    def general_presentation(self, vec, point_min, point_max, threshold=0.35):
        threshold = threshold * self.distance(point_min, point_max)
        res = [vec[0], ]
        flag = 0
        ind = 0
        while ind < len(vec):
            if ind == len(vec) - 2:
                p1 = vec[ind]
                p2 = vec[ind + 1]
                p3 = vec[0]
            elif ind == len(vec) - 1:
                p1 = vec[ind]
                p2 = vec[0]
                p3 = vec[1]
            else:
                p1 = vec[ind]
                p2 = vec[ind + 1]
                p3 = vec[ind + 2]
            side = self.get_side(p1, p2, p3)
            if self.distance(p1, p3) <= threshold and np.any(p2 != res[-1]) and side > 0:
                res.append(p3)
            elif np.any(res[-1] != p2):
                res.append(p2)
            if side > 0:
                flag += 1
            else:
                flag = 0
            if flag > 1:
                res[-1] = p3
            ind += 1
        return res[:-1]

    def get_intersect(self, a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        a1_a2 = np.array([a2[0] - a1[0], a2[1] - a1[1]])
        a1_b1 = np.array([b1[0] - a1[0], b1[1] - a1[1]])
        a1_b2 = np.array([b2[0] - a1[0], b2[1] - a1[1]])
        b1_b2 = np.array([b2[0] - b1[0], b2[1] - b1[1]])
        b1_a1 = np.array([a1[0] - b1[0], a1[1] - b1[1]])
        b1_a2 = np.array([a2[0] - b1[0], a2[1] - b1[1]])
        if np.cross(a1_a2, a1_b1) * np.cross(a1_a2, a1_b2) > 0:
            return False
        elif np.cross(b1_b2, b1_a1) * np.cross(b1_b2, b1_a2) > 0:
            return False
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return False
        return (round(x / z), round(y / z))

    def gen(self, idx, flag, size):
        if flag:
            while idx < size:
                idx += 1
                yield idx % size
        else:
            while idx > -1:
                idx -= 1
                yield idx % size


    def find_inner_point(self, arm, points, ind, poly_2):
        res = []
        if ind == -1:
            ind_2 = 0
        else:
            ind_2 = -1
        for i in range(len(arm)):
            if arm[i] == points[ind]:
                break
        strategy = True
        if arm[i - 1] == points[ind_2]:
            strategy = False
        for idx in self.gen(i, strategy, len(arm)):
            res.append(arm[idx])
            for triple in poly_2:
                if triple[0] == arm[idx] or triple[-1] == arm[idx]:
                    res.append(triple[1])
                    return res
        return res

    def del_more_zeros(self, obj_ext, obj_flag):
        tail = 0
        head = 0
        count = 0
        for i in range(len(obj_ext)):
            if obj_flag[i] == 0:
                head = i
                count += 1
            elif obj_flag[i] == 1 and count > 2:
                obj_ext = obj_ext[:tail + 1] + obj_ext[head:]
                obj_flag = obj_flag[:tail + 1] + obj_flag[head:]
                return obj_ext, obj_flag
        return obj_ext, obj_flag

    def separate_hand_obj(self, polygon, arm):
        res = []
        obj_ext = []
        list_of_triples = []
        obj_flag = []
        arm_flag = []
        for i in range(len(polygon)):
            temp_l = [polygon[i], polygon[(i + 1) % len(polygon)]]
            for j in range(len(arm)):
                intersect_res = self.get_intersect(polygon[i], polygon[(i + 1) % len(polygon)],
                                                   arm[j], arm[(j + 1) % len(arm)])
                if intersect_res:
                    temp_l.append(intersect_res)
                    list_of_triples.append([arm[j], intersect_res, arm[(j + 1) % len(arm)]])
            temp_l.sort()
            obj_flag.extend([1] + [0] * (len(temp_l) - 2))
            if polygon[i] == temp_l[0]:
                obj_ext.extend(temp_l[:-1])
            else:
                temp_l.reverse()
                obj_ext.extend(temp_l[:-1])
            arm_ext = []
        if len(list_of_triples) == 0:
            return [polygon, ]
        for idx in range(len(arm)):
            arm_ext.append(arm[idx])
            arm_flag.append(1)
            for triple in list_of_triples:
                if np.all(triple[0] == arm[idx]) or np.all(triple[-1] == arm[idx]):
                    arm_ext.append(triple[1])
                    arm_flag.append(0)
                    break
        idx = 0
        while obj_flag[idx] != 0 and idx < len(obj_flag):
            idx += 1
        if obj_flag[idx + 1] == 0:
            obj_ext = obj_ext[idx + 1:] + obj_ext[:idx + 1]
            obj_flag = obj_flag[idx + 1:] + obj_flag[:idx + 1]
        else:
            obj_ext = obj_ext[idx:] + obj_ext[:idx]
            obj_flag = obj_flag[idx:] + obj_flag[:idx]
        obj_ext, obj_flag = self.del_more_zeros(obj_ext, obj_flag)
        obj_arm_dict = {}
        for idx in range(len(obj_ext)):
            if obj_flag[idx] == 0:
                for j in range(len(arm_ext)):
                    if np.all(obj_ext[idx] == arm_ext[j]):
                        #obj_arm_dict[idx] = arm_ext.index(obj_ext[idx])
                        obj_arm_dict[idx] = j
        tail = 0
        count_ones = 0
        for idx in range(len(obj_ext) - 1):
            if obj_flag[idx] == 0 and obj_flag[(idx + 1) % len(obj_flag)] == 0 and count_ones == 0:
                tail = idx + 1
            elif obj_flag[idx] == 0 and obj_flag[(idx + 1) % len(obj_flag)] == 1:
                head = idx + 1
                count_ones += 1
            elif obj_flag[idx] == 1 and obj_flag[(idx + 1) % len(obj_flag)] == 1:
                head = idx + 1
                count_ones += 1
            else:
                segment = obj_ext[tail: head + 1]
                arm_tail = obj_arm_dict[head + 1]
                arm_head = obj_arm_dict[tail]
                if arm_tail > arm_head:
                    temp_l = arm_ext[arm_head: arm_tail]
                    temp_l.reverse()
                else:
                    temp_l = arm_ext[arm_tail: arm_head]
                segment.extend(temp_l)
                res.append(segment)
                count_ones = 0
        return [res, arm]

    def get_approx_for_circle(self, world_body):
        center = world_body.worldCenter
        radius = world_body.fixtures[0].shape.radius - 3
        return [(radius * np.cos(self.pi_alpha * alp) + center[0],
                 radius * np.sin(self.pi_alpha * alp) + center[1]) for alp in ALPHA_LIST]

    def get_approx_for_poly(self, data, angle=0, pos=(0, 0)):
        # angle *= (np.pi / 180)
        pos = np.array(pos)
        result = []
        cos_alpha = np.cos(angle)
        sin_alpha = np.sin(angle)
        rotation_matrix = np.array([[cos_alpha, -sin_alpha],
                                    [sin_alpha, cos_alpha]])
        for point in data:
            result.append(np.dot(rotation_matrix, np.array(point)) + pos)
        return result

    def obj_func(self, data, approx_contour):
        obj_data = {}
        obj_data['rois'] = []
        # try:
        #     approx_contour = approx_contour['points']
        # except IndexError:
        #     pass
        if len(approx_contour) <= 2:
            return
        point_max = np.max(approx_contour, axis=0)
        point_min = np.min(approx_contour, axis=0)
        if point_min[1] < 60:
            return
        roi = self.find_roi(approx_contour)
        x_mean = 0
        y_mean = 0
        for point in roi:
            x_mean += point.args[0]
            y_mean += point.args[1]
            obj_data['rois'].append(
                self.quadrant_roi_analysis(point, approx_contour, 10, self.img)
            )
            cv.circle(self.img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
        for point in approx_contour:
            # small black dots - points after polygonizing
            cv.circle(self.img, (point[0], point[1]), 2, (0, 255, 0), -1)
        # cv.imshow('Image', self.img)
        obj_data['center'] = (int(round(x_mean / len(roi))), int(round(y_mean / len(roi))))
        dist = np.max(point_max - point_min) // 2 + 2
        width_height = point_max - point_min
        obj_data['width'] = width_height[0]
        obj_data['height'] = width_height[1]
        if len(approx_contour) == 3:
            obj_data['name'] = 'triangle'
        elif len(approx_contour) == 8:
            obj_data['name'] = 'circle'
        else:
            obj_data['name'] = 'hand'
        obj_data['general_presentation'] = self.quadrant_roi_analysis(
            obj_data['center'],
            self.general_presentation(approx_contour,
                                      point_min,
                                      point_max),
            dist,
            self.img)
        data.append(obj_data)

    def run(self, last_position=None):
        #ret, thresh = cv.threshold(self.img_gray, 0, 200, 0)
        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        data = []
        arm = self.hand_polygon + np.array([self.arm_size[0] + 1, self.arm_size[1]])
        arm_flag = False
        if len(self.my_world.bodies) != 0:
            for world_body in self.my_world.bodies:
                if len(world_body.fixtures) == 0:
                    continue
                if type(world_body.fixtures[0].shape) == Box2D.b2CircleShape:
                    approx_contour = self.get_approx_for_circle(world_body)
                elif type(world_body.fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
                    approx_contour = self.get_approx_for_poly(world_body.fixtures[0].shape.vertices,
                                                              world_body.transform.angle,
                                                              world_body.transform.position)
                else:
                    continue
                approx_contour = [our_zoom(p) for p in approx_contour]
                line = self.separate_hand_obj(approx_contour, arm)
                if len(line) == 2:
                    temp_data = []
                    for approx_contour in line[0]:
                        self.obj_func(temp_data, approx_contour)
                    temp_obj = temp_data[0]
                    for temp in temp_data[1:]:
                        temp_obj['rois'].extend(temp['rois'])
                    data.append(temp_obj)
                else:
                    for approx_contour in line:
                        self.obj_func(data, approx_contour)
        self.obj_func(data, arm)
        if last_position:
            for obj, last_center in zip(data, last_position):
                obj['offset'] = (obj['center'][0] - last_center[0],
                                 obj['center'][1] - last_center[1])
        else:
            for obj in data:
                obj['offset'] = (0, 0)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return data
