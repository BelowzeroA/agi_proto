from typing import List, Tuple

import cv2 as cv
import numpy as np
from sympy import Segment, Point, Line, N, acos
import Box2D

from .separator import Separator
from .roi_analysis import RoiAnalysis

ALPHA_LIST = [22.5 + a for a in [0, 45, 90, 135, 180, 225, 270, 315]]


def our_zoom(vertices):
    x = vertices[0]
    y = vertices[1]
    return round(x * 10) + 320, round((20 - y) * 10) + 240


class ImageProcessor(Separator, RoiAnalysis):
    def __init__(self, world, filename='pics/test_picture_3.JPG', arm_size=(0, 0, 0, 0)):
        self.ALPHA = 180 / np.arccos(-1)
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

    def get_side(self, p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def segments_have_common_point(self, s1: Segment, s2: Segment) -> bool:
        try:
            return s1.p1 == s2.p1 or s1.p1 == s2.p2 or s1.p2 == s2.p1 or s1.p2 == s2.p2
        except AttributeError:
            return False

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

    def get_approx_for_circle(self, world_body):
        center = world_body.worldCenter
        radius = world_body.fixtures[0].shape.radius - 3
        return [(radius * np.cos(self.pi_alpha * alp) + center[0],
                 radius * np.sin(self.pi_alpha * alp) + center[1]) for alp in ALPHA_LIST]

    def get_approx_for_poly(self, data, angle=0, pos=(0, 0), in_radians=False):
        if not in_radians:
            angle *= (np.pi / 180)
        pos = np.array(pos)
        result = []
        cos_alpha = np.cos(angle)
        sin_alpha = np.sin(angle)
        rotation_matrix = np.array([[cos_alpha, -sin_alpha],
                                    [sin_alpha, cos_alpha]])
        for point in data:
            result.append(np.dot(rotation_matrix, np.array(point)) + pos)
        return result

    def obj_filter(self, approx_contour):
        i = 1
        len_counter = len(approx_contour)
        while i < len_counter:
            if np.all(approx_contour[i - 1] == approx_contour[i]):
                approx_contour = approx_contour[:i - 1] + approx_contour[i:]
                len_counter = len(approx_contour)
            else:
                i += 1
        if len(approx_contour) < 3:
            return False
        else:
            return approx_contour

    def obj_func(self, data, approx_contour):
        if len(approx_contour) == 0:
            return
        obj_data = {}
        obj_data['rois'] = []
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
            cv.circle(self.img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
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
        data = []
        arm = self.hand_polygon + np.array([self.arm_size[0] + 1, self.arm_size[1]])
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
                        approx_contour = self.obj_filter(approx_contour)
                        if approx_contour:
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
        print(len(data))
        return data
