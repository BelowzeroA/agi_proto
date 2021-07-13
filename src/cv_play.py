from typing import List, Tuple

import cv2 as cv
import numpy as np
from skimage.measure import approximate_polygon
from sympy import Segment, Point, Line, N, acos


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


def main():
    filename = 'pics/test_picture_3.JPG'
    img = cv.imread(filename)
    img_gray = cv.imread(filename, 0)

    ret, thresh = cv.threshold(img_gray, 230, 200, 0)

    # Compute contours of the shapes
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    arrow_contour = np.squeeze(contours[6])

    # polygonize the rounded square
    approx_contour = approximate_polygon(arrow_contour, tolerance=2.5)

    # compute the ROI centers
    roi = find_roi(approx_contour)

    for point in roi:
        # Big red dots - centers of ROI
        cv.circle(img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)

    for point in approx_contour:
        # small black dots - points after polygonizing
        cv.circle(img, (point[0], point[1]), 2, (0, 0, 0), -1)
    # cv.drawContours(img, [approx_arrow_contour], -1, (0, 255, 0), 3)

    cv.imwrite('contoured.jpg', img)

    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()