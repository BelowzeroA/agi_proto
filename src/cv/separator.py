import numpy as np
import shapely

class Separator():

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

    def del_more_zeros(self, obj_ext, obj_flag):
        tail = 0
        head = 0
        count = 0
        print('len_before', len(obj_ext), obj_flag)
        for i in range(len(obj_ext)):
            if i >= 1 and obj_flag[i - 1] == 1 and obj_flag[i] == 0:
                tail = i
                count += 1
            elif obj_flag[i] == 0:
                head = i
                count += 1
            elif obj_flag[i] == 1 and count > 2:
                obj_ext = obj_ext[:tail + 1] + obj_ext[head:]
                obj_flag = obj_flag[:tail + 1] + obj_flag[head:]
                print('len_after', len(obj_ext), obj_flag)
                return obj_ext, obj_flag
            else:
                count = 0
        return obj_ext, obj_flag

    def is_segment_contains_point(self, segment, point):
        segment = shapely.geometry.LineString(segment)
        point = shapely.geometry.Point(point)
        return segment.contains(point)

    def make_correctly(self, segment):
        res = [segment[0], ]
        for i in range(1, len(segment) - 1):
            if self.is_segment_contains_point([segment[0], segment[-1]], segment[i]):
                res.append(segment[i])
            elif self.is_segment_contains_point([segment[0], segment[-1]],
                                                (segment[i][0], segment[i][1] - 1)):
                res.append((segment[i][0], segment[i][1] - 1))
            elif self.is_segment_contains_point([segment[0], segment[-1]],
                                                (segment[i][0], segment[i][1] + 1)):
                res.append((segment[i][0], segment[i][1] + 1))
            else:
                continue
        res.append(segment[-1])
        return res

    def separate_hand_obj(self, polygon, arm):
        res = []
        obj_ext = []
        list_of_triples = []
        obj_flag = []
        arm_flag = []
        for i in range(len(polygon)):
            temp_l = [polygon[i],]
            for j in range(len(arm)):
                intersect_res = self.get_intersect(polygon[i], polygon[(i + 1) % len(polygon)],
                                                   arm[j], arm[(j + 1) % len(arm)])
                if intersect_res:
                    temp_l.append(intersect_res)
                    list_of_triples.append([arm[j], intersect_res, arm[(j + 1) % len(arm)]])
            # temp_l = self.make_correctly(temp_l)
            temp_l.sort(key=lambda x: x[0])
            obj_flag.extend([1] + [0] * (len(temp_l) - 1))
            if polygon[i] == temp_l[0]:
                temp_l.append(polygon[(i + 1) % len(polygon)])
                obj_ext.extend(temp_l[:-1])
            else:
                temp_l = [polygon[(i + 1) % len(polygon)],] + temp_l
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
        idx = 0
        while obj_flag[idx] != 0 and idx + 1 < len(obj_flag):
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
