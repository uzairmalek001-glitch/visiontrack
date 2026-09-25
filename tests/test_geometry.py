import unittest

from visiontrack.vision.geometry import (
    bbox_area,
    bbox_intersects,
    bbox_intersects_polygon,
    bbox_iou,
    bbox_union,
    cardinal_direction,
    distance,
    is_simple_polygon,
    point_in_polygon,
    polygon_area,
    polygon_array,
    segments_intersect,
    side_of_line,
)


class BBoxTests(unittest.TestCase):
    def test_area(self):
        self.assertEqual(bbox_area((0, 0, 10, 5)), 50)

    def test_intersects_true_for_overlap(self):
        self.assertTrue(bbox_intersects((0, 0, 10, 10), (5, 5, 10, 10)))

    def test_intersects_false_for_disjoint(self):
        self.assertFalse(bbox_intersects((0, 0, 10, 10), (100, 100, 10, 10)))

    def test_intersects_true_for_touching_edges(self):
        self.assertTrue(bbox_intersects((0, 0, 10, 10), (10, 0, 10, 10)))

    def test_iou_identical_boxes_is_one(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_iou_disjoint_is_zero(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)), 0.0)

    def test_iou_partial_overlap(self):
        # two 10x10 boxes overlapping in a 5x10 strip -> intersection 50, union 150
        iou = bbox_iou((0, 0, 10, 10), (5, 0, 10, 10))
        self.assertAlmostEqual(iou, 50 / 150)

    def test_union_bounds_all_boxes(self):
        self.assertEqual(bbox_union([(0, 0, 5, 5), (10, 10, 5, 5)]), (0, 0, 15, 15))

    def test_union_empty_is_none(self):
        self.assertIsNone(bbox_union([]))


class PolygonTests(unittest.TestCase):
    def setUp(self):
        self.square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_point_inside(self):
        self.assertTrue(point_in_polygon((5, 5), polygon_array(self.square)))

    def test_point_outside(self):
        self.assertFalse(point_in_polygon((50, 50), polygon_array(self.square)))

    def test_point_on_boundary_counts_as_inside(self):
        self.assertTrue(point_in_polygon((0, 5), polygon_array(self.square)))

    def test_area_of_unit_square_like_shape(self):
        self.assertAlmostEqual(polygon_area(self.square), 100.0)

    def test_simple_square_is_simple(self):
        self.assertTrue(is_simple_polygon(self.square))

    def test_self_intersecting_bowtie_is_not_simple(self):
        bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
        self.assertFalse(is_simple_polygon(bowtie))

    def test_bbox_intersects_polygon_when_box_inside(self):
        self.assertTrue(bbox_intersects_polygon((2, 2, 3, 3), self.square, polygon_array(self.square)))

    def test_bbox_intersects_polygon_when_disjoint(self):
        self.assertFalse(bbox_intersects_polygon((100, 100, 5, 5), self.square, polygon_array(self.square)))

    def test_bbox_intersects_polygon_when_box_straddles_edge(self):
        # box centered on the polygon's right edge, no corner inside, no vertex inside the box
        self.assertTrue(bbox_intersects_polygon((8, 4, 4, 2), self.square, polygon_array(self.square)))


class LineTests(unittest.TestCase):
    def test_segments_intersect_crossing(self):
        self.assertTrue(segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)))

    def test_segments_do_not_intersect_parallel(self):
        self.assertFalse(segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)))

    def test_side_of_line_opposite_signs_on_opposite_sides(self):
        a, b = (0, 0), (10, 0)
        above = side_of_line((5, -5), a, b)
        below = side_of_line((5, 5), a, b)
        self.assertTrue(above * below < 0)

    def test_side_of_line_zero_on_the_line(self):
        self.assertEqual(side_of_line((5, 0), (0, 0), (10, 0)), 0)


class MiscGeometryTests(unittest.TestCase):
    def test_distance_pythagorean(self):
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0)

    def test_cardinal_direction_right(self):
        self.assertEqual(cardinal_direction(10, 0), "right")

    def test_cardinal_direction_down(self):
        self.assertEqual(cardinal_direction(0, 10), "down")

    def test_cardinal_direction_stationary_below_min_speed(self):
        self.assertEqual(cardinal_direction(0.1, 0.1, min_speed=1.0), "stationary")


if __name__ == "__main__":
    unittest.main()
