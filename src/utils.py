from models.point import Point
import math


def calculate_center_point(pt1: Point, pt2: Point) -> Point:
    """
    Calculates the center point of a rectangle defined by two Points.

    This function takes two Points representing the upper-left (pt1) and
    lower-right (pt2) corners of a rectangle. It calculates the center point
    of this rectangle, which is particularly useful in scenarios such as
    determining the center of a bounding box in object detection tasks.

    The center point is calculated by averaging the x-coordinates and
    y-coordinates of the two provided points, respectively. This method
    ensures that the calculated center is geometrically accurate regardless
    of the rectangle's size or position.

    Parameters:
    pt1 (Point): The upper-left corner of the rectangle.
    pt2 (Point): The lower-right corner of the rectangle.

    Returns:
    Point: A Point instance representing the center of the rectangle.

    Example:
    Given pt1 = Point(2, 4) and pt2 = Point(6, 8),
    the function will return Point(4, 6) as the center point.
    """
    cx: int = (pt1.x + pt2.x) // 2
    cy: int = (pt1.y + pt2.y) // 2
    return Point(x=cx, y=cy)


def euclidean_distance(pt1: Point, pt2: Point) -> float:
    """
    Calculates the Euclidean distance between two points.

    This function computes the Euclidean distance, which is the "ordinary" straight-line
    distance between two points in Euclidean space. The distance is calculated using the
    Pythagorean theorem.

    Parameters:
    pt1 (Point): The first point, an instance of the Point class with 'x' and 'y' attributes.
    pt2 (Point): The second point, also an instance of the Point class with 'x' and 'y' attributes.

    Returns:
    float: The Euclidean distance between the two points.

    Example:
    Given pt1 = Point(3, 4) and pt2 = Point(6, 8),
    the function will calculate and return the distance as 5.0, which is the straight-line
    distance between these two points in a 2D space.
    """
    return math.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2)
