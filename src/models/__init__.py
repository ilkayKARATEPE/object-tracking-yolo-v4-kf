from typing import NamedTuple


class Point(NamedTuple):
    """
    Represents a point in 2D space.

    Attributes:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.

    NamedTuple creates an immutable and lightweight object. Unlike regular tuples, NamedTuple instances
    allow for accessing elements by name (using dot notation) in addition to index, enhancing code readability.

    Example:
        point = Point(3, 4)
        print(point.x)  # Output: 3
        print(point.y)  # Output: 4
    """

    x: int
    y: int
