class BoundingBox:
    """Class holding boxes compatible with OpenCV drawing"""

    def __init__(self, top: int, left: int, width: int, height: int):
        """
        :param id: bounding box ID
        :param topleft: top left point of the bounding box
        :param width: bounding box width
        :param height: bounding box height
        """
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def topleft(self) -> ():
        """
        :return: returns top left point
        """
        topleft = (self.top, self.left)
        return topleft

    def topright(self) -> ():
        """
        :return: calculates and returns top right, useful for drawing
        """
        bottomright = (self.top + self.width, self.left)
        return bottomright

    def bottomleft(self) -> ():
        """
        :return: calculates and returns bottom left right, useful for drawing
        """
        bottomright = (self.top, self.left + self.height)
        return bottomright

    def bottomright(self) -> ():
        """
        :return: calculates and returns bottom right, useful for drawing
        """
        bottomright = (self.top + self.width, self.left + self.height)
        return bottomright

    def contains_point(self, point: (float, float)) -> bool:
        """
        :param point: pointed to be checked
        :return: boolean if point is in the area
        """
        return (self.topleft()[0] <= point[0] <= self.bottomright()[0] and
                self.topleft()[1] <= point[1] <= self.bottomright()[1])

    def get_area(self):
        """
        :return: calculated area
        """
        return self.width * self.height

    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        :param other: other boundingbox to be unioned with
        :return: new bounding box, union of two
        """
        bbox = BoundingBox(0, 0, 0, 0)
        bbox.top = min(self.topleft()[0], other.topleft()[0])
        bbox.left = min(self.topleft()[1], other.topleft()[1])
        bottomright = (max(self.bottomright()[0], other.bottomright()[0]),
                       max(self.bottomright()[1], other.bottomright()[1]))

        bbox.width = (bottomright[0] - bbox.topleft()[0])
        bbox.height = (bottomright[1] - bbox.topleft()[1])
        return bbox

    def intersection(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        :param other: other boundingbox to be intersected with
        :return: new bounding box, union of two
        """
        bbox = BoundingBox(0, 0, 0, 0)
        bbox.top = max(self.topleft()[0], other.topleft()[0])
        bbox.left = max(self.topleft()[1], other.topleft()[1])
        bottomright = (min(self.bottomright()[0], other.bottomright()[0]),
                       min(self.bottomright()[1], other.bottomright()[1]))

        bbox.width = (bottomright[0] - bbox.topleft()[0])
        bbox.height = (bottomright[1] - bbox.topleft()[1])

        return bbox

    def iou(self, other: 'BoundingBox') -> float:
        return self.intersection(other).get_area() / self.union(other).get_area()
