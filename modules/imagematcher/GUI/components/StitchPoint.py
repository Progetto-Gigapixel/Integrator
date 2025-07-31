class Point:
    """ Wrap class for a point in the image, with x and y coordinates, dot and text indexes from the Canvas"""
    def __init__(self, x:int, y: int,dot: int,text: str, text_id: int):
        self.x = x
        self.y = y
        self.dot = dot
        self.text = text
        self.text_id = text_id
        self.error_circle = None
        self.reprojection_error = 0

    def set_error_circle(self, index):
        self.error_circle = index

    def set_reprojection_error(self, reprojection : int ):
        self.reprojection_error = reprojection
