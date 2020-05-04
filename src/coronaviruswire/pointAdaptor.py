from psycopg2.extensions import adapt, register_adapter, AsIs


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def adapt_point(point):
    x = adapt(point.x)
    y = adapt(point.y)
    return AsIs("'(%s, %s)'" % (x, y))


def adapt_point_array(points):
    fmt_string = "ARRAY["
    for point in points:
        fmt_string += "point(%s, %s)," % (point.x, point.y)
    fmt_string = fmt_string[:-1]
    fmt_string += "]"

    return AsIs(fmt_string)
