import numpy as np
from scipy import linalg


def line_segment_intersection(r1,r2,q1,q2):
    b = np.array([q1-r1]).transpose()
    A = np.array([r2-r1, q1-q2]).transpose()
    x = linalg.solve(A,b)
    r2_r1 = np.array([r2-r1]).transpose()
    xr2_r1 = np.array([x[0]*r2_r1[0],x[0]*r2_r1[1]]).reshape((2))
    p = np.array([r1[0]+xr2_r1[0],r1[1]+xr2_r1[1]])
    
    return p

def check_if_point_inside_triangle(point, triangle):
    #  Barycentric method
    #http://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
    x1,y1 =triangle[0,0],triangle[0,1]
    x2,y2 =triangle[1,0],triangle[1,1]
    x3,y3 =triangle[2,0],triangle[2,1]
    x, y =point[0], point[1]

    denominator=  ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    a =  ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
    b =((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
    c = 1 - a - b
    
    return 0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1

