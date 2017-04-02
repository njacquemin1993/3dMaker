#!/usr/bin/env python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def img_open(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def img_close(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def img_edge(img, kernel_size, it):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    d_img = cv2.dilate(img, kernel, iterations=it)
    e_img = cv2.erode(img, kernel, iterations=it)
    return e_img - d_img

def get_edges(img):
    original_img = cv2.imread(img)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY )
    (thresh, bw_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return 255-cv2.Canny(bw_img,100,200)

def get_coords_edges(edges, center):
    list_pixels = np.ndarray.tolist(edges)
    height = len(list_pixels)
    width = len(list_pixels[0])
    list_coords = []
    for ycoord, line in enumerate(list_pixels):
        for xcoord, pixel in enumerate(line):
            if pixel == 0:
                list_coords.append((xcoord-center[0], height-ycoord, 0))
    return list_coords, 1

def rotate(coords, axis, angle):
    if angle%360 == 0:
        return coords
    if type(coords) is type((1, 2)):
        cosinus = np.cos(np.deg2rad(angle))
        sinus = np.sin(np.deg2rad(angle))
        x = coords[0]
        y = coords[1]
        z = coords[2]  
        if axis == 0:        
            return (x, y*cosinus - z*sinus, y*sinus + z*cosinus)
        elif axis == 1:
            return (z*sinus + x*cosinus, y, z*cosinus - x*sinus)
        else:
            return (x*cosinus - y*sinus, x*sinus + y*cosinus, z)
    else:
        return [rotate(x, axis, angle) for x in coords]

def interpolate_full(coords):
    interpolated = []
    for i in range(601):
        tmp = interpolate_slice([(x[0],x[2]) for x in coords if x[1] <= i+0.5 and x[1] > i-0.5])
        for x in tmp:
            interpolated.append((x[0], x[1], i))
    return interpolated

def sort_and_clean(data):
    dic = {}
    for x in data:
        if x[2] not in dic:
            dic[x[2]] = x
    data = [dic[x] for x in dic]
    data = sorted(data, key=lambda x: x[2])
    return data

def clean_data(coords):
    x = [x[0] for x in coords]
    y = [x[1] for x in coords]
    theta = [np.arctan2(x[1], x[0]) for x in coords]
    data = zip(x, y, theta)
    data = sort_and_clean(data)
    data.append(data[0])
    data[-1] = (data[-1][0], data[-1][1], data[-1][2] + 2*np.pi)  
    theta = [x[2] for x in data]
    data = np.c_[[x[0] for x in data], [x[1] for x in data]]
    return theta, data

def interpolate_slice(coords):
    if len(coords)>0:
        theta, data = clean_data(coords)
        cs = CubicSpline(theta, data, bc_type='periodic')
        xs = 2 * np.pi * np.linspace(0, 1, 100)
        coords = [(cs(x)[0], cs(x)[1]) for x in xs]
    return coords
    
def main(src_path, img_dic):
    list_coords = []
    for img in img_dic:
        edges = get_edges(os.path.join(src_path, img))
        coords, axis = get_coords_edges(edges, img_dic[img][1])
        coords = rotate(coords, axis, img_dic[img][0])
        dot = img_dic[img][1]
        cv2.rectangle(edges,(dot[0]-1, dot[1]-1), (dot[0]+1, dot[1]+1),(0,255,0),3)
        list_coords.extend(coords)
    list_coords = interpolate_full(list_coords)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [x[0] for x in list_coords]
    ys = [x[1] for x in list_coords]
    zs = [x[2] for x in list_coords]
    ax.scatter(xs, ys, zs)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(0, 600)
    plt.show()
    print(len(list_coords))
    with open('out.obj', 'w') as f:
        for c in list_coords:
            f.write('v {} {} {}\n'.format(c[0], c[1], c[2]))

if __name__ == "__main__":
    source_path = "."
    img_dic = {'img1.jpg' : [45, (270, 10)], 'img2.jpg' : [135, (275, 3)], 'img3.jpg' : [0, (300, 10)], 'img5.jpg' : [90, (245, 5)]}
    main(source_path, img_dic)
