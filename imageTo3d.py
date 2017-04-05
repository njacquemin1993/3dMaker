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
            if max(x) > 1000:
                print(x)
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
    theta = [np.arctan2(x[1], x[0])%(np.pi*2) for x in coords]
    data = zip(x, y, theta)
    data = sort_and_clean(data)
    data.append(data[0])
    data[-1] = (data[-1][0], data[-1][1], data[-1][2] + 2*np.pi)  
    theta = [x[2] for x in data]
    data = np.c_[[x[0] for x in data], [x[1] for x in data]]
    return theta, data

def interpolate_slice_old(coords): #TODO find bug
    if len(coords)>0:
        theta, data = clean_data(coords)
        cs = CubicSpline(theta, data, bc_type='periodic')
        xs = 2 * np.pi * np.linspace(0, 1, 100)
        coords = [(cs(x)[0], cs(x)[1]) for x in xs]
    return coords
    
def approx(x, angles, coords):
    i = 0
    while x <= angles[len(angles)-1-i] and i < len(angles)-1:
        i += 1
    i = len(angles)-1-i
    coord1 = coords[i]
    coord2 = coords[(i+1)%len(coords)]
    r1 = np.sqrt(coord1[0]**2 + coord1[1]**2)
    r2 = np.sqrt(coord2[0]**2 + coord2[1]**2)
    r = r1 + (r2-r1)*((x - angles[i])%(np.pi*2))/((angles[(i+1)%len(coords)]-angles[i])%(np.pi*2))
    return (r*np.cos(x), r*np.sin(x))
    
def interpolate_slice(coords):
    if len(coords)>0:
        theta, data = clean_data(coords)
        angles = 2 * np.pi * np.linspace(0, 1, 100)
        exit
        coords = [approx(angle, theta, data) for angle in angles]
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
    coords_to_stl(list_coords)

def coords_to_stl(coords):
    dic_coords = {}
    for x,y,z in coords:
        if z not in dic_coords:
            dic_coords[z] = []
        dic_coords[z].append({'x':x, 'y':y, 'r': np.sqrt(x*x+y*y), 'a':np.arctan2(y, x)})
    list_remove = []
    for z in dic_coords:
        x = [toto['x'] for toto in dic_coords[z]]
        y = [toto['y'] for toto in dic_coords[z]]
        if max(max(x), max(y)) > 1000:
            list_remove.append(z)
    for z in list_remove:
        del dic_coords[z]
    with open('out.stl', 'w') as f:
        dic_keys = sorted(dic_coords.keys())
        f.write('solid name\n\n')
        for i, z in enumerate(dic_keys):
            if z == dic_keys[-1] or z == dic_keys[0]:
                layer = sorted(dic_coords[z], key=lambda x: x['a'])
                length = len(layer)
                for c in range(length):
                    point1 = (layer[c]['x'], layer[c]['y'], z)
                    point2 = (layer[(c+1)%length]['x'], layer[(c+1)%length]['y'], z)
                    point3 = (0, 0, z)
                    write_triangle(f, point1, point2, point3)
            else:
                layer_1 = sorted(dic_coords[z], key=lambda x: x['a'])
                layer_2 = sorted(dic_coords[dic_keys[i+1]], key=lambda x: x['a'])
                length = len(layer_1)
                for c in range(length):
                    point1 = (layer_1[c]['x'], layer_1[c]['y'], z)
                    point2 = (layer_1[(c+1)%length]['x'], layer_1[(c+1)%length]['y'], z)
                    point3 = (layer_2[c]['x'], layer_2[c]['y'], dic_keys[i+1])
                    point4 = (layer_2[(c+1)%length]['x'], layer_2[(c+1)%length]['y'], dic_keys[i+1])
                    write_triangle(f, point1, point2, point3)
                    write_triangle(f, point4, point2, point3)
        f.write('endsolid name')
         
def write_triangle(f, p1, p2, p3):
    normal = get_normal(p1, p2, p3)
    f.write('facet normal {} {} {}\n'.format(normal[0], normal[1], normal[2]))
    f.write('\touter loop\n')
    f.write('\t\tvertex {} {} {}\n'.format(p1[0], p1[1], p1[2]))
    f.write('\t\tvertex {} {} {}\n'.format(p2[0], p2[1], p2[2]))
    f.write('\t\tvertex {} {} {}\n'.format(p3[0], p3[1], p3[2]))
    f.write('\tendloop\n')
    f.write('endfacet\n\n')

def get_normal(p1, p2, p3):
    v1 = [p1[i]-p2[i] for i in range(3)]
    v2 = [p1[i]-p3[i] for i in range(3)]
    x = v1[1]*v2[2] - v1[2]*v2[1]
    y = v1[2]*v2[0] - v1[0]*v2[2]
    z = v1[0]*v2[1] - v1[1]*v2[0]
    n = np.sqrt(x*x+y*y+z*z)
    if n==0:
        n=1
    return (x/n, y/n, z/n)

if __name__ == "__main__":
    source_path = "."
    img_dic = {'img1.jpg' : [45, (270, 10)], 'img2.jpg' : [135, (275, 3)], 'img3.jpg' : [0, (300, 10)], 'img5.jpg' : [90, (245, 5)]}
    main(source_path, img_dic)
