import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.path as mpath
from matplotlib.patches import Polygon, Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
import imageio
import numpy as np
import random
import h5py
from itertools import product
import sys, time
import glob
from ezprogbar import ProgressBar

dpi = 120
plt.rcParams['figure.dpi'] = dpi
pix = 96
num_classes = 10
class_name = {
    0: '0-triangles',
    1: '1-triangles',
    2: '2-triangles',
    3: '3-triangles',
    4: '4-triangles',
    5: '5-triangles',
    6: '6-triangles',
    7: '7-triangles',
    8: '8-triangles',
    9: '9-triangles',
}

def write_data(h5file, class_size):
    print("Writing file:", h5file)
    f = h5py.File(h5file, 'w')
    pb = ProgressBar(num_classes * class_size)
    i = 0
    for img,tab in gen_data(class_size):
        f.create_dataset(f'img_{i}', data=img, compression='gzip')
        cls = np.count_nonzero(tab == 0)  # number of triangles
        f.create_dataset(f'cls_{i}', data=cls)
        f.create_dataset(f'tab_{i}', data=tab)
        i += 1
        pb.advance()
    num_images = i

    f.create_dataset('num_images', data=num_images)
    f.create_dataset('num_classes', data=num_classes)
    f.create_dataset('features', data=range(num_classes))
    f.close()
    return h5file

def gen_data(class_size):
    pairs = list(product(range(num_classes), range(class_size)))
    random.shuffle(pairs)
    for num_tri,i in pairs:
        yield gen_image(num_tri)

def gen_image(num_tri):
    vsize = hsize = float(pix)/dpi
    #print("vsize, hsize:", vsize, hsize)
    #fig, ax = plt.subplots(figsize=(9,9))
    fig = plt.figure(figsize=(hsize, vsize), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis('off')
    #ax = fig.add_subplot(1,1,1)
    patches = []
    num_geos = 9
    geo_indices = num_tri * [0]
    for i in range(num_geos - num_tri):
        n =  np.random.choice([1,2],1)[0]
        #n =  np.random.choice([0,4,5,6,7,8,9,10],1)[0]
        geo_indices.append(n)
    random.shuffle(geo_indices)

    #rec = Rectangle((0,0), 3.0, 3.0, fill=False, edgecolor="red", facecolor="none") ; patches.append(rec)

    tab = []

    for x0 in [0.5, 1.5, 2.5]:
        row = []
        for y0 in [0.5, 1.5, 2.5]:
            n = geo_indices.pop()
            row.insert(0, n)
            if n==0 or n==1:
                coords = gen_poly_coords(3+n, x0, y0, 0.5)
                polygon = Polygon(coords, fill=True, linewidth=None)
                patches.append(polygon)
            else:
                w = np.random.uniform(0.6, 0.9)
                h = np.random.uniform(0.6, 0.9)
                a = np.random.uniform(0, 90)
                el = Ellipse((x0,y0), width=w, height=h, angle=a, linewidth=None)
                patches.append(el)
        tab.append(row)

    #pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, linewidth=0)
    colors = 5*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    ax.axis('off')
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 3.0])
    ax.relim()
    #ax.autoscale_view(True, True, True)
    #If we haven't already shown or saved the plot, then we need to
    #draw the figure first...
    fig.canvas.draw()
    #Now we can save it to a numpy array.
    #print("Canvas width/height:", fig.canvas.get_width_height())
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    #print("img shape:", img.shape)
    img = img.reshape((pix, pix, 3))
    #print("img shape:", img.shape)
    #imsave("foo.jpg", img)
    #img = imresize(img, (64, 64, 3))
    #plt.imshow(img, interpolation='none', cmap=matplotlib.cm.jet)
    #plt.show()
    plt.clf()
    plt.close()
    tab = np.array(tab).T
    return img,tab

def gen_poly_coords(n, x0, y0, rad=3):
    angle = np.pi / n
    points = []
    for i in range(n):
        r = rad * np.random.uniform(0.9, 1.0)
        a = np.random.uniform(2*i*angle, (2*i+1) * angle)
        x = x0 + r * np.cos(a)
        y = y0 + r * np.sin(a)
        points.append([x,y])
    return np.array(points)

def get_img(hfile, i):
    f = h5py.File(hfile,'r')
    img = np.array(f.get('img_' + str(i)))
    cls = f.get('cls_' + str(i))[()]
    f.close()
    return img, cls

def draw_sample(hfile, n, rows=4, cols=4, imfile=None, fontsize=10):
    plt.figure(figsize=(12, 7.6))
    plt.rcParams['figure.dpi'] = 180
    for i in range(0, rows*cols):
        ax = plt.subplot(rows, cols, i+1)
        img, cls = get_img(hfile, n+i+1)
        plt.imshow(img, cmap='jet')
        plt.title(class_name[cls], fontsize=fontsize, fontweight='bold', y=1.08)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.subplots_adjust(wspace=0.70, hspace=0.05)
    if imfile:
        plt.savefig(imfile, bbox_inches='tight')
    plt.show()

def draw_image(im, title=None):
    #plt.imshow(im, cmap='gnuplot2', extent=(0,47,0,47), aspect='auto')
    #plt.imshow(im, cmap='gnuplot2', interpolation='bessel')
    plt.imshow(im, cmap='jet', interpolation='none')
    #plt.imshow(im, cmap='jet')
    if title:
        plt.title(title)
    #plt.axis('off')
    plt.show()

#def plot_image(im):
#    im = 255 * (im - im.min()) / (im.max() - im.min())
#    im = im.astype(int)
#    plt.imshow(im, cmap='jet', interpolation='none')

def draw_image2(hfile, i):
    img, cls = get_img(hfile, i)
    plt.imshow(img, cmap='jet')
    plt.title(class_name[cls], fontsize=15, fontweight='bold', y=1.05)
    plt.show()

def draw_poly(coords, edgecolor='blue', facecolor='AliceBlue', fill=False, linewidth=1, closed=True, marker="s", markersize=3, alpha=1.0):
    coords = list(coords)
    if isinstance(coords[0], tuple):
        points = coords
    else:
        points = []
        for i in range(0, len(coords), 2):
            p = (float(c) for c in coords[i:i+2])
            points.append(p)
    poly = Polygon(
        points,
        closed=closed,
        edgecolor=edgecolor,
        facecolor=facecolor,
        fill=fill,
        linewidth=linewidth,
        alpha = alpha,
    )
    axes = plt.gca()
    axes.add_patch(poly)
    plt.grid('on', linestyle=':')
    return poly

def extract_image_from_h5_file(h5file, i, imfile=None):
    with h5py.File(h5file,'r') as f:
        num_images = f.get('num_images')[()]
        print("Num images:", num_images)
        img_key = f'img_{i}'
        cls_key = f'cls_{i}'
        tab_key = f'tab_{i}'
        for key in [img_key, cls_key, tab_key]:
            print(f"{key}  -->  {np.array(f.get(key))}")
        img = np.array(f.get(img_key))
        print('Shape:', img.shape)
        plt.imshow(img, interpolation='none', cmap=matplotlib.cm.jet)
        imfile = f"image_{i}.png"
        plt.imshow(img, interpolation='none', cmap=matplotlib.cm.jet)
        #plt.savefig(imfile, bbox_inches='tight')
        if imfile:
            imageio.imwrite(imfile, img)
        plt.show()

def test_h5_file(h5file):
    with h5py.File(h5file,'r') as f:
        num_images = f.get('num_images')[()]
        print("Num images:", num_images)
        for i in range(0,num_images):
            img_key = f'img_{i}'
            cls_key = f'cls_{i}'
            tab_key = f'tab_{i}'
            for key in [img_key, cls_key, tab_key]:
                print(f"{key}  -->  {np.array(f.get(key))}")
            img = np.array(f.get(img_key))
            print('Shape:', img.shape)
            plt.imshow(img, interpolation='none', cmap=matplotlib.cm.jet)
            plt.show()
            plt.clf()
            plt.close()

def test_h5_file_keys(h5file):
    with h5py.File(h5file,'r') as f:
        for key in sorted(f.keys()):
            if 'img' in key:
                print(key)
                img = np.array(f.get(key))
                #print('Shape:', img.shape)
                draw_image(img)
            else:
                print(f"{key} --> {np.array(f.get(key))}")
            input("Hit any key to continue ...")
