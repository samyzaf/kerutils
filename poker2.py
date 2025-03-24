import matplotlib
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.path as mpath
import numpy as np
import random
import h5py
from ezprogbar import ProgressBar

suit = {
    1: 'h',
    2: 's',
    3: 'd',
    4: 'c',
}

deck = []
for i in range(1,14):
    for j in [1,2,3,4]:
        deck.append(str(i)+suit[j])

hand_name = {
    0: 'Nothing in hand',
    1: 'One pair',
    2: 'Two pairs',
    3: 'Three of a kind',
    4: 'Straight',
    5: 'Flush',
    6: 'Full house',
    7: 'Four of a kind',
    8: 'Straight flush',
    9: 'Royal flush',
}

def write_data(h5file, pokfile):
    print("Writing file:", h5file)
    f = h5py.File(h5file, 'w')
    i = 0
    p = open(pokfile, 'r')
    lines = p.readlines()
    p.close()
    pb = ProgressBar(len(lines))
    for line in lines:
        line = line.strip()
        h = [int(ch) for ch in line.split(',')]
        cls = h.pop()
        img = gen_hand(h)
        a = np.array(h)
        f.create_dataset(f"hnd_{i}", data=a)
        f.create_dataset(f"cls_{i}", data=cls)
        f.create_dataset(f"img_{i}", data=img, compression='gzip')
        i += 1
        pb.advance()
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()

    f.create_dataset('num_images', data=i)
    f.close()
    return h5file

def gen_hand(h, type='RGB'):
    x0, y0 = 5, 5
    dx = 27
    dy = 21
    hand = Image.new(type, (192,192), "aliceblue")
    for i in range(5):
        s = h[2*i] ; r = h[2*i+1]
        c = Image.open("images/" + str(r) + suit[s] + ".gif")
        hand.paste(c, (x0 + i*dx, y0+i*dy))
        c.close()
    img = np.array(hand)
    hand.close()
    return img

def sort_data():
    f = open("pool.csv")
    tab = dict((i,[]) for i in range(10))
    done = set()
    for line in f:
        line = line.strip()
        if line in done:
            continue
        done.add(line)
        hand = [int(i) for i in line.split(',')]
        cls = hand[-1]
        tab[cls].append(hand)
    f.close()
    for i in tab:
        print(i, len(tab[i]))
    #data.extend(tab[9])
    #data.extend(tab[8])
    data = []
    data.extend(tab[7])
    data.extend(tab[6])
    data.extend(tab[5])
    data.extend(tab[4])
    n = 10500
    data.extend(random.sample(tab[3], n))
    data.extend(random.sample(tab[2], n))
    data.extend(random.sample(tab[1], n))
    data.extend(random.sample(tab[0], n))
    print(len(data))

    train = []
    train.extend(tab[8])
    train.extend(tab[9])
    train.extend(random.sample(data, 40000-25))
    test = [h for h in data if not h in train]
    test.extend(tab[8])
    test.extend(tab[9])
    print(len(train))
    print(len(test))
    f = open('train.csv', 'w')
    for h in train:
        line = ','.join([str(i) for i in h]) + '\n'
        f.write(line)
    f.close()
    f = open('test.csv', 'w')
    for h in test:
        line = ','.join([str(i) for i in h]) + '\n'
        f.write(line)
    f.close()

def gen_figs():
    f = open('hands.csv')
    i = 1
    for line in f:
        line = line.strip()
        h = [int(ch) for ch in line.split(',')]
        cls = h.pop()
        img = gen_hand(h)
        title = hand_name[cls]
        plt.title(hand_name[cls], fontsize=27, fontweight='bold', y=1.02)
        ticks=[0,32,64,96,128,160,192]
        plt.xticks(ticks, fontsize=16)
        plt.yticks(ticks, fontsize=16)
        plt.imshow(img, cmap='jet')
        plt.savefig("h%d.png" % i)
        i += 1
    f.close()

def test_h5_file(h5file):
    with h5py.File(h5file,'r') as f:
        num_images = f.get('num_images')[()]
        print("Num images:", num_images)
        for i in range(num_images):
            cls_key = 'cls_' + str(i)
            img_key = 'img_' + str(i)
            hnd_key = 'hnd_' + str(i)
            cls = int(np.array(f.get(cls_key)))
            print(np.array(f.get(cls_key)))
            print(np.array(f.get(hnd_key)))
            img = np.array(f.get(img_key))
            print('Shape:', img.shape)
            title = hand_name[cls]
            plt.title(hand_name[cls], fontsize=24, fontweight='bold', y=1.02)
            ticks=[0,32,64,96,128,160,192]
            plt.xticks(ticks, fontsize=16)
            plt.yticks(ticks, fontsize=16)
            plt.imshow(img, cmap='jet')
            plt.show()
            fig = input("Save figure? (y/n): ")
            if fig == 'y':
                plt.savefig(fig)
            plt.clf()
            plt.close()
