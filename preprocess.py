import numpy as np
import os
import cv2

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }


# do the multi-hot encoding
def multihot_encode(x, num_classes):
    encoded = np.zeros(num_classes)
    encoded[np.array(x)] = 1.0

    return encoded


# preprocess the image
def image_preprocess(path, id):
    # the image goes in RGBY (4 channels)
    # read 4 channels and normalize those
    print(path)
    print(id)
    colors = ('red', 'green', 'blue', 'yellow')
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.tif'), flags).astype(np.float32) / 255 for color in
           colors]

    return np.stack(img, axis=-1)