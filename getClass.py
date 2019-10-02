import glob
import numpy as np

labels = {'car': [1,0,0], 'person':[0,1,0], 'bicycle':[0,0,1]}

dataset = np.load('working.npy')
classes = []

def get_data_bbox(dirs):
    for d in dirs:
        batch = []
        bbox = glob.glob(d + 'mask-item-*.bbox')
        for box in bbox:
            dist_file = box.split('.bbox')[0] + '.dist'
            box_contnent = None
            with open(box, 'r') as b:
                box_content = b.read().split('\n')[0].split(',')
            with open(dist_file,encoding='utf8',errors='ignore') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                label = content[-2]
                content = content[:-2]
                if label not in list(labels.keys()):
                    continue
                classes.append(label)

for d in dataset:
    get_data_bbox([d])

classes = list(set(classes))
np.save('classes.npy',classes)
