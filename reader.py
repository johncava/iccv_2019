import glob
import numpy as np

#directory = './data'
#dirs = glob.glob(directory + '/*/')

def get_data_mask(dirs):
    dataset = []
    for d in dirs:
        batch = []
        masks = glob.glob(d + 'mask-item-*.jpg')
        for mask in masks:
            dist_file = mask.split('.jpg')[0] + '.dist'
            with open(dist_file, 'r') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                content = content[:-2]
                label = content[-1]
                distances = np.array([float(dis) for dis in content])
                #print(np.mean(distances), np.median(distances), np.min(distances))
                batch.append([mask,float(str(np.min(distances)))])
        dataset.append(batch)
    return dataset

labels = {'car': [1,0,0], 'person':[0,1,0], 'bicycle':[0,0,1]}

def get_data_mask_graph(dirs):
    dataset = []
    for d in dirs:
        batch = []
        masks = glob.glob(d + 'mask-item-*.jpg')
        for mask in masks:
            dist_file = mask.split('.jpg')[0] + '.dist'
            box_contnent = None
            with open(dist_file,'r') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                label = content[-2]
                content = content[:-2]
                if label not in list(labels.keys()):
                    continue
                distances = np.array([float(dis) for dis in content])
                batch.append([mask,labels[label],float(str(np.min(distances)))])
        dataset.append(batch)
    return dataset

def get_data_bbox(dirs):
    dataset = []
    for d in dirs:
        batch = []
        bbox = glob.glob(d + 'mask-item-*.bbox')
        for box in bbox:
            dist_file = box.split('.bbox')[0] + '.dist'
            box_contnent = None
            with open(box, 'r') as b:
                box_content = b.read().split('\n')[0].split(',')
            with open(dist_file,'r') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                label = content[-2]
                content = content[:-2]
                if label not in list(labels.keys()):
                    continue
                distances = np.array([float(dis) for dis in content])
                batch.append([[float(bc) for bc in box_content] + labels[label],float(str(np.min(distances)))])
        dataset.append(batch)
    return dataset

def get_data_bbox_graph(dirs):
    dataset = []
    for d in dirs:
        batch = []
        bbox = glob.glob(d + 'mask-item-*.bbox')
        for box in bbox:
            dist_file = box.split('.bbox')[0] + '.dist'
            box_contnent = None
            with open(box, 'r') as b:
                box_content = b.read().split('\n')[0].split(',')
            with open(dist_file,'r') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                label = content[-2]
                content = content[:-2]
                if label not in list(labels.keys()):
                    continue
                distances = np.array([float(dis) for dis in content])
                batch.append([[float(bc) for bc in box_content],labels[label],float(str(np.min(distances)))])
        dataset.append(batch)
    return dataset

def get_data_dis(dirs):                                                        
    dataset = []                                                                
    for d in dirs:
        bbox = glob.glob(d + 'mask-item-*.bbox')
        for box in bbox:
            dist_file = box.split('.bbox')[0] + '.dist'
            box_contnent = None
            with open(box, 'r') as b:
                box_content = b.read().split('\n')[0].split(',')
            with open(dist_file,'r') as f:
                content = f.read().split('\n')
                if len(content) <= 2:
                    continue
                label = content[-2]
                content = content[:-2]
                if label not in list(labels.keys()):
                    continue
                distances = np.array([float(dis) for dis in content])
                dataset.append([[float(bc) for bc in box_content] + labels[label],float(str(np.min(distances)))])                                               
    return dataset
#get_data_bbox()
