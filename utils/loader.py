"""
1.Download MS Coco images
2.extract Directory
3.Load Images and create Batches
"""

import os
import numpy as np
import wget
from zipfile import ZipFile as zip

from PIL import Image
import matplotlib.pyplot as plt 
import pycocotools
from pycocotools.coco import COCO

def coco_downloader():
    #train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    #test_url = 'http://images.cocodataset.org/zips/test2017.zip'
    #valid_url = 'http://images.cocodataset.org/zips/val2017.zip'
    #url =[train_url,test_url,valid_url]
    #for urls in url:
    #    wget.download(urls)
    trainData = '../Data/train/'
    testData = '../Data/test/'
    validData = '../Data/valid/'
    trainAnnotations = '../annotations/train/'
    testAnnotations = '../annotations/test/'
    validAnnotations = '../annotations/valid/'
    dir = [trainData,testData,validData,trainAnnotations,testAnnotations,validAnnotations]
    for files in os.listdir(validData):
        print (files)
        zipfile = zip(validData + files + '/')
        zipfile.extractall(validData)
    return None

def categoriseLabels():
    return None

def cocoSegmentationToSegmentationMap(annFile, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''
    # Init
    coco=COCO(annFile)
    try:
        curImg = coco.imgs[imgId]
        imageSize = (curImg['height'], curImg['width'])
        labelMap = np.zeros(imageSize)
        # Get annotations of the current image (may be empty)
        imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
        if includeCrowd:
            annIds = coco.getAnnIds(imgIds=imgId)
        else:
            annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)
        # Combine all annotations of this image in labelMap
        #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
        for a in range(0, len(imgAnnots)):
            labelMask = coco.annToMask(imgAnnots[a]) == 1
            #labelMask = labelMasks[:, :, a] == 1
            newLabel = imgAnnots[a]['category_id']
            if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
                raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))
            labelMap[labelMask] = newLabel
    
    except:
        labelMap=[]
        print ('KeyError', imgId ,'does not exist')
    return labelMap

def getImageID(ImageDirectory):
    ImageID=[]
    ImageNames=[]
    for images in  os.listdir(ImageDirectory):
        ImageNames.append(images)
        print ('images',images)
        partialresult=images.split('.')[0].split('000000')[1]
        print ('trails',images.split('.')[0].strip('0'))
        ImageID.append(images.split('.')[0].strip('0'))
    return ImageID, ImageNames

ImageID,ImageNames = getImageID('../Data/valid/val2017/')

#currentImgID=30785


def coco_annotations2mask(imgId,images,annFile):
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    #catIds = coco.getCatIds(catNms=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'stop', 'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball', 'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis', 'racket', 'bottle', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza', 'donut', 'cake', 'chair', 'couch' 'potted', 'plant', 'bed', 'dining', 'table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'])
    
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds)
    
    print ('ImgIds',imgId)
  
    I =Image.open('../Data/valid/val2017/'+str(images))# 000000030785.jpg
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=imgId)
    try:
        anns = coco.loadAnns(annIds)
        print ('anns',anns)
        coco.showAnns(anns)
        plt.show()
        annIds = coco.getAnnIds(imgIds=imgId,  iscrowd=None)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i]) #we can also use binary or to sum the non-overlapping masks
        plt.imshow(mask)
        plt.show()

    except:
        print ('annotation array is empty',anns)
    return None

#TESTING
#coco_downloader()

dataDir='..'
dataType='val2017'
annFile = '{}/annotations/annotations_trainval2017/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)


#coco_annotations2mask(currentImgID,annFile)


print ('Segmentation MAP')
segmentationMap=cocoSegmentationToSegmentationMap(annFile, 23781, checkUniquePixelLabel=False, includeCrowd=False)
plt.imshow(segmentationMap)
plt.show()
plt.imsave('../mask/valid/val2017/'+ str(23781) + 'output.png',segmentationMap,cmap='gray')

#coco_annotations2mask(30785,'000000030785.jpg',annFile)


#coco_annotations2mask(3501,'000000003501.jpg',annFile)
#coco_annotations2mask(7991,'000000007991.jpg',annFile)


#coco_annotations2mask(29640,'000000029640.jpg',annFile)

coco_annotations2mask(23781,'000000023781.jpg',annFile)

#iterate over all files
#for  idx in range(0,len(ImageID)):
#    print (ImageID[idx],ImageNames[idx])
#    coco_annotations2mask(ImageID[idx],ImageNames[idx],annFile)



