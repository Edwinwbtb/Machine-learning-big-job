import numpy as np
import pandas as pd
import os
from cv2 import cv2


lab2= "./data/dataset/labels/"
picpath= "./data/dataset/images/"
labpath= "./data/dataset/images/val/"
piclist=os.listdir(picpath)
lablist=os.listdir(labpath)

for i in range(len(piclist)):
    picture = piclist[i]
    pre = picture[:-4]
    lab = pre + '.txt'
    picurl= picpath + picture
    laburl= labpath + lab
    laburl2 = lab2 + lab
    img = cv2.imread(picurl)
    df=pd.read_table(laburl,sep=" ",names=["image","class","xmax","ymax","xmin","ymin"])
    df["class"]=0
    df["x"]=(df.xmax+df.xmin)/img.shape[1]/2
    df["y"]=(df.ymax+df.ymin)/img.shape[0]/2
    # w代表width
    df["w"]= abs((df.xmax-df.xmin)/img.shape[1])
    # h代表height
    df["h"]= abs((df.ymax-df.ymin)/img.shape[0])

    df.drop(['image', 'xmax','xmin','ymax','ymin'],axis=1, inplace=True)
    df.to_csv(laburl2, sep=' ', index=False, header=None)
