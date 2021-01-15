import os
import pandas as pd
from cv2 import cv2

txtsource = './runs/detect/exp7/labels/'
imgsource = './data/dataset//images/val/'
output = './runs/result.txt'
thre, count = 0.01, 0

with open(output, 'w') as f:
    for root, dirs, files in os.walk(txtsource, topdown=False):
        for txtname in files:
            data = pd.read_csv(txtsource + txtname, sep=' ', header=None, names=['class','x','y','w','h', 'conf'])
            name = txtname.split('.')[0]
            img = cv2.imread(imgsource + name + '.jpg')
            H, W, _ = img.shape
            for idx, row in data.iterrows():
                x, y, w, h, conf = row['x'], row['y'], row['w'], row['h'], row['conf']
                if conf<thre:
                    print('conf:',conf)
                    continue
                x1, y1, x2, y2 = W*(x-w/2), H*(y-h/2), W*(x+w/2), H*(y+h/2)
                line = '%s %.3f %.1f %.1f %.1f %.1f' % (name, conf, x1, y1, x2, y2)
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
                # cv2.imwrite(name+'.jpg', img)
                count += 1
                # print(name, x, y, w, h, conf, H, W)
                print(line)
                f.write(line+'\n')
print('%d results saved in %s' % (count, output))
            # break
