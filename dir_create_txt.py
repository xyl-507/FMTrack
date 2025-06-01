import os
 
img_path = '/public/datasets/VTUAV/testingset/'
img_list=os.listdir(img_path)
img_list.sort()
print('img_list: ',img_list)
 
with open('testingsetList.txt','w') as f:
    for img_name in img_list:
        f.write(img_name+'\n')