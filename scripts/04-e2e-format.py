import os
from tqdm import tqdm
from PIL import Image

data_path = "./data/gt_bounding_boxes/"
output_path = "./E2E-MLT/data/sevn/"

for path in tqdm(os.listdir(data_path + 'images'), desc='Formating'):
    frame = path.split('.')[0]
    im = Image.open(data_path + 'images/' + path)
    im.save(output_path + path, "PNG")
    with open(f'{output_path}gt_{frame}.txt', 'w') as f:
        f.write(f'0,0,{im.size[0]},0,{im.size[0]},{im.size[1]},0,{im.size[1]},1,{frame.split("_")[0]}')
    f.close()
