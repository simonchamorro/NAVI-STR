import math
import os
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
from nfov import NFOV
import imageio as im
import numpy as np
import cv2
from matplotlib import pyplot as plt
import networkx as nx

nfov = NFOV()
images_path = '/home/martin/code/NAVI-STR/data/panos/'
dest = "./PyTorch-YOLOv3/data/sevn/slam/"
coords_file = "/home/martin/code/SEVN/SEVN_gym/data/coord.hdf5"
graph_fname = "/home/martin/code/SEVN/SEVN_gym/data/graph.pkl"

plot = False

coord_df = pd.read_hdf(coords_file, key="df", index=False)
G = nx.read_gpickle(graph_fname)
pos = {k: v.get('coords')[0:2] for k, v in G.nodes(data=True)}
streets = [node for node in G.nodes if node in
           coord_df[coord_df.type == 'street_segment'].index]

# for img in tqdm(os.listdir(images_path), desc="Moving empty images to another folder"):
#     frame = img.split('_')[1].split('.')[0]

    # nx.draw_networkx_nodes(G, pos,
    #                        nodelist=streets,
    #                        node_color='#79d151',
    #                        node_size=1,
    #                        alpha=0.8)
if plot:
    for seg in range(30):
        seg1 = [node for node in G.nodes if node in
            coord_df.loc[coord_df.group == seg].index]
        try:
            idx = seg1[int(len(seg1)/2)]
            plt.text(coord_df.loc[idx].x, coord_df.loc[idx].y, str(seg))
        except Exception as e:
            print(seg)
            print(e)
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=seg1,
                               node_color='#000000',
                               node_size=1,
                               alpha=0.8)
    plt.axis('equal')
    plot_fname = 'colored_graph.png'
    print('hist_fname {}:'.format(plot_fname))
    plt.savefig(plot_fname, transparent=True, dpi=1000)

seg1_num = 20
seg2_num = 21

seg1_coords = coord_df.query(f"type == 'street_segment' & group == {seg1_num} & y > 70 & y < 77")
seg2_coords = coord_df.query(f"type == 'street_segment' & group == {seg2_num} & y > 70 & y < 77")

seg1 = [node for node in G.nodes if node in seg1_coords.index]
seg2 = [node for node in G.nodes if node in seg2_coords.index]

# import pdb; pdb.set_trace()
# nx.draw_networkx_nodes(G, pos,
#                        nodelist=seg1,
#                        node_color='#79d151',
#                        node_size=1,
#                        alpha=0.8)
# nx.draw_networkx_nodes(G, pos,
#                        nodelist=seg2,
#                        node_color='#79d151',
#                        node_size=1,
#                        alpha=0.8)
# plt.axis('equal')
# plot_fname = 'parallel_segs.png'
# plt.savefig(plot_fname, transparent=True, dpi=1000)

center_point = (73.5, 73.5)
image_width = 3840
for img in tqdm(os.listdir(images_path), desc="Moving empty images to another folder"):
    frame = int(img.split('_')[1].split('.')[0])
    if frame in seg1_coords.frame or frame in seg2_coords.frame:
        angle = coord_df.loc[frame].angle
        x = coord_df.loc[frame].x
        y = coord_df.loc[frame].y
        print(x)
        print(y)
        a = np.abs(x - center_point[0])
        o = np.abs(y - center_point[1])
        center_pix = ((math.tan(o / a) / (2*math.pi)) + (angle - 90)/360 * 2 * math.pi) * image_width
        import pdb; pdb.set_trace()

        copyfile(images_path + img, dest + img)

