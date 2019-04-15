import os
import glob
import numpy as np
from natsort import natsorted
import pickle

def rearrange_pickle_files(demo_files, n_folders, n_examples_per_folder):
    demos = {}
    for key in range(n_folders):
      for i in range(n_examples_per_folder):
        idx = key*n_examples_per_folder + i
        with open(demo_files[idx], 'rb') as f:
            demo_temp = pickle.load(f)
            if i == 0:
              demo = demo_temp
              for k in demo.keys():
                demo[k] = np.expand_dims(demo[k],axis=0)
            else:
              for k in demo.keys():
                demo[k] = np.vstack((demo[k],np.expand_dims(demo_temp[k],axis=0)))
        f.close()
        #os.remove(demo_files[idx])
      demos[key] = demo
      with open('data/vision_reach/color_data/demos_' + str(key) + '.pkl' , 'wb') as f:
        pickle.dump(demos[key], f, protocol=2)
      f.close()

demo_gif_dir='data/vision_reach/color_data/'
n_folders = len(os.listdir(demo_gif_dir))
gif_path = demo_gif_dir + 'color_0/*.gif'
n_examples_per_folder = len(glob.glob(gif_path))
demo_files = natsorted(glob.glob('data/vision_reach/color_data/*/*.pkl'))
rearrange_pickle_files(demo_files, n_folders, n_examples_per_folder)
