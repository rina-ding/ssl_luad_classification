# copied from https://github.com/easonyang1996/CS-CO/blob/9c491c75b7251c996de370b5fd6934eba1675218/data_preprocess/csco_vahadane.py
from glob import glob as glob_function
import os
from natsort import natsorted
import shutil
from itertools import groupby
import random
import math
import shutil

import sys
sys.path.append('./spams-python')
import spams
import cv2
import numpy as np
from PIL import Image
import argparse

IMAGE_SIZE = 224
class vahadane(object):
    
  def __init__(self, STAIN_NUM=2, THRESH=0.9, LAMBDA1=0.01, LAMBDA2=0.01, ITER=100, fast_mode=0, getH_mode=0):
      self.STAIN_NUM = STAIN_NUM
      self.THRESH = THRESH
      self.LAMBDA1 = LAMBDA1
      self.LAMBDA2 = LAMBDA2
      self.ITER = ITER
      self.fast_mode = fast_mode # 0: normal; 1: fast
      self.getH_mode = getH_mode # 0: spams.lasso; 1: pinv;


  def show_config(self):
      print('STAIN_NUM =', self.STAIN_NUM)
      print('THRESH =', self.THRESH)
      print('LAMBDA1 =', self.LAMBDA1)
      print('LAMBDA2 =', self.LAMBDA2)
      print('ITER =', self.ITER)
      print('fast_mode =', self.fast_mode)
      print('getH_mode =', self.getH_mode)


  def getV(self, img):
      
      I0 = img.reshape((-1,3)).T
      I0[I0==0] = 1
      V0 = np.log(255 / I0)

      img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
      mask = img_LAB[:, :, 0] / 255 < self.THRESH
      I = img[mask].reshape((-1, 3)).T
      I[I == 0] = 1
      V = np.log(255 / I)

      return V0, V


  def getW(self, V):
      W = spams.trainDL(np.asfortranarray(V), K=self.STAIN_NUM,
                        lambda1=self.LAMBDA1, iter=self.ITER, mode=2,
                        modeD=0, posAlpha=True, posD=True, verbose=False,
                        numThreads=1)
      W = W / np.linalg.norm(W, axis=0)[None, :]
      if (W[0,0] < W[0,1]):
          W = W[:, [1,0]]
      return W


  def getH(self, V, W):
      if (self.getH_mode == 0):
          H = spams.lasso(np.asfortranarray(V), np.asfortranarray(W), mode=2,
                          lambda1=self.LAMBDA2, pos=True, verbose=False,
                          numThreads=1).toarray()
      elif (self.getH_mode == 1):
          H = np.linalg.pinv(W).dot(V)
          H[H<0] = 0
      else:
          H = 0
      return H


  def stain_separate(self, img, W=None):
      if (self.fast_mode == 0):
          V0, V = self.getV(img)
          if W is None:
              W = self.getW(V)
          H = self.getH(V0, W)
      elif (self.fast_mode == 1):
          m = img.shape[0]
          n = img.shape[1]
          grid_size_m = int(m / 5)
          lenm = int(m / 20)
          grid_size_n = int(n / 5)
          lenn = int(n / 20)
          W = np.zeros((81, 3, self.STAIN_NUM)).astype(np.float64)
          for i in range(0, 4):
              for j in range(0, 4):
                  px = (i + 1) * grid_size_m
                  py = (j + 1) * grid_size_n
                  patch = img[px - lenm : px + lenm, py - lenn: py + lenn, :]
                  V0, V = self.getV(patch)
                  W[i*9+j] = self.getW(V)
          W = np.mean(W, axis=0)
          V0, V = self.getV(img)
          H = self.getH(V0, W)
      return W, H


  def SPCN(self, img, Ws, Hs, Wt, Ht):
      Hs_RM = np.percentile(Hs, 99)
      Ht_RM = np.percentile(Ht, 99)
      Hs_norm = Hs * Ht_RM / Hs_RM
      Vs_norm = np.dot(Wt, Hs_norm)
      Is_norm = 255 * np.exp(-1 * Vs_norm)
      I = Is_norm.T.reshape(img.shape).astype(np.uint8)
      return I

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

def get_HorE(concentration):
    return np.clip(255*np.exp(-1*concentration), 0, 255).reshape(IMAGE_SIZE,
                                                                 IMAGE_SIZE).astype(np.uint8)

def run_stain_separation_E_stain(input_patient_dir_path):
  vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=1,
                   ITER=50)
  all_tiles = sorted(glob_function(os.path.join(input_patient_dir_path, '*.png')))
  dest_path = input_patient_dir_path + '_E_stain'
  if not os.path.exists(dest_path):
    os.mkdir(dest_path)
  for i in range(len(all_tiles)):
    tile_path = all_tiles[i]
    img = read_image(tile_path)
    stain, concen = vhd.stain_separate(img)
    E_ori = get_HorE(concen[1])
    E_ori_png = Image.fromarray(E_ori)
    print(os.path.join(dest_path, os.path.basename(tile_path)))
    E_ori_png.save(os.path.join(dest_path, os.path.basename(tile_path)))

def run_stain_separation_H_stain(input_patient_dir_path):
  vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=1,
                   ITER=50)
  all_tiles = sorted(glob_function(os.path.join(input_patient_dir_path, '*.png')))
  dest_path = input_patient_dir_path + '_H_stain'
  if not os.path.exists(dest_path):
    os.mkdir(dest_path)
  for i in range(len(all_tiles)):
    tile_path = all_tiles[i]
    img = read_image(tile_path)
    stain, concen = vhd.stain_separate(img)
    H_ori = get_HorE(concen[0,:])
    H_ori_png = Image.fromarray(H_ori)
    print(os.path.join(dest_path, os.path.basename(tile_path)))
    H_ori_png.save(os.path.join(dest_path, os.path.basename(tile_path)))

def run_stain_separation_H_E_stain(input_patient_dir_path):
    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=1,
                    ITER=50)
    all_tiles = sorted(glob_function(os.path.join(input_patient_dir_path, '*.png')))
    dest_path_h = input_patient_dir_path + '_H_stain'
    dest_path_e = input_patient_dir_path + '_E_stain'
    if not os.path.exists(dest_path_h):
        os.mkdir(dest_path_h)
    
    if not os.path.exists(dest_path_e):
        os.mkdir(dest_path_e)

    for i in range(len(all_tiles)):
        tile_path = all_tiles[i]
        img = read_image(tile_path)
        stain, concen = vhd.stain_separate(img)
        H_ori = get_HorE(concen[0,:])
        H_ori_png = Image.fromarray(H_ori)
        E_ori = get_HorE(concen[1])
        E_ori_png = Image.fromarray(E_ori)

        print(os.path.join(dest_path_h, os.path.basename(tile_path)))
        H_ori_png.save(os.path.join(dest_path_h, os.path.basename(tile_path)))
        E_ori_png.save(os.path.join(dest_path_e, os.path.basename(tile_path)))

def mask_percent(np_img):
  
  # Determine the percentage of a NumPy array that is completely black or white.
  p = np.percentile(np_img, 90)
  np_img = np.clip(np_img * 255.0 / p, 0, 255).astype(np.uint8)
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    black_percentage = (100 - np.count_nonzero(np_sum) / np_sum.size * 100) 
    white_percentage = ((np.sum(np_sum >= p)) / np_sum.size * 100)
  else:
    black_percentage = (100 - np.count_nonzero(np_img) / np_img.size * 100) 
    white_percentage = ((np.sum(np_img >= p)) / np_img.size * 100)
  return black_percentage, white_percentage

def organize_stain_prediction_data_and_train_val_split(input_dir_he_root, input_dir_h_root, input_dir_e_root, output_dir):
  
  for i in range(len(input_dir_he_root)):
    input_dir_he = input_dir_he_root[i]
    input_dir_h = input_dir_h_root[i]
    input_dir_e = input_dir_e_root[i]
    all_pngs_he = glob_function(os.path.join(input_dir_he, '*.png'))
    all_pngs_h = glob_function(os.path.join(input_dir_h, '*.png'))
    all_pngs_e = glob_function(os.path.join(input_dir_e, '*.png'))
    if len(all_pngs_he) == len(all_pngs_e) and len(all_pngs_he) == len(all_pngs_h):
        print(input_dir_he)
        he_train_path = os.path.join(output_dir, 'train', 'he_stain')
        h_train_path = os.path.join(output_dir, 'train', 'h_stain')
        e_train_path = os.path.join(output_dir, 'train', 'e_stain')
        he_val_path = os.path.join(output_dir, 'val', 'he_stain')
        h_val_path = os.path.join(output_dir, 'val', 'h_stain')
        e_val_path = os.path.join(output_dir, 'val', 'e_stain')

        if not os.path.exists(he_train_path):
            os.makedirs(he_train_path)
        if not os.path.exists(he_val_path):
            os.makedirs(he_val_path)

        if not os.path.exists(h_train_path):
            os.makedirs(h_train_path)
        if not os.path.exists(h_val_path):
            os.makedirs(h_val_path)

        if not os.path.exists(e_train_path):
            os.makedirs(e_train_path)
        if not os.path.exists(e_val_path):
            os.makedirs(e_val_path)

        total_num_pngs = len(glob_function(os.path.join(input_dir_he, '*.png')))

        all_indices = list(range(total_num_pngs))
        train_size = math.floor(total_num_pngs * 0.8)
        train_indices = random.sample(all_indices, train_size)
        val_indices = list(set(all_indices) - set(train_indices))

        train_pngs_he = [all_pngs_he[index] for index in train_indices]
        val_pngs_he = [all_pngs_he[index] for index in val_indices]
        train_pngs_h = [all_pngs_h[index] for index in train_indices]
        val_pngs_h = [all_pngs_h[index] for index in val_indices]
        train_pngs_e = [all_pngs_e[index] for index in train_indices]
        val_pngs_e = [all_pngs_e[index] for index in val_indices]

        for train_png in train_pngs_he:
            shutil.copy(train_png, he_train_path)
        for val_png in val_pngs_he:
            shutil.copy(val_png, he_val_path)

        for train_png in train_pngs_h:
            shutil.copy(train_png, h_train_path)
        for val_png in val_pngs_h:
            shutil.copy(val_png, h_val_path)

        for train_png in train_pngs_e:
            shutil.copy(train_png, e_train_path)
        for val_png in val_pngs_e:
            shutil.copy(val_png, e_val_path)

    print(len(glob_function(os.path.join(output_dir, 'train', 'he_stain', '*'))))
    print(len(glob_function(os.path.join(output_dir, 'val', 'he_stain', '*'))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_input_tiles', type = str, default = None, help = 'path_to_tiles_that_need_to_be_stain_separated')
    parser.add_argument('--path_to_output_tiles', type = str, default = None, help = 'path to the stain separated files prepared for the modeling')

    args = parser.parse_args()

    input_root_path = args.path_to_input_tiles
    input_dir = natsorted(glob_function(os.path.join(input_root_path, '*', 'tiles_png')))
    dest_dir = args.path_to_output_tiles
 
    # for i in range(len(input_dir)):
    #     print(i)
    #     print(input_dir[i])
    #     run_stain_separation_H_E_stain(input_dir[i])
    
    input_dir_he = natsorted(glob_function(os.path.join(input_root_path, '*', 'tiles_png')))
    input_dir_h = natsorted(glob_function(os.path.join(input_root_path, '*', 'tiles_png_H_stain')))
    input_dir_e = natsorted(glob_function(os.path.join(input_root_path, '*', 'tiles_png_E_stain')))
    output_dir = dest_dir
    organize_stain_prediction_data_and_train_val_split(input_dir_he, input_dir_h, input_dir_e, output_dir)
