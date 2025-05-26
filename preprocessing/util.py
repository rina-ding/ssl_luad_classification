
from glob import glob
import os
from natsort import natsorted
import shutil
from itertools import groupby
import random
import math
import shutil
import pandas as pd
from random import randrange
from operator import itemgetter
from glob import glob
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


def train_val_split(all_dir, train_dir, val_dir):
  random.shuffle(all_dir)
  for i in range(math.floor(len(all_dir)*0.8)):
    shutil.move(all_dir[i], train_dir)
  for i in range(math.floor(len(all_dir)*0.8), len(all_dir)):
    shutil.move(all_dir[i], val_dir)
 
  print('Train sample size ', len(glob(os.path.join(train_dir, '*'))))
  print('Val sample size ', len(glob(os.path.join(val_dir, '*'))))
  

def organize_and_split_ss1_data(wsi_tiles_root_dir, dest_dir):

  dest_dir_train_val = os.path.join(dest_dir, 'train_plus_val_ssl1')
  if not os.path.exists(dest_dir_train_val):
    os.makedirs(dest_dir_train_val)

  wsi_tiles_dirs = sorted(glob(os.path.join(wsi_tiles_root_dir, '*')))

  for i in range(len(wsi_tiles_dirs)):
    print(wsi_tiles_dirs[i])
    all_tiles = natsorted(glob(os.path.join(wsi_tiles_dirs[i], 'low_high_magnification_pairs_png', '*.png')))    
    all_lower_mag_tiles = natsorted(glob(os.path.join(wsi_tiles_dirs[i], 'low_high_magnification_pairs_png', '*lower_mag_tile.png')))

    if len(all_tiles) >=34: # 34 means we have at least 2 lower-magnification tiles. Each lower-magnification tile derives 16 higher-magnification tile. 1+16 = 17. And 17 x 2 = 34
      for j in range(len(all_tiles)):
        if j % 17 == 0: 
          if j <= len(all_tiles) - 17:
            tile_lower_magnification = all_tiles[j]
            tiles_higher_magnification = all_tiles[j+1:j+17] 

            # Positive pairs
            nth_tile = 0
            for tile_higher_magnification in tiles_higher_magnification:
              nth_tile += 1
              positive_pair_dir = os.path.join(dest_dir_train_val, 'yes_' + os.path.basename(wsi_tiles_dirs[i]) + '_' + os.path.basename(tile_lower_magnification).replace('-lower_mag_tile.png', '_' + str(nth_tile)))
            
              if not os.path.exists(positive_pair_dir):
                os.makedirs(positive_pair_dir)
              shutil.copy(tile_lower_magnification, positive_pair_dir)
              shutil.copy(tile_higher_magnification, positive_pair_dir)

            # Negative pairs
            current_lower_mag_tile_index = all_lower_mag_tiles.index(tile_lower_magnification)
            random_negative_lower_mag_tile = all_lower_mag_tiles[random.choice([*range(current_lower_mag_tile_index), *range(current_lower_mag_tile_index + 1, len(all_lower_mag_tiles))])]
            random_negative_lower_mag_tile_index = all_tiles.index(random_negative_lower_mag_tile)
            random_negative_higher_mag_tiles = all_tiles[random_negative_lower_mag_tile_index + 1 : random_negative_lower_mag_tile_index + 17]
            nth_tile = 0
            for negative_tile_higher_mag in random_negative_higher_mag_tiles:
              nth_tile += 1
              negative_pair_dir = os.path.join(dest_dir_train_val, 'no_' + os.path.basename(wsi_tiles_dirs[i]) + '_' + os.path.basename(tile_lower_magnification).replace('-lower_mag_tile.png', '_' + str(nth_tile)))
              
              if not os.path.exists(negative_pair_dir):
                os.makedirs(negative_pair_dir)
              shutil.copy(tile_lower_magnification, negative_pair_dir)
              shutil.copy(negative_tile_higher_mag, negative_pair_dir)
    
    dest_dir_train = os.path.join(dest_dir, 'train_ssl1')
    if not os.path.exists(dest_dir_train):
      os.makedirs(dest_dir_train)
    
    dest_dir_val = os.path.join(dest_dir, 'val_ssl1')
    if not os.path.exists(dest_dir_val):
      os.makedirs(dest_dir_val)

    train_val_split(glob(os.path.join(dest_dir_train_val, '*')), dest_dir_train, dest_dir_val)

def extract_r_c_key(text):
    """
    Extract r and c values from the filename like '...-r1-c15-...'
    and return a string like 'r1-c15' to group by.
    """
    r_match = re.search(r"-r(\d+)", text)
    c_match = re.search(r"-c(\d+)", text)
    if r_match and c_match:
        return f"r{r_match.group(1)}-c{c_match.group(1)}"
    else:
        return "unknown"

def organize_and_split_ss2_data(wsi_tiles_root_dir, dest_dir):
  dest_dir_train_val = os.path.join(dest_dir, 'train_plus_val_ssl2')
  if not os.path.exists(dest_dir_train_val):
    os.makedirs(dest_dir_train_val)

  wsi_tiles_dirs = sorted(glob(os.path.join(wsi_tiles_root_dir, '*')))

  for case in range(len(wsi_tiles_dirs)):
    pid = os.path.basename(wsi_tiles_dirs[case])
    print(wsi_tiles_dirs[case])
    all_tiles = natsorted(glob(os.path.join(wsi_tiles_dirs[case], 'low_high_magnification_pairs_png', '*.png')))    
    all_tiles_basenames = [os.path.basename(x) for x in all_tiles]

    keyf = extract_r_c_key
    groups = []
    for k, g in groupby(all_tiles_basenames, keyf):
      groups.append(natsorted(list(g)))   # Make sure the png names are sorted so that we can automatically assign 1-16 as position 
    
    for group in groups:
      if len(group) == 17: # Make sure there are 16 higher mag tiles from the lower mag tile
        for i in range(len(group)):
          if 'new' not in group[i]:
            lower_mag_tile_path = os.path.join(os.path.dirname(all_tiles[case]), group[i])
        for i in range(len(group)):
          if 'new' in group[i]:
            higher_mag_tile_path = os.path.join(os.path.dirname(all_tiles[case]), group[i])
            # input_dir[case].split('/')[6] is the patient ID
            this_pair_path = os.path.join(dest_dir_train_val, 'p' + str(i) + '_' + pid + '_' + group[i].replace('.png', ''))

            if not os.path.exists(this_pair_path):
              os.mkdir(this_pair_path)
            shutil.copy(lower_mag_tile_path, this_pair_path)
            shutil.copy(higher_mag_tile_path, this_pair_path)
      
    dest_dir_train = os.path.join(dest_dir, 'train_ssl2')
    if not os.path.exists(dest_dir_train):
      os.makedirs(dest_dir_train)
    
    dest_dir_val = os.path.join(dest_dir, 'val_ssl2')
    if not os.path.exists(dest_dir_val):
      os.makedirs(dest_dir_val)

    train_val_split(glob(os.path.join(dest_dir_train_val, '*')), dest_dir_train, dest_dir_val)


def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  t = Time()
  rgb = np.asarray(pil_img)
  np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.

  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  result.show()


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  t = Time()
  result = rgb * np.dstack([mask, mask, mask])
  np_info(result, "Mask RGB", t.elapsed())
  return result


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
