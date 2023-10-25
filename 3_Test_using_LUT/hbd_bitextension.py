from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import shutil

import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4       # N bit uniform sampling
TEST_DIR = './test/'      # Test images

def compute_HBD(interp_type):
        # origin copy
        shutil.copy("Model_S_x{}_{}bit_int8_origin.npy".format(UPSCALE, SAMPLING_INTERVAL), "hbd/Model_S_{}_int{}.npy".format(interp_type, 8-0))
        
        for missing_bits in range(1, 9):
                LUT_PATH = "lbd/Model_S_{}_int{}.npy".format(interp_type, 8-missing_bits)    # Trained SR net params        
                print(">>>>>>>>>>>>>>>>>>>>>.", LUT_PATH.split('.')[0])

                # Load LUT
                LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
                print("Resulting lbd's min-max = {}~{}".format(LUT.min(), LUT.max()))
                h, w = LUT.shape

                octave = 2 ** missing_bits # bit 6 : 4
                if interp_type=="random":
                        offset = np.random.randint(0, octave, (h, w)) #2차원 배열
                elif interp_type in ["floor", "round"]:
                       offset = np.zeros((h, w))
                elif interp_type in ["ceil"]: # fill with 1
                       offset = np.full((h, w), octave-1)
                # print(randint) # 255 - 128 = 127, 0 - 128 = -128
                hbd = ((LUT * octave) + offset - 128)#.astype(np.int8)

                print("Resulting hbd's min-max = {}~{}".format(hbd.min(), hbd.max()))
                print(hbd)
                np.save("hbd/Model_S_{}_int{}.npy".format(interp_type, 8-missing_bits), hbd)

# "hbd/Model_S_{}_int{}.npy".format(interp_type, 8-missing_bits)
# 7bit: /2 (128)
# 6bit: /2/2 = /4 (64) = 256 / 4 = 64
# 5bit: /2/2/2 = /8 = 256 / 8

if __name__=='__main__':
    if not isdir('hbd'):
            mkdir('hbd')

#     compute_HBD("random")
    compute_HBD("ceil")




