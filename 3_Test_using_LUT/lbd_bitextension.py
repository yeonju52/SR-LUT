from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm

import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4       # N bit uniform sampling
TEST_DIR = './test/'      # Test images

def compute_LBD(interp_type):
        LUT_PATH = "Model_S_x{}_{}bit_int8_origin.npy".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
        print(">>>>>>>>>>>>>>>>>>>>>.", LUT_PATH.split('.')[0])

        # Load LUT
        LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        # print("Resulting LUT size: ", LUT.shape)

        # print("Resulting min-max = {}~{}".format(LUT.min(), LUT.max()))
        LUT += 128 # 
        h, w = LUT.shape
        # print(LUT)

        # bit compression
        for missing_bits in range(1, 9):
                octave = 2 ** missing_bits # bit 6 : 4
                lbd = (LUT // octave).astype(np.int8)
                if interp_type=="round":
                        lowbit = LUT % octave
                        # chltkddnl qlxmdlf Eo, dhffladlf Eo, wndrksdlf Eo, ghftnfkaus
                        condition = (lbd < 256 / octave - 1) & ((lowbit > octave/2) | ((lowbit == octave/2) & (lbd % 2 == 1)))
                        np.putmask(lbd, condition, lbd+1)
                        np.clip(lbd, 0, 256 / octave - 1)
                print("Resulting min-max = {}~{}".format(lbd.min(), lbd.max()))
                print(lbd)

                np.save("lbd/Model_S_{}_int{}.npy".format(interp_type, 8-missing_bits), lbd)
        
if __name__=='__main__':
    if not isdir('lbd'):
            mkdir('lbd')

#     compute_LBD("random")
    compute_LBD("ceil")

# 7bit: /2 (128)
# 6bit: /2/2 = /4 (64) = 256 / 4 = 64
# 5bit: /2/2/2 = /8 = 256 / 8




