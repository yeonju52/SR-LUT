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

for missing_bits in [1, 2, 3, 4]:
        LUT_PATH = "lbd/Model_S_x{}_{}bit_int{}.npy".format(UPSCALE, SAMPLING_INTERVAL, 8-missing_bits)    # Trained SR net params        
        print(">>>>>>>>>>>>>>>>>>>>>.", LUT_PATH.split('.')[0])

        # Load LUT
        LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        print("Resulting lbd's min-max = {}~{}".format(LUT.min(), LUT.max()))
        h, w = LUT.shape

        octave = 2 ** missing_bits # bit 6 : 4
        randint = np.random.randint(0, octave, (h, w)) #2차원 배열
        # print(randint) # 255 - 128 = 127, 0 - 128 = -128
        hbd = ((LUT * octave) + randint - 128)#.astype(np.int8)

        print("Resulting hbd's min-max = {}~{}".format(hbd.min(), hbd.max()))
        print(hbd)
        np.save("hbd/Model_S_x{}_{}bit_int{}".format(UPSCALE, SAMPLING_INTERVAL, 8-missing_bits), hbd)

# 7bit: /2 (128)
# 6bit: /2/2 = /4 (64) = 256 / 4 = 64
# 5bit: /2/2/2 = /8 = 256 / 8




