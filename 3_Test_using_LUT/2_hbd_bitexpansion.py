import numpy as np
from os import mkdir
from os.path import basename, isdir
import shutil



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4       # N bit uniform sampling

def compute_HBD(interp_type):
        # origin copy
        shutil.copy("Model_S_x{}_{}bit_int8_origin.npy".format(UPSCALE, SAMPLING_INTERVAL), "hbd/Model_S_{}_int{}_LUT.npy".format(interp_type, 8-0))
        for missing_bits in range(1, 8): # origin: 1, 8
                LUT_PATH = "lbd/Model_S_{}_int{}_LUT.npy".format(interp_type, 8-missing_bits)    # Trained SR net params        
                print(">>>>>>>>>>>>>>>>>>>>>.", basename(LUT_PATH).split('.')[0])

                # Load LUT
                LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
                print(LUT.shape, "Resulting lbd's min-max = {}~{}".format(LUT.min(), LUT.max()))
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

                print(hbd.shape, "Resulting hbd's min-max = {}~{}".format(hbd.min(), hbd.max()))
                print(hbd)
                np.save("hbd/Model_S_{}_int{}_LUT.npy".format(interp_type, 8-missing_bits), hbd)

if __name__=='__main__':
        if not isdir('hbd'):
                mkdir('hbd')

        compute_HBD("floor")
        compute_HBD("round")
        compute_HBD("ceil")
        # compute_HBD("random")

# lbd : 0 ~ ()
# 8bit: x
# 7bit: 256/2 - 1 = 127
# 6bit: 256/2/2 - 1 = 256/4 - 1 = 256/4  - 1 = 63
# 5bit: 256/2/2/2 - 1 = 256/8 - 1 = 256/8 - 1 = 31
# 4bit: 256/2/2/2/2 - 1 = 256/16 - 1 = 256/16 - 1 = 15

# hbd: -128 ~ ()
# 8bit: original
# 7bit: 255 - 1 - 128 [-1] = 126
# 6bit: 255 - 3 - 128 [-3] = 124
# 5bit: 255 - 3 - 128 [-7] = 120
# 4bit: 255 - 3 - 128 [-15] = 112


