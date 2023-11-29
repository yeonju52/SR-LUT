import numpy as np
from os import mkdir
from os.path import isdir



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4       # N bit uniform sampling

def compute_LBD(interp_type):
        LUT_PATH = "Model_S_x{}_{}bit_int8_origin.npy".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
        print(">>>>>>>>>>>>>>>>>>>>>.", LUT_PATH.split('.')[0])

        LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        
        LUT += 128 # why? numpy float32: [-128, 128] 
        
        # bit compression
        for missing_bits in range(1, 8):
                octave = 2 ** missing_bits
                lbd = (LUT // octave).astype(np.int8)
                if interp_type=="round":
                        lowbit = LUT % octave
                        # 최상위 비트일 때 & (올림일 때 | (중간일 때 & 홀수라면))
                        condition = (lbd < 256 / octave - 1) & ((lowbit > octave/2) | ((lowbit == octave/2) & (lbd % 2 == 1)))
                        np.putmask(lbd, condition, lbd+1)
                        np.clip(lbd, 0, 256 / octave - 1)
                print(LUT.shape, "Resulting min-max = {}~{}".format(lbd.min(), lbd.max()))
                print(lbd)

                np.save("lbd/Model_S_{}_int{}_LUT.npy".format(interp_type, 8-missing_bits), lbd)
        
if __name__=='__main__':
        if not isdir('lbd'):
                mkdir('lbd')
        
        compute_LBD("floor") 
        compute_LBD("round")
        compute_LBD("ceil")
        # compute_LBD("random")


# 7bit: /2 (128)
# 6bit: /2/2 = /4 (64) = 256 / 4 = 64
# 5bit: /2/2/2 = /8 = 256 / 8




