import numpy as np
from os import mkdir
from os.path import isdir, basename


# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 4       # N bit uniform sampling
TEST_DIR = './test/'      # Test images

def compute_LowBit():
        LUT_PATH = "./Model_S_x{}_{}bit_int8_origin.npy".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
        print(">>>>>>>>>>>>>>>>>>>>>.", basename(LUT_PATH).split('.')[0])
        LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        
        LUT += 128 # float32 > [-128:128]
        
        # bit compression
        for missing_bits in range(1, 8):
                octave = 2 ** missing_bits # bit 6 : 4
                lowbit = (LUT % octave).astype(np.int8)
                print("({}) lowbit min-max = {}~{}".format(missing_bits, lowbit.min(), lowbit.max()))
                np.save("bypass/low_bit/{}bit_missing_{}bit_LUT.npy".format(missing_bits, 8-missing_bits), lowbit)

def compare_HBDwithOrigin(interp_type):
        for missing_bits in range(1, 8):
                LUT_PATH = "lbd/Model_S_{}_int{}.npy".format(interp_type, 8-missing_bits)    # Trained SR net params        
                LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)

                octave = 2 ** missing_bits
                LOW_PATH = "bypass/low_bit/{}bit_missing_{}bit_LUT.npy".format(missing_bits, 8-missing_bits)
                offset = np.load(LOW_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE) # np.zeros((h, w))
                print("missing {} bits, ({}) LBD(0 - 2**{}-1): {} - {}".format(missing_bits, 8-missing_bits, missing_bits, offset.min(), offset.max()))
                print("missing {} bits, ({}) HBD(0 - 2**{}-1): {} - {}".format(missing_bits, 8-missing_bits, 8-missing_bits, LUT.min(), LUT.max()))
                restore_bits = ((LUT * octave) + offset - 128)#.astype(np.int8)
                # np.save("hbd/bypass_{}_int{}.npy".format(interp_type, 8-missing_bits), restore_bits)

                bit8_PATH = "hbd/Model_S_{}_int{}.npy".format(interp_type, 8)
                origin = np.load(bit8_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)

                bypass = origin-restore_bits
                print("I want to get All zero 2d arr: ", np.count_nonzero(bypass))

if __name__=='__main__':
        if not isdir('bypass/low_bit'):
                mkdir('bypass/low_bit')
        compute_LowBit()
        print("========"*6)
        compare_HBDwithOrigin("floor") # bypass: (round) x, (floor, ceil) o
