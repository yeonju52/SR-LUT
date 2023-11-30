import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

original_dir = 'test/HR'
sr_dir = 'output/output_S_round_int8'
sr_4_dir = 'output/output_S_round_int4'

# 이미지 파일 목록 가져오기
original_files = sorted(os.listdir(original_dir))

plt.figure(figsize=(18,6))
plt.subplots_adjust(hspace=1)

idx = 1
for image_file in original_files:
    org_img_path = os.path.join(original_dir, image_file)
    original_image = cv2.imread(org_img_path, cv2.IMREAD_COLOR)
    sr_img_path = os.path.join(sr_dir,  os.path.splitext(image_file)[0]) + '_LUT_interp_int8.png'
    sr_img_path_4 = os.path.join(sr_4_dir,  os.path.splitext(image_file)[0]) + '_LUT_interp_int4.png'   
    sr_image = cv2.imread(sr_img_path, cv2.IMREAD_COLOR)
    sr_4_image = cv2.imread(sr_img_path_4, cv2.IMREAD_COLOR)

    # 이미지를 YUV 색상 공간으로 변환
    original_yuv = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
    sr_yuv = cv2.cvtColor(sr_image, cv2.COLOR_BGR2YUV)
    sr_4_yuv = cv2.cvtColor(sr_4_image, cv2.COLOR_BGR2YUV)
    
    # Y 채널 추출
    original_y_channel = original_yuv[:, :, 0]
    sr_y_channel = sr_yuv[:, :, 0]
    sr_4_y_channel = sr_4_yuv[:, :, 0]

    psnr_values = []
    psnr_4_values = []

    # Y 채널 값에 따라 PSNR 계산
    for value in range(0, 256):
        sr_y_channel_modified = np.clip(original_y_channel + (value - 128), 0, 255)
        sr_4_y_channel_modified = np.clip(original_y_channel + (value - 128), 0, 255)

        sr_yuv_modified = sr_yuv.copy()
        sr_yuv_modified[:, :, 0] = sr_y_channel_modified
        sr_image_modified = cv2.cvtColor(sr_yuv_modified, cv2.COLOR_YUV2BGR)

        sr_4_yuv_modified = sr_4_yuv.copy()
        sr_4_yuv_modified[:, :, 0] = sr_4_y_channel_modified
        sr_4_image_modified = cv2.cvtColor(sr_4_yuv_modified, cv2.COLOR_YUV2BGR)

        # PSNR 계산
        mse = np.mean((original_image - sr_image_modified) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse))
        psnr_values.append(psnr)

        mse4 = np.mean((original_image - sr_4_image_modified) ** 2)
        psnr_4 = 20 * np.log10(255 / np.sqrt(mse4))
        psnr_4_values.append(psnr_4)

    plt.subplot(1, 5, idx)
    idx += 1

    y_values = list(range(0, 256))

    plt.plot(y_values, psnr_values, color='blue')
    plt.plot(y_values, psnr_4_values, color='red')
    plt.title(image_file.split('_')[0])
    plt.xlabel('Y Channel Value')
    plt.ylabel('PSNR (dB)')
    plt.grid()

plt.savefig('results_Y_48.png')
plt.show()