## Practical Single-Image Super-Resolution Using Look-Up Table

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.html) 


## Dependency
- Python 3.6
- PyTorch 
- glob
- numpy
- pillow
- tqdm
- tensorboardx


## Original Code
   ### 1. Training deep SR network
   ### 2. Transferring to LUT
      The resulting LUT: `./Model_S_x4_4bit_int8.npy`.
      - Title's rule:
      
   > *2. Transferring To LUT*의 결과, LUT가 생성된다. 이때, LUT는 2차원 정수 배열이고, `Model_\*_x\*_\*bit_int*.npy`의 형식으로 저장한다. 
   ### 3. Testing using LUT   
   > *3. Testing using LUT*를 통해, PSNR을 구한다.

## (Experiment1) bit compression


numpy의 타입이 np.int8이므로 정수의 capactiy는 8bit이다. 8bit를 bit compression을 할 때의 PSNR을 관찰한다.
- 8bit와 [1~8]bit로 줄였을 때의 PSNR을 비교한다.
- bit compression의 방법은 Trunc(Floor), Ceil, Round, Random 총 4가지 방식으로 실험하고, 결과를 비교한다.
  - Random의 경우, nearest even rounding 방법으로 구현했다.
 
  ### 실험 결과
  
  <img width="891" alt="image" src="https://github.com/yeonju52/SR-LUT/assets/77441026/8b8fe5ec-95bd-4bb9-8c74-4e02826261fa">



Forked by [original code](https://github.com/yhjo09/SR-LUT)

## BibTeX
```
@InProceedings{jo2021practical,
   author = {Jo, Younghyun and Kim, Seon Joo},
   title = {Practical Single-Image Super-Resolution Using Look-Up Table},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year = {2021}
}
```

