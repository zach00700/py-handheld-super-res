'''
Retrieved from:
https://docs.opencv.org/master/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html

Original author:   Karpushin Vladislav
Compatibility:     OpenCV >= 3.0

python calcGST.py --input ../input/gst_input.jpg
'''

import cv2 as cv
import numpy as np
from numpy import linalg as LA
import argparse
import time

# for demo purposes
W = 35          # window size is WxW
C_Thr = 0.43    # threshold for coherency
LowThr = 35     # lower threshold for orientation, ranging from 0 to 180 degrees
HighThr = 57    # upper threshold for orientation, ranging from 0 to 180 degrees

# Parameters for super resolution
T_s = 16 # [16, 32, 64] px
k_detail = 0.25 # [0.25:0.33] px
k_denoise = 3.0 # [3.0:5.0] 
D_th = 0.001 # [0.001:0.010]
D_tr = 0.006 # [0.006:0.020]
k_stretch = 4
k_shrink = 2
t = 0.12
s1 = 12
s2 = 2
M_th = 0.8 # px



# inputIMG is an image file
def calcGST(inputIMG, w):
    img = inputIMG.astype(np.float32)
    
    # GST components calculation
    # J =  (J11 J12; J12 J22) - GST
    # The Sobel differential is computed by convolving
    # [ -1,  0, +1 ]                    [ -1, -2, -1 ]
    # [ -2,  0, +2 ] (horizontal)   or  [  0,  0,  0 ]    (vertical)
    # [ -1,  0, +1 ]                    [ +1, +2, +1 ]
    # with the image
    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
    imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
    
    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
    # apply a box (i.e. normalized equal-weight) filter to the differential array
    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w,w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w,w))
    J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w,w))
 
    # eigenvalue calculation
    # lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    # lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    start_time = time.time()
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv.multiply(tmp2, tmp2)
    tmp3 = cv.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5*(tmp1 + tmp4)    # biggest eigenvalue
    lambda2 = 0.5*(tmp1 - tmp4)    # smallest eigenvalue
    print("total time for eigenvalues:", time.time() - start_time) 
    
    
    # Compute eigenvectors
    # For a given matrix
    # [ J11 J12 ]
    # [ J12 J22 ]
    # the eigenvector satisfies the equation 
    # (J11 - lambda) * e_x + J12 * e_y = 0
    # and
    # (J12) * e_x + (J22 - lambda) * e_y = 0
    # a possible solution is e = [-J12, (J11 - lambda)] and e = [(J11 - lambda), -J22] 
    # which are equivalent when normalized to length 1.
    # To get e0, we substitute lambda_1; to get e1, we substitute lambda2
    # Lastly, we normalize these curves
    
    sstart_time = time.time()
    e0_X = -J12
    e0_Y = J11 - lambda1
    norm_e0 = np.sqrt(cv.multiply(e0_X, e0_X) + cv.multiply(e0_Y, e0_Y))
    e0_X = cv.divide(e0_X, norm_e0)
    e0_Y = cv.divide(e0_Y, norm_e0)
    
    e1_X = -J12
    e1_Y = J11 - lambda2
    norm_e1 = np.sqrt(cv.multiply(e1_X, e1_X) + cv.multiply(e1_Y, e1_Y))
    e1_X = cv.divide(e1_X, norm_e1)
    e1_Y = cv.divide(e1_Y, norm_e0)
    print("total time for eigenvectors:", time.time() - start_time) 
    
    # Eigenvalue and eigenvector calculation using numpy
    # it is much slower probably due to the matrix building at temp_arr.
    # I just did this to show it could be done
    
    # lambda1_new = np.empty_like(img)
    # lambda2_new = np.empty_like(img)
    
    # start_time = time.time()
    # for i in range(row_size):
        # for j in range(col_size):
            # temp_arr = np.array([ [ J11[i, j], J12[i, j] ], [ J12[i, j], J22[i, j] ] ])
            # lambda_vals, _ = LA.eig(temp_arr)
            
            # # the results of linalg.eig() is not ordered
            # # so use max() and min() to set local lambda functions
            # lambda1_new[i, j] = max(lambda_vals)
            # lambda2_new[i, j] = min(lambda_vals)
    # print("total time:", time.time() - start_time) 
    
    # lambda_diff_new, lambda_sum_new = lambda1_new - lambda2_new, lambda1_new + lambda2_new

    # Coherency calculation.
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    #           = tmp4/tmp1
    # Coherency is anisotropy degree (consistency of local orientation)
    
    
    # since lambda1 - lambda2 = tmp4 and lambda1 + lambda2 = tmp1
    
    # imgCoherencyOut = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
    imgCoherencyOut = cv.divide(tmp4, tmp1)
    
    # np.savetxt('coherency.csv', imgCoherencyOut, delimiter=',')

    # orientation angle calculation
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    imgOrientationOut = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees = True)
    imgOrientationOut = 0.5 * imgOrientationOut
    
    start_time = time.time()
    # sqrt_lambda1 = np.sqrt(lambda1)
    sqrt_lambda1 = np.sqrt(0.5 * (tmp1 + tmp4))
    # sqrt_lambda2 = np.sqrt(lambda2)
    ones_matrix = np.ones_like(img) 
    
    # Compute k1 and k2
    A = ones_matrix + np.sqrt(imgCoherencyOut)
    D = (1 + D_th) * np.ones_like(img) - cv.divide(sqrt_lambda1, D_tr * ones_matrix)
    D[D < 0] = 0
    D[D > 1] = 1
    
    k1_hat = k_detail * k_stretch * A
    k2_hat = cv.divide(k_detail * ones_matrix, k_shrink * A)
    
    k1 = ((ones_matrix - D) * k1_hat + k_detail * k_denoise * D) ** 2
    k2 = ((ones_matrix - D) * k2_hat + k_detail * k_denoise * D) ** 2
    print("Time elapsed to compute k1 and k2:", time.time() - start_time)
    
    # Compute omega
    
    return k1, k2, imgCoherencyOut, imgOrientationOut
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Anisotropic image segmentation tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', required=True)
    args = parser.parse_args()
    imgIn = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    if imgIn is None:
        print('Could not open or find the image: {}'.format(args.input))
        exit(0)
        
    imgk1, imgk2, imgCoherency, imgOrientation= calcGST(imgIn, W)

    # if pixel has coherency and orientation above their respective thresholds, then it belongs to the readout
    _, imgCoherencyBin = cv.threshold(imgCoherency, C_Thr, 255, cv.THRESH_BINARY)
    _, imgOrientationBin = cv.threshold(imgOrientation, LowThr, HighThr, cv.THRESH_BINARY)

    imgBin = cv.bitwise_and(imgCoherencyBin, imgOrientationBin)

    imgCoherency = cv.normalize(imgCoherency, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    imgOrientation = cv.normalize(imgOrientation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    imgk1 = cv.normalize(imgk1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    imgk2 = cv.normalize(imgk2, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    cv.imshow('result.jpg', np.uint8(0.5*(imgIn + imgBin)))
    cv.imshow('Coherency.jpg', imgCoherency)
    cv.imshow('Orientation.jpg', imgOrientation)
    cv.imshow('k1.jpg', imgk1)
    cv.imshow('k2.jpg', imgk2)
    cv.waitKey(0)