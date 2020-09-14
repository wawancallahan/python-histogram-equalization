import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(img_in):
    # segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
        
    # mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out



# Grayscale
# img = cv2.imread('./wiki.jpg')

# hist,bins = np.histogram(img.flatten(),256,[0,256])

# cdf = hist.cumsum()
# # cdf_normalized = cdf * hist.max()/ cdf.max()

# # plt.plot(cdf_normalized, color = 'b')
# # plt.hist(img.flatten(),256,[0,256], color = 'r')
# # plt.xlim([0,256])
# # plt.legend(('cdf','histogram'), loc = 'upper left')
# # plt.show()

# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# new_cdf = np.ma.filled(cdf_m,0).astype('uint8')

# img2 = new_cdf[img]

# plt.figure(1)
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.hist(img2.flatten(),256,[0,256], color = 'b')
# plt.xlim([0,256])

# cv2.imshow('original', img)
# cv2.imshow('img', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# RGB

img = cv2.imread("sw.jpeg")

# convert image from RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

# convert image back from HSV to RGB
img2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('original', img)
cv2.imshow('img', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
