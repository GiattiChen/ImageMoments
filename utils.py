import cv2
import numpy as np
import scipy as sp
import scipy.special
import scipy.misc
import math
import matplotlib.pyplot as plt
from scipy.special import factorial

import os, json
from PIL import Image, ImageDraw, ImageFont,ImageFilter
import PIL.Image
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Example of calling the function: get_glyph_img("上","STHeiti Medium.ttc",pt = 28)
# Return: numpy.ndarray
def get_glyph_img(character, fontname,
    pt = 256, binarize = True,
    descent_offset = True, offset=0):
    """Get image of a certain glyph from the given font
    Parameters
    ----------
    character: str
        The character to draw
    pt: int
        Number of pixels horizontally and vertically
    binarize: boolean
        Return a binary image or not
    fontname: str
        The font file name
    
    Returns
    ----------
    glyph_img: numpy.ndarray
        A greyscale image of the glyph, with its foreground in white (255) and
        background in black (0). 
    """
    width = height = pt
    font = ImageFont.truetype(fontname, pt)  # the size of the char is pt
    
    _, descent = font.getmetrics()
#     start = -descent+offset if descent_offset else 0
    start = offset if descent_offset else 0
    img = PIL.Image.new('L', (pt, pt), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, start), character, 255, font = font)
    img_array = np.array(img, dtype = np.uint8)
    if not binarize: return img_array
    else: return (img_array > 128).astype(np.uint8)
    
#     ascent, descent = font.getmetrics()
#     (font_width, font_height), (offset_x, offset_y) = font.font.getsize(character)
#     img = Image.new('L', (width, height), 0)   # set up the background with size pt*pt
#     draw = ImageDraw.Draw(img)
#     draw.text((width/2-(font_width/2+offset_x),height/2-(font_height/2+offset_y)) ,character, fill=255,font=font)
# #     draw.text(((width-font_width)/2-offset_x/2, (height-font_height)/2-offset_y/2-descent/2), character, fill=255,font=font) # draw character
#     img.show()
#     return img


def descent_offset(pt, fontname,characters):
    # calculate the offset needed to set the character in the middle of the graph
    res = np.zeros(len(characters))
    width = height = pt
    font = ImageFont.truetype(fontname, pt)  # the size of the char is pt
    for i in range (len(characters)):
        (font_width, font_height), (offset_x, offset_y) = font.font.getsize(characters[i])
        res[i] = height/2-(font_height/2+offset_y)
    
    return np.mean(res)

def npz_char(sentence, pt = 256, fontname1 = "Songti.ttc", fontname2 = "STHeiti Medium.ttc"):
    # sentence --> the list of characters
    sentence_train = sentence
    print('The length of the sentence_train is {}'.format(len(sentence_train)))
    
    sentence_train = list(sentence_train)
    # delete same elements in the list
    dic_train={}
    characters_train = list(dic_train.fromkeys(sentence_train).keys())
    # Determine descent_offset for both fonts 
    offset1 = descent_offset(pt,fontname1,characters_train)
    print("For {}, the offset is {}".format(fontname1,offset1))
    offset2 = descent_offset(pt,fontname2,characters_train)
    print("For {}, the offset is {}".format(fontname2,offset2))
    
    # Training set #
    num_chars_train = len(characters_train)
    font1_chars_train = np.zeros((num_chars_train, pt*pt),dtype = np.uint8)  # (num_chars, 784)
    font2_chars_train = np.zeros((num_chars_train, pt*pt),dtype = np.uint8)  # (num_chars, 784)
    
    for i in range(num_chars_train):
        font1_chars_train[i] = get_glyph_img(characters_train[i], fontname1, pt = pt, binarize = True, offset = offset1).flatten()
        font2_chars_train[i] = get_glyph_img(characters_train[i], fontname2, pt = pt, binarize = True, offset = offset2).flatten()
    
    # Valid set #
    # PS: select the Validing Set from Training Set
    idx = np.random.choice(range(len(sentence_train)), size =len(sentence_train)//5, replace=False)
#     idx = np.random.choice(range(2000), size =len(sentence_train)//5, replace=False)
    idx.sort()
    sentence_valid = list(sentence[i] for i in idx)
    print('The length of the sentence_valid is {}'.format(len(sentence_valid)))
    sentence_valid = ''.join(sentence_valid)
    sentence_valid = list(sentence_valid)
#     print(sentence_valid)
    dic_valid={}
    characters_valid = list(dic_valid.fromkeys(sentence_valid).keys()) # delete same elements in the list
    num_chars_valid = len(characters_valid)
    font1_chars_valid = np.zeros((num_chars_valid, pt*pt),dtype = np.uint8)  # (num_chars, 784)
    font2_chars_valid = np.zeros((num_chars_valid, pt*pt),dtype = np.uint8)  # (num_chars, 784)
    
    for i in range(num_chars_valid):
        # .flatten 28*28->784
        font1_chars_valid[i] = get_glyph_img(characters_valid[i], fontname1, pt = pt, binarize = True, offset = offset1).flatten()
        font2_chars_valid[i] = get_glyph_img(characters_valid[i], fontname2, pt = pt, binarize = True, offset = offset2).flatten()
    # Testing set
    # Not used until now
    
    # Save the data
    np.savez('/Users/Giatti/Desktop/AI篆刻/ImageMoments/char_moments_2.npz', \
             X_train = font1_chars_train, y_train = font2_chars_train, \
             X_valid = font1_chars_valid, y_valid = font2_chars_valid)
    return



def i_to_x(i,k):
    return (2*i-k+1)/(k-1)

def j_to_y(j,k):
    return (2*j-k+1)/(k-1)
# the function for calculating H_p_tilde(i,k;sigma)


def discrete_hermite(p,i,sigma,k):
    x = i_to_x(i,k)
    return ((2**p)*factorial(p)*math.sqrt(math.pi)*sigma)**(-1/2)*math.exp(-x**2 / (2*sigma**2))*sp.special.eval_hermite(p, x/sigma)
    
    
def discrete_hermite_Q(n,k,sigma):
    Q = np.zeros((n+1,k), dtype="double")
    for i in range(n+1):
        for j in range(k):
            Q[i,j] = discrete_hermite(i,j,sigma,k)
    return Q

# physicists hermite polynomials
def get_hermite_fast(x, kmax, d):
    H = np.zeros((kmax, len(x)))
    H[0,:] = np.ones(len(x))
    H[1,:] = 2*x
    for k in range(kmax- 2):
        kp = k+2
        H[kp,:] = 2*x*H[kp-1,:] - 2*(kp-1) * H[kp-2,:]
    return H
    
    
def normalization_denominator(i,j,U):
    denominator = 0
    for p in range(n+1):
        for q in range (n+1):
            # equation: denominator += normalization_pq(p,q) * discrete_hermite(p,i,sigma) * discrete_hermite(q,j,sigma)
            denominator += U[p,q] * Q[p,i] * Q[q,j] 
    return denominator


def normalization_pq(p,q,Q):
    # equation: u_pq += discrete_hermite(p,i,sigma) * discrete_hermite(q,j,sigma)
    u_pq = 0
#     method 1
    u_pq = np.sum(Q[p]) * np.sum(Q[q])
#     method 2
#     for i in range (k):
#         for j in range (k):
#             u_pq += Q[p,i] * Q[q,j] 
#     method 3 <-- not accurate: sum should be 0 when p is even, while not actually 
#     sum = 0
#     for i in range (Q.shape[1]):
#         sum+=Q[p,i]
    return u_pq


def normalization(n,Q,img_rec,img_binary):
    # img_binary -> char_B
    U = np.zeros((n+1,n+1), dtype = "double") 
    for p in range(n+1):
        for q in range(n+1):
            U[p,q] = normalization_pq(p,q,Q)
    
    # method 1: using normalization_denominator function: integration by my own function <- slower
    # denominator = np.zeros(img_rec.shape)
    # for i in range (k):
    #     for j in range (k):
    #         denominator[i,j] = normalization_denominator(i,j,U)

    # method 2 <- matrix multiplication
    denominator = np.dot(np.dot(Q.T,U),Q)
    # I_hat = I_tilde / denominator
    img_nor_rec = img_rec / denominator
    
    row, col = img_binary.shape
    black_pixel = int(row*col - img_binary.sum()) # the number of pixels == 0
    flat = img_nor_rec.flatten()
    flat.sort()
    threshold = flat[black_pixel]
    ret2, img_nor_binary_rec = cv2.threshold(img_nor_rec,threshold ,1, cv2.THRESH_BINARY)
    print("阈值是:", ret2, "\norder是:", n)
    return img_nor_binary_rec


def img_recover(n,Q,M,img_binary):
    # img_binary -> char_B
    # M -> char_B's moment
    F_rec = np.dot(np.dot(Q.T,M),Q)
    img_rec = F_rec.T
    img_nor_binary_rec = normalization(n,Q,img_rec,img_binary)
    return img_rec, img_nor_binary_rec

    
    
    
    

