import cv2
import numpy as np
from scipy.stats import kurtosis, skew
import mahotas as mt
import skimage.morphology 
from elliptic_fourier_descriptors import elliptic_fourier_descriptors

### Preprocessing
def pad_image_to_square(img, addition=0):
    height, width = img.shape[:2]
    if width > height:
        
        dif = width - height
        if dif % 2 == 0:
            top, bottom = dif//2, dif//2
        else:
            top, bottom = dif//2 + 1, dif//2
        #print(top, bottom)
        constant = cv2.copyMakeBorder(img,top + addition,bottom + addition,
                                      addition,addition,
                                      cv2.BORDER_CONSTANT,
                                      value=[255,255,255])

    else:
        dif = height - width
        if dif % 2 == 0:
            left, right = dif//2, dif//2
        else:
            left, right = dif//2 + 1, dif//2
        constant = cv2.copyMakeBorder(img,addition,addition,
                                      left + addition,right + addition,
                                      cv2.BORDER_CONSTANT,
                                      value=[255,255,255])

    return constant

def get_binary_mask(img_gs, return_contour=False):
    _, thresh = cv2.threshold(img_gs, 239, 1, cv2.THRESH_BINARY_INV)

    # only keep biggest contour
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img_gs.shape[:2], np.uint8)
    # hull = cv2.convexHull(biggest_contour)
    cv2.drawContours(mask, [biggest_contour], -1, 1, -1)
    if return_contour:
        return mask, biggest_contour
    return mask

def crop_and_flip_leave(img, mask=None):
    if not mask:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = get_binary_mask(gray)
    # crop
    leave_indices = np.argwhere(mask > 0)
    x1 = np.min(leave_indices[:,0])
    x2 = np.max(leave_indices[:,0]) + 1
    y1 = np.min(leave_indices[:,1])
    y2 = np.max(leave_indices[:,1]) + 1
    
    img =  img[x1:x2, y1:y2, :]
    mask = mask[x1:x2, y1:y2]
    
    # flip
    h, w = img.shape[:2]
    left = np.sum(mask[:,:w//2])
    right = np.sum(mask[:,w//2:])
    if left < right:
        img = cv2.flip(img, 1)
        
    upper = np.sum(mask[:h//2,:])
    lower = np.sum(mask[h//2:,:])
    if upper > lower:
        img = cv2.flip(img, 0)


    return img

def get_rotation_angle(mask):   
    leaf_indices = np.argwhere(mask > 0).astype(np.float32)
    mean = np.mean(leaf_indices, axis=0)
    center, eigenvectors = cv2.PCACompute(leaf_indices, mean=np.asarray([mean]))
    angle = np.arctan(eigenvectors[0,0]/eigenvectors[0,1])  
    angle = angle * 180.0/np.pi 
    return center[0,::-1], angle, eigenvectors



def rotate_img(img):

    ## get rotate angle
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = get_binary_mask(gray)
    if False:
        resized = cv2.resize(mask, (300,300))
        cv2.imshow("a", resized*255)
        cv2.waitKey()
    gray[mask  < 1] = 0
    _, angle, _ = get_rotation_angle(mask)

    ## padding image that prevents leaves from cut out while rotating
    pad_size = int(0.5*max(img.shape[:2]))
    img = cv2.copyMakeBorder(img,pad_size,pad_size,
                            pad_size,pad_size,
                            cv2.BORDER_CONSTANT,
                            value=[255,255,255])
    ## rotate
    height, width = img.shape[:2]
    # rotation_matrix = cv2.getRotationMatrix2D(tuple(center +np.asarray([pad_size,pad_size])) , angle, 1)  # get Rotation Matrix
    rotation_matrix = cv2.getRotationMatrix2D((height//2, width//2) , angle, 1)  # get Rotation Matrix
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=(255, 255, 255))
    
    ## crop the leave patch and pad to square size
    rotated_img = pad_image_to_square(crop_and_flip_leave(rotated_img))

    return rotated_img

###### Handfeature extraction

def get_color_features(leaf, mask):
    bgr = leaf.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    mask = mask > 0
    blue = bgr[:,:,0][mask]
    green = bgr[:,:,1][mask]
    red = bgr[:,:,2][mask]
    hsv_hue = hsv[:,:,0][mask]
    hsv_sat = hsv[:,:,1][mask]
    hsv_val = hsv[:,:,2][mask]
    hls_hue = hls[:,:,0][mask]
    hls_lig = hls[:,:,1][mask]
    hls_sat = hls[:,:,2][mask]

    channels = [blue, green, red, hsv_hue, hsv_sat, hsv_val, hls_hue, hls_lig, hls_sat]
    means = list(map(np.mean, channels))
    stds = list(map(np.std, channels))
    skews = list(map(skew, channels))
    kurtosisses = list(map(kurtosis, channels))
    return means + stds + skews + kurtosisses

def get_shape_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    M = cv2.moments(contour)
    _, _, physiological_width, physiological_height = cv2.boundingRect(contour)
    aspect_ratio = float(physiological_width)/physiological_height
    rectangularity = physiological_width*physiological_height/area
    circularity = ((perimeter)**2)/area
    equi_diameter = np.sqrt(4*area/np.pi)
    form_factor = (4*np.pi*area)/(perimeter**2)
    narrow_factor = equi_diameter/physiological_height
    ratio1 = perimeter/equi_diameter
    ratio2 = perimeter/(physiological_width+physiological_height)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return list(M.values()) + [area,perimeter,physiological_width,physiological_height,aspect_ratio,rectangularity,circularity,equi_diameter,\
                  equi_diameter,form_factor,narrow_factor,ratio1,ratio2,solidity]

def get_vein_image(gray):
    #Preprocessing
    blur = cv2.GaussianBlur(gray, (25,25),0)
    
    r4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    e4 = cv2.erode(blur, r4, iterations=1)
    d4 = cv2.dilate(e4, r4, iterations=1)
    sd4 = blur - d4
    
    return (sd4 > 0).astype(np.uint8)

def get_texture_features(gray):
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)
    return list(ht_mean)

def get_elliptic_fourier_descriptors(mask, N=10):
    efds, _, _ = elliptic_fourier_descriptors(mask, N=N)
    efds = efds[0]
    return list(efds.flatten())

def get_xyprojection(mask, bins=30):
    width = mask.shape[0]//bins
    n_total_pixels = width*mask.shape[0]

    projection = np.empty(shape=(bins*2), dtype=np.float32)
    for i, bin_s in enumerate(range(0, mask.shape[0], width)):
        bin_e = bin_s + width
        projection[i] = np.sum(mask[bin_s:bin_e,:] > 0)
        projection[i+bins] = np.sum(mask[:,bin_s:bin_e] > 0)
    projection = projection/n_total_pixels
    return projection

def get_features(image):
    ## rotate image
    rotated_img = rotate_img(image)
    resized = cv2.resize(rotated_img, (300,300))

    ## extract vein image
    vein = get_vein_image(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY))
    vein = 255*cv2.resize(vein, (300,300))

    ## preparation for handcrafted feature extraction
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask, contour = get_binary_mask(gray, return_contour=True)

    ## color
    color = get_color_features(resized, mask)

    ## shape
    shape = get_shape_features(contour)

    ## texture
    texture = get_texture_features(gray)

    ## fourier
    fourier = get_elliptic_fourier_descriptors(mask, N=10)

    ## xyprojection
    xyprojection = get_xyprojection(mask)

    return resized, vein, color, shape, texture, fourier, xyprojection