import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import convolve
# from jupyterthemes import jtplot
# jtplot.style(theme='grade3', grid=False, ticks=True, context='paper', figsize=(20, 15), fscale=1.4)


### HELPER FUNCTIONS
## HELP WITH LOADING AND WRITING TO FILE
def load_images(PATH, OUT_PATH=None):
    cap = cv2.VideoCapture(PATH)
    again = True
    i = 0
    imgs = []
    while again:
        again, img = cap.read()
        if again:
            img_r = cv2.resize(img, None, fx=0.25, fy=0.25)
            imgs += [img_r]
            if not OUT_PATH is None:
                filename = OUT_PATH + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
                cv2.imwrite(filename, img_r)
            i += 1
        else:
            break
    return imgs

def create_gif(filenames, PATH):
    kargs = { 'duration': 0.0333 }
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(PATH, images, **kargs)
    
## HELP WITH VISUALIZING 
def imshow_with_trajectory(images, warp_stack, PATH, ij):
    traj_dict = {(0,0):'Width', (0,1):'sin(Theta)', (1,0):'-sin(Theta)', (1,1):'Height', (0,2):'X', (1,2):'Y'}
    i,j = ij
    filenames = []
    for k in range(1,len(warp_stack)):
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})

        a0.axis('off')
        a0.imshow(images[k])

        a1.plot(np.arange(len(warp_stack)), np.cumsum(warp_stack[:,i,j]))
        a1.scatter(k, np.cumsum(warp_stack[:,i,j])[k], c='r', s=100)
        a1.set_xlabel('Frame')
        a1.set_ylabel(traj_dict[ij]+' Trajectory')
        
        if not PATH is None:
            filename = PATH + "".join([str(0)]*(3-len(str(k)))) + str(k) +'.png'
            plt.savefig(filename)
            filenames += [filename]
        plt.close()
    return filenames

def get_border_pads(img_shape, warp_stack):
    maxmin = []
    corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
    warp_prev = np.eye(3)
    for warp in warp_stack:
        warp = np.concatenate([warp, [[0,0,1]]])
        warp = np.matmul(warp, warp_prev)
        warp_invs = np.linalg.inv(warp)
        new_corners = np.matmul(warp_invs, corners)
        xmax,xmin = new_corners[0].max(), new_corners[0].min()
        ymax,ymin = new_corners[1].max(), new_corners[1].min()
        maxmin += [[ymax,xmax], [ymin,xmin]]
        warp_prev = warp.copy()
    maxmin = np.array(maxmin)
    bottom = maxmin[:,0].max()
    print('bottom', maxmin[:,0].argmax()//2)
    top = maxmin[:,0].min()
    print('top', maxmin[:,0].argmin()//2)
    left = maxmin[:,1].min()
    print('right', maxmin[:,1].argmax()//2)
    right = maxmin[:,1].max()
    print('left', maxmin[:,1].argmin()//2)
    return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])

### CORE FUNCTIONS
## FINDING THE TRAJECTORY
def get_homography(img1, img2, motion = cv2.MOTION_EUCLIDEAN):
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix=np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix=np.eye(2, 3, dtype=np.float32)
    warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=motion)[1]
    return warp_matrix 

def create_warp_stack(imgs):
    warp_stack = []
    for i, img in enumerate(imgs[:-1]):
        warp_stack += [get_homography(img, imgs[i+1])]
    return np.array(warp_stack)

def homography_gen(warp_stack):
    H_tot = np.eye(3)
    wsp = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)#[:2]


## DETERMINING THE SMOOTHED TRAJECTORY
def gauss_convolve(trajectory, window, sigma):
    kernel = signal.gaussian(window, std=sigma)
    kernel = kernel/np.sum(kernel)
    return convolve(trajectory, kernel, mode='reflect')

def moving_average(warp_stack, sigma_mat):
    x,y = warp_stack.shape[1:]
    original_trajectory = np.cumsum(warp_stack, axis=0)
    smoothed_trajectory = np.zeros(original_trajectory.shape)
    for i in range(x):
        for j in range(y):
            kernel = signal.gaussian(1000, sigma_mat[i,j])
            kernel = kernel/np.sum(kernel)
            smoothed_trajectory[:,i,j] = convolve(original_trajectory[:,i,j], kernel, mode='reflect')
    smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0,1,-1], mode='reflect'), axis=0, arr=smoothed_trajectory)
    smoothed_warp[:,0,0] = 0
    smoothed_warp[:,1,1] = 0
    return smoothed_warp, smoothed_trajectory, original_trajectory

## APPLYING THE SMOOTHED TRAJECTORY TO THE IMAGES
def apply_warping_fullview(images, warp_stack, PATH=None):
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
    H = homography_gen(warp_stack)
    imgs = []
    for i, img in enumerate(images[1:]):
        H_tot = next(H)+np.array([[0,0,left],[0,0,top],[0,0,0]])
        img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom))
        if not PATH is None:
            filename = PATH + "".join([str(0)]*(3-len(str(i)))) + str(i) +'.png'
            cv2.imwrite(filename, img_warp)
        imgs += [img_warp]
    return imgs
