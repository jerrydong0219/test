import os
import cv2
import numpy as np
import pickle

from contextlib import contextmanager
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler, logProcesses

def init_logger(log_file=None):
    log_file = os.path.join(log_file, 'log.txt')
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def tensor2img_sr(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    # if out_type == np.uint8:
    #     img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np

def read_img(path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img



def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


"""人脸检测>低质人脸过滤 马赛克检测"""
def detect_pixelization(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. canny算法，输出边缘图
    img_canny = cv2.GaussianBlur(img_gray, (3,3), 1.4)
    edges = cv2.Canny(img_canny, 20, 50)

    # 2. 生成匹配模板。包括 T 形模板和 十字形模板
    L = 7
    cross_templates = generate_templates(L)

    # 3. 模板匹配。得到疑似马赛克块的角点。
    #  {'cross':cross, 'top_t':top_t, 'bottom_t':bottom_t, 'left_t':left_t, 'right_t':right_t}
    shifts_for_templates = { 'cross': [0,0], 'top_t':[-(L-1)/2,0], 'bottom_t': [(L-1)/2,0], 'left_t':[0, -(L-1)/2], 'right_t': [0, (L-1)/2] }
    shift_ind = 0
    corner_points = set()

    tmp_edges = edges.copy().astype(np.float32)/255.0
    tmp_inverted_edges = 1.0 - tmp_edges
    for template_name, template in cross_templates.items():
        
        positive_match_results = cv2.filter2D(tmp_edges, ddepth=-1, kernel=template.astype(np.float32))
        negative_match_results = cv2.filter2D(tmp_inverted_edges, ddepth=-1, kernel=1.0-template.astype(np.float32))
        match_results = positive_match_results + negative_match_results

        tolerance_ratio = 0.95
        if template_name == 'cross':
            max_possible_value = (2*L-1)*(2*L-1)
        else:
            max_possible_value = (2*L-1)*L
        threshold = tolerance_ratio* max_possible_value # 通过实验确定的一个经验值
        yy, xx = np.where(match_results >= threshold)

        display = edges.copy()
        for y, x in zip(yy, xx):
            # 注意滤波核的中心不是 角点，要加上一个偏移量。
            corner_x, corner_y = x + int(shifts_for_templates[template_name][1]), y + int(shifts_for_templates[template_name][0])
            corner_points.add((corner_x, corner_y))
            cv2.circle(display, (corner_x, corner_y), radius=5, color=(255,0,0), thickness=1) # +L to move to center
        
        shift_ind += 1

    # 4. 阈值判断。若图像中疑似马赛克的角点的个数超过一定阈值，就认为图像中包含马赛克。
    det_points_num = len(corner_points)
    print("马赛克分数: ", det_points_num)

    if det_points_num >= 10:
        print(">>>> >>> >>> >>> >>> >>> >>> >> 图中包含马赛克！ ")
        return det_points_num
    else:
        # print(">>> 图中不包含马赛克！ ")
        return det_points_num


def generate_templates(L=7):
    """
    生成特定的模板，来检测正方形的角点
    """
    # -----
    #   |
    w, h = 2*L-1, L
    top_t = np.zeros((h,w))
    bottom_t = np.zeros((h,w))

    top_t[0, :] = 1
    top_t[:, L-1] = 1
    bottom_t[-1, :] = 1
    bottom_t[:, L-1]  = 1

    # |
    # |---
    # |

    w, h = L, 2*L-1
    left_t = np.zeros((h,w))
    right_t = np.zeros((h,w))
    left_t[:, 0] = 1
    left_t[L-1, :] = 1
    right_t[:, -1] = 1
    right_t[L-1, :] = 1
    
    #        |
    #     ---|---
    #        |
    w, h = 2*L-1, 2*L-1
    cross = np.zeros((h,w))
    cross[L-1,:] = 1
    cross[:, L-1] = 1

    return {'cross':cross, 'top_t':top_t, 'bottom_t':bottom_t, 'left_t':left_t, 'right_t':right_t}


###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'webp', 'HEIC']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
            # sizes = len(paths)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes
