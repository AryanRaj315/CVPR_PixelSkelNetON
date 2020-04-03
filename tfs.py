import random
import cv2
import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask = None):
        img = image
        if mask is not None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        else:
            for t in self.transforms:
                img = t(img)
            return img

class Shift:
    def __init__(self, prob = 0.5, size = 256):
        self.prob = prob
        self.size = size

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            
            h, w = img.shape[:2]
            rows, cols,_ = np.nonzero(img)
            x = np.random.randint(-rows.min(), 256-rows.max())
            y = np.random.randint(-cols.min(), 256-cols.max())
            transformationMatrix = np.float32([[1, 0, int(x)], [0, 1, int(y)]]) 
            (rows_img, cols_img) = img.shape[:2] 
            img = cv2.warpAffine(img, transformationMatrix, (cols_img, rows_img))
            mask = cv2.warpAffine(mask, transformationMatrix, (cols_img, rows_img))
        return img, mask

class ToTensor:
    def __init__(self):
        self.prob=1

    def __call__(self, image, mask = None):
        img = image
        img = torch.from_numpy(np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        if mask is not None:
            mask = np.expand_dims(mask / (255.0 if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
            return img, torch.from_numpy(mask)
        else:
            return img
    
class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)

        self.height = size[0]
        self.width = size[1]

    def __call__(self, image, mask):
        img = image
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]
        return img, mask

class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            img =cv2.flip(img, 1)
            mask =cv2.flip(mask, 1)
        return img, mask

class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        return img, mask
    
class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
          
            height, width = img.shape[:2]
            image_center = (width/2, height/2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

            abs_cos = abs(rotation_mat[0,0])
            abs_sin = abs(rotation_mat[0,1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]

            img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
            mask = cv2.warpAffine(mask, rotation_mat, (bound_w, bound_h))
        return img, mask

class RotateAndCenterCrop:
    def __init__(self, limit=90, prob=0.5):
        self.prob=prob
        self.limit=limit

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[:2]
            image_center = (width/2, height/2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

            abs_cos = abs(rotation_mat[0,0])
            abs_sin = abs(rotation_mat[0,1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]

            img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
            mask = cv2.warpAffine(mask, rotation_mat, (bound_w, bound_h))

            h, w = img.shape[:2]
            dy = (h - height) // 2
            dx = (w - width) // 2

            y1 = dy
            y2 = y1 + height
            x1 = dx
            x2 = x1 + width
            img = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
        return img, mask
    
class Normalize:
    def __init__(self, mean, std):
        self.mean=mean
        self.std=std
            
    def __call__(self, image, mask):
        img = image
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        max_pixel_value_img = img.max()
        max_pixel_value_mask = img.max()
        img = img/max_pixel_value_img 
        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std

        mask = mask/max_pixel_value_mask
        return img, mask

class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class Recenter:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        img = image
        if random.random() < self.prob:
            ori_h, ori_w = img.shape[:2]
#             print("Inside Recenter")
#             print(img.shape, mask.shape)
            mask_cr = mask[np.ix_(img.any(1)[:,0],img.any(0)[:,0])]
            img_cr = img[np.ix_(img.any(1)[:,0],img.any(0)[:,0])]
            crop_h, crop_w = img_cr.shape[:2]
            img = np.zeros_like(img)
            mask = np.zeros_like(img[:,:,0])
#             print(mask.shape, img.shape)
            img[(ori_h-crop_h)//2:(ori_h+crop_h)//2, (ori_w-crop_w)//2:(ori_w+crop_w)//2] = img_cr
            mask[(ori_h-crop_h)//2:(ori_h+crop_h)//2, (ori_w-crop_w)//2:(ori_w+crop_w)//2] = mask_cr
        return img, mask