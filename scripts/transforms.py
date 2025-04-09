import numpy as np

class Crop(object):
    """
    Adapted from https://github.com/Duplums/SMLvsDL. 
    Parameters :
        shape : (tuple or list of int) the shape of the patch to crop
    Aim : 
        Crop the given n-dimensional array centered.
    Output : 
        Cropped array
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, arr):
        assert isinstance(arr, np.ndarray)
        assert type(self.shape) == int or len(self.shape) == len(arr.shape), "Shape of array {} does not match {}".\
            format(arr.shape, self.shape)

        img_shape = np.array(arr.shape) # in our case: img_shape is 1, 121, 145, 121
        size = np.copy(self.shape)
        indexes = []

        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            indexes.append(slice(delta_before, delta_before + size[ndim]))

        return arr[tuple(indexes)]
    
class Padding(object):
    """
    Adapted from https://github.com/Duplums/SMLvsDL. 
    
    Aim : 
        Pad an image.
    Output : 
        Cropped array
    """
        

    def __init__(self, shape, mode):
        """ 
        Parameters :
            shape : (list of int) the desired shape.
        """
        self.shape = shape
        self.mode = mode

    def __call__(self, arr):
        """ 
        Parameters :
            arr : (np.array) input array.
        Aim : 
            call the apply_padding function to pad 'arr' and return the padded the array.
        """
        if len(arr.shape) >= len(self.shape):
            return self._apply_padding(arr)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr):
        """ 
        Parameters :
            arr : (np.array) input array.
        Aim : 
            Pad the array 'arr' with 0.
        Output :
            fill_arr : (np.array) the padded array.
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, mode= self.mode)
        return fill_arr
    
class Normalize(object):
    """
    From https://github.com/Duplums/SMLvsDL.
    Aim : Normalization of an array (an image) so that it has a mean of 0 and a standard deviation of 1. 
    """

    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        """
        Parameters:
            mean : (float) normalized mean of array = 0.
            std : (float) normalized standard deviation of array = 1.
            eps : (float) epsilon value to prevent division by 0. 
                its value is any small value close to 0.
        Aim : 
            Normalize the voxel values within the image 
        Output : 
            Normalized array.
        """
        self.mean=mean
        self.std=std
        self.eps=eps
    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean
