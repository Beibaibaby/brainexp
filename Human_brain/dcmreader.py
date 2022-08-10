import pydicom as dicom
import matplotlib.pylab as plt
import cv2

# specify your image path
image_path = '/Users/dragon/Desktop/brainexp/101_T1.20200915.105517.14.rest_ep2d_bold_MB8_2mm.Echo_1.0001.dcm'
ds = dicom.dcmread(image_path)

pixel_array_numpy = ds.pixel_array
print(pixel_array_numpy.shape)
image_format = '.jpg' # or '.png'
image_path = image_path.replace('.dcm', image_format)

cv2.imwrite(image_path, pixel_array_numpy)