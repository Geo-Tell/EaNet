import numpy as np
import cv2
def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ])
  
def colormap(pred, num_classes, dataset='cityscapes'):
	if dataset == 'cityscapes':
		color = create_cityscapes_label_colormap()
	else:
		raise Exception('Such dataset doesn\'t have corresponding color list')
	num_colors = len(color)
	if num_colors != num_classes:
		raise Exception('The number of classes doesn\'t match with that of colors')
	img_shape = pred.shape
	canvas = np.zeros([img_shape[0], img_shape[1], 3])
	for i in range(num_classes):
		class_po = (pred == i)
		
		for j in range(3):
			canvas[:,:,j][class_po] = color[i][j]
	return canvas