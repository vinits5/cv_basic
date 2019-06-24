import numpy as np 
import matplotlib.pyplot as plt 
import os

def read_img(filename):
	img = plt.imread(filename)
	return img

def display(img, grayscale):
	if grayscale: plt.imshow(img, cmap='gray')
	else: plt.imshow(img)
	plt.show()

class BasicOps:
	def __init__(self, image):
		if image != None:
			self.define_image(image)

	def define_image(self, image):	
		self.img, self.ROWS, self.COLS, self.CHANNELS = image, image.shape[0], image.shape[1], image.shape[2]		

	# Pixel Transform (Apply Kernels): Image Processing from Szeliski Book. Page No. 101
	def gen_img(self):
		kernel = np.zeros((3,3))
		kernel[1,1]=1
		updated_img = np.zeros(self.img.shape,dtype=int)
		
		for row in range(1,self.ROWS-1):
			for col in range(1,self.COLS-1):
				for channel in range(self.CHANNELS):
					updated_img[row,col,channel]=np.sum(kernel*self.img[row-1:row+2,col-1:col+2,channel])
		return updated_img

	# Pixel Transform (Apply Kernels): Image Processing from Szeliski Book. Page No. 101	
	def apply_kernel_withPadding(self, kernel):
		k_size = kernel.shape[0]
		half_k_size = int(k_size/2)
		
		self.ROWS = img.shape[0]
		self.COLS = img.shape[1]
		self.CHANNELS = img.shape[2]

		padded_img = np.zeros([self.ROWS+k_size-1,self.COLS+k_size-1,self.CHANNELS])
		padded_img[half_k_size:half_k_size+self.ROWS, half_k_size:half_k_size+self.COLS, :] = np.copy(self.img)

		updated_img = np.zeros(img.shape,dtype=int)
		for row in range(half_k_size,self.ROWS+half_k_size):
			for col in range(half_k_size,self.COLS+half_k_size):
				for channel in range(self.CHANNELS):
					updated_img[row-half_k_size, col-half_k_size, channel]=np.sum(kernel*padded_img[row-half_k_size:row+half_k_size+1, col-half_k_size:col+half_k_size+1, channel])
		return updated_img

	# Brighten the image.
	def brighten(self, alpha):
		# Point Operators. Multiple each pixel with alpha value.
		return (alpha*self.img).astype('int')

	# Change the contrast of the image.
	def contrast(self, alpha):
		# Point Operators. Manipulate each pixel as (1-alpha)*128 + alpha*pixel.
		return ((1-alpha)*np.ones(self.img.shape, dtype=np.float64)*128 + alpha*self.img.astype(np.float64)).astype('int')

	# Edge Detection by applying a Kernel.
	def edge_detection(self):
		# Apply following kernel on the image to detect the edge.
		kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
		return self.apply_kernel_withPadding(kernel)

	# Find a gaussian filter of given size and sigma.
	def gaussian_filter(self, size, sigma):
		if sigma == 0: sigma = 1e-07
		g_filter = np.zeros((size,size))
		
		for row in range(-size/2, size/2+1):
			for col in range(-size/2, size/2+1):
				g_filter[row+size/2, col+size/2] = np.exp(-(row*row+col*col)/2*sigma*sigma)
		
		g_filter = (1.0/(2*np.pi*sigma*sigma))*g_filter
		g_filter = g_filter/np.sum(g_filter)
		return g_filter

	# Convolve the image with gaussian filter.
	def apply_gaussian_filter(self, size, sigma):
		assert size>0, 'Filter size should be always greater than 1.'
		if sigma == 0: sigma = 1e-07
		g_filter = self.gaussian_filter(size, sigma)
		return self.apply_kernel_withPadding(g_filter)

	# Sharpen the image.
	def sharpen(self, img):
		# 1. u_img = Apply Gaussian Filter.
		# 2. sharp_img = img + 0.5*(img - u_img)
		updated_img = self.apply_gaussian_filter(3, 2)
		updated_img = (1.5*self.img.astype(np.float64) - 0.5*updated_img.astype(np.float64)).astype('int')
		return updated_img

	# Display Histogram.
	def histogram(self):
		red_channel, green_channel, blue_channel = self.img[:,:,0], self.img[:,:,1], self.img[:,:,2]
		plt.hist(red_channel.flatten(), bins=255, range=(0,255), histtype='step', color='red', label='Red Channel')
		plt.hist(blue_channel.flatten(), bins=255, range=(0,255), histtype='step', color='blue', label='Blue Channel')
		plt.hist(green_channel.flatten(), bins=255, range=(0,255), histtype='step', color='green', label='Green Channel')
		plt.legend(fontsize=15)
		plt.xlim([0,255])
		plt.show()

# Not working properly.
def composite_images(img_top, img_bottom, mask):
	updated_image = np.zeros(img_bottom.shape)
	top_mask = np.copy(mask[:,:,2]).astype(np.float64)
	bottom_mask = np.copy(img_bottom[:,:,2]).astype(np.float64)
	img_top = np.copy(img_top).astype(np.float64)
	img_bottom = np.copy(img_bottom).astype(np.float64)
	for row in range(img_bottom.shape[0]):
		for col in range(img_bottom.shape[1]):
			updated_image[row,col,0] = (top_mask[row,col]*img_top[row,col,0] + bottom_mask[row,col]*img_bottom[row,col,0]*(1-top_mask[row,col])) / (top_mask[row,col]+bottom_mask[row,col]*(1-top_mask[row,col]))
			updated_image[row,col,1] = (top_mask[row,col]*img_top[row,col,0] + bottom_mask[row,col]*img_bottom[row,col,1]*(1-top_mask[row,col])) / (top_mask[row,col]+bottom_mask[row,col]*(1-top_mask[row,col]))
			updated_image[row,col,2] = (top_mask[row,col]*img_top[row,col,0] + bottom_mask[row,col]*img_bottom[row,col,2]*(1-top_mask[row,col])) / (top_mask[row,col]+bottom_mask[row,col]*(1-top_mask[row,col]))
	return updated_image.astype('int')


if __name__=='__main__':
	# ops = BasicOps(None)

	# Apply Kernel
	# img = ops.read_img(os.path.join(os.getcwd(),'input','princeton_small.jpg'))
	# kernel = np.zeros((21,21))
	# kernel[10,10]=1
	# updated_img = ops.apply_kernel_withPadding(img, kernel)

	# Change Brightness
	# img = ops.read_img(os.path.join(os.getcwd(),'input','princeton_small.jpg'))
	# updated_img = ops.brighten(img, 0.2)

	# Change Contrast
	# img = ops.read_img(os.path.join(os.getcwd(),'input','c.jpg'))
	# updated_img = ops.contrast(img, -0.5)

	# Edge Detection
	# img = ops.read_img(os.path.join(os.getcwd(),'input','princeton_small.jpg'))
	# updated_img = ops.edge_detection(img)

	# Apply Gaussian Filter and Blur the image.
	# img = ops.read_img(os.path.join(os.getcwd(),'input','princeton_small.jpg'))
	# updated_img = ops.apply_gaussian_filter(img, 3, 0.1)

	# Sharpen the image
	# img = read_img(os.path.join(os.getcwd(),'input','princeton_small.jpg'))
	# ops.define_image(img)
	# updated_img = ops.sharpen(img)

	# Composite Images.
	# img_top = read_img(os.path.join(os.getcwd(),'input','comp_foreground.jpg'))
	# img_bottom = read_img(os.path.join(os.getcwd(),'input','comp_background.jpg'))
	# mask = read_img(os.path.join(os.getcwd(),'input','comp_mask.jpg'))
	# updated_img = composite_images(img_top, img_bottom, mask)

	# Display Histogram.
	# img = read_img(os.path.join(os.getcwd(),'input','c.jpg'))
	# ops.histogram(img)

	# display(updated_img, False)