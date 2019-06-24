 // Source file for image class



// Include files 
#include "R2/R2.h"
#include "R2Pixel.h"
#include "R2Image.h"




////////////////////////////////////////////////////////////////////////
// Constructors/Destructors
////////////////////////////////////////////////////////////////////////


R2Image::
R2Image(void)
	: pixels(NULL),
		npixels(0),
		width(0), 
		height(0)
{
}



R2Image::
R2Image(const char *filename)
	: pixels(NULL),
		npixels(0),
		width(0), 
		height(0)
{
	// Read image
	Read(filename);
}



R2Image::
R2Image(int width, int height)
	: pixels(NULL),
		npixels(width * height),
		width(width), 
		height(height)
{
	// Allocate pixels
	pixels = new R2Pixel [ npixels ];
	assert(pixels);
}



R2Image::
R2Image(int width, int height, const R2Pixel *p)
	: pixels(NULL),
		npixels(width * height),
		width(width), 
		height(height)
{
	// Allocate pixels
	pixels = new R2Pixel [ npixels ];
	assert(pixels);

	// Copy pixels 
	for (int i = 0; i < npixels; i++) 
		pixels[i] = p[i];
}



R2Image::
R2Image(const R2Image& image)
	: pixels(NULL),
		npixels(image.npixels),
		width(image.width), 
		height(image.height)
		
{
	// Allocate pixels
	pixels = new R2Pixel [ npixels ];
	assert(pixels);

	// Copy pixels 
	for (int i = 0; i < npixels; i++) 
		pixels[i] = image.pixels[i];
}



R2Image::
~R2Image(void)
{
	// Free image pixels
	if (pixels) delete [] pixels;
}



R2Image& R2Image::
operator=(const R2Image& image)
{
	// Delete previous pixels
	if (pixels) { delete [] pixels; pixels = NULL; }

	// Reset width and height
	npixels = image.npixels;
	width = image.width;
	height = image.height;

	// Allocate new pixels
	pixels = new R2Pixel [ npixels ];
	assert(pixels);

	// Copy pixels 
	for (int i = 0; i < npixels; i++) 
		pixels[i] = image.pixels[i];

	// Return image
	return *this;
}



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static double 
RandomNumber(void) 
{
#if defined(_WIN32)
	int r1 = rand();
	double r2 = ((double) rand()) / ((double) (RAND_MAX + 1));
	return (r1 + r2) / ((double) (RAND_MAX + 1));
#else
	return drand48();
#endif
}



////////////////////////////////////////////////////////////////////////
// Image processing functions
// YOU IMPLEMENT THE FUNCTIONS IN THIS SECTION
////////////////////////////////////////////////////////////////////////

// Luminance Operations ////////////////////////////////////////////////

void R2Image::
AddNoise(double magnitude)
{
	// Add noise to an image.  The amount of noise is given by the magnitude
	// in the range [0.0..1.0].  0.0 adds no noise.  1.0 adds a lot of noise.

#if 1
	// This implementation is provided as an example of one way to manipulate pixels
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			R2Pixel& pixel = Pixel(i, j);
			pixel[0] += magnitude * (RandomNumber() - 0.5);
			pixel[1] += magnitude * (RandomNumber() - 0.5);
			pixel[2] += magnitude * (RandomNumber() - 0.5);
			pixel.Clamp();
		}
	}
#else
	// This implementation is provided as an example of another way to manipulate pixels
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			R2Pixel pixel = Pixel(i, j);
			pixel.SetRed(pixel.Red() + magnitude * (RandomNumber() - 0.5));
			pixel.SetGreen(pixel.Green() + magnitude * (RandomNumber() - 0.5));
			pixel.SetBlue(pixel.Blue() + magnitude * (RandomNumber() - 0.5));
			pixel.Clamp();
			SetPixel(i, j, pixel);
		}
	}
#endif
}



void R2Image::
Brighten(double factor)
{
	// Brighten the image by multiplying each pixel component by the factor,
	// then clamping the result to a valid range.

	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	for (int j=0; j<height; j++){
		for (int i=0; i<width; i++){
			R2Pixel& pixel = Pixel(i,j);
			pixel[0] *= factor;				// Multiply each pixel by factor to brighten
			pixel[1] *= factor;
			pixel[2] *= factor;
			pixel.Clamp();
		}
	}
	// fprintf(stderr, "Brighten(%g) not implemented\n", factor);
}



void R2Image::
ChangeContrast(double factor)
{
	// Change the contrast of an image by interpolating between the image
	// and a constant gray image with the average luminance.
	// Interpolation reduces constrast, extrapolation boosts constrast,
	// and negative factors generate inverted images.
	for (int j=0; j<height; j++){
		for (int i=0; i<width; i++){
			R2Pixel& pixel = Pixel(i,j);
			// Change contrast by interpolation with gray scale image.
			pixel[0] = (1-factor)*(128/255.0) + factor*pixel[0];
			pixel[1] = (1-factor)*(128/255.0) + factor*pixel[1];
			pixel[2] = (1-factor)*(128/255.0) + factor*pixel[2];
			pixel.Clamp();
		}
	}
	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "ChangeContrast(%g) not implemented\n", factor);
}


// Convolution Function (with Padding)
void R2Image::Convolve(int* width_idx, int* height_idx, int filter_size, double* filter){
	double value_0=0, value_1=0, value_2=0;
	int filter_idx = 0;
	for (int j=*height_idx-filter_size/2; j<=*height_idx+filter_size/2; j++){
		for (int i=*width_idx-filter_size/2; i<=*width_idx+filter_size/2; i++){
			int pixel_i=i, pixel_j=j;
			// Condition to pad the pixels.
			if (pixel_i<0 || pixel_j<0 || pixel_i>=width || pixel_j>=height){
				if(pixel_i<0){pixel_i=0;}
				if(pixel_i>=width){pixel_i=width-1;}
				if(pixel_j<0){pixel_j=0;}
				if(pixel_j>=height){pixel_j=height-1;}
			}
			// Convolution operation.
			R2Pixel& pixel = Pixel(pixel_i,pixel_j);
			value_0 += pixel[0]*filter[filter_idx];
			value_1 += pixel[1]*filter[filter_idx];
			value_2 += pixel[2]*filter[filter_idx];
			filter_idx++;
		}
	}
	// Update the value of pixels.
	R2Pixel& pixel = Pixel(*width_idx, *height_idx);
	pixel[0] = value_0;
	pixel[1] = value_1;
	pixel[2] = value_2;
}

// Function to calculate Gaussian Filter.
// Argument:
	// filter_size: 	size of filter.
	// sigma:			standard deviation of gaussian.
double* R2Image::GaussianFilter(int filter_size, double* sigma){
	double filter[filter_size*filter_size];			// Array to store 2D Gaussian Filter.

	// Loop to calculate values of Gaussian Kernel.
	int filter_idx = 0;
	double filter_sum = 0;
	for(int j=-filter_size/2; j<(filter_size/2)+1; j++){
		for(int i=-filter_size/2; i<(filter_size/2)+1; i++){
			double value = (1/(2*PI*(*sigma)*(*sigma))) * exp(-(i*i+j*j)/(2*(*sigma)*(*sigma)));
			filter[filter_idx] = value;
			filter_sum += value;
			filter_idx++;
		}
	}

	// Divide the kernel by sum of all elements.
	for(int i=0; i<filter_size*filter_size; i++){
		filter[i] = filter[i]/filter_sum;
	}
	return filter;
}

// Linear filtering ////////////////////////////////////////////////

void R2Image::
Blur(double sigma)
{
	// Blur an image with a Gaussian filter with a given sigma.
	const int filter_size = ceil(3*sigma)*2+1;		// Define size of filter.

	// Find Gaussian Filter.
	double* filter = GaussianFilter(filter_size, &sigma);

	// Convolve filter over image.
	for(int j=0; j<height; j++){
		for(int i=0; i<width; i++){
			Convolve(&i, &j, filter_size, filter);
		}
	}
	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "Blur(%g) not implemented\n", sigma);
}



void R2Image::
Sharpen()
{
	// Sharpen an image using a linear filter
	// Ref: https://web.stanford.edu/class/cs448f/lectures/2.1/Sharpening.pdf

	double* img_updated = (double*) calloc(width*height*3,sizeof(double));
	int filter_size = 3;
	double sigma = 2;
	double* filter = GaussianFilter(filter_size, &sigma);

	// Apply Gaussian Filter with size 3.
	int counter = 0;
	for(int j=1; j<height-1; j++){
		for(int i=1; i<width-1; i++){
			// Convolve(&i, &j, filter_size, filter);
			R2Pixel& pixel0 = Pixel(i-1,j-1);
			R2Pixel& pixel1 = Pixel(i,j-1);
			R2Pixel& pixel2 = Pixel(i+1,j-1);
			R2Pixel& pixel3 = Pixel(i-1,j);
			R2Pixel& pixel4 = Pixel(i,j);
			R2Pixel& pixel5 = Pixel(i+1,j);
			R2Pixel& pixel6 = Pixel(i-1,j+1);
			R2Pixel& pixel7 = Pixel(i,j+1);
			R2Pixel& pixel8 = Pixel(i+1,j+1);

			img_updated[counter] = filter[0]*(pixel0[0])+filter[1]*(pixel1[0])+filter[2]*(pixel2[0])+filter[3]*(pixel3[0])+filter[4]*(pixel4[0])+filter[5]*(pixel5[0])+filter[6]*(pixel6[0])+filter[7]*(pixel7[0])+filter[8]*(pixel8[0]);
			img_updated[counter+1] = filter[0]*(pixel0[1])+filter[1]*(pixel1[1])+filter[2]*(pixel2[1])+filter[3]*(pixel3[1])+filter[4]*(pixel4[1])+filter[5]*(pixel5[1])+filter[6]*(pixel6[1])+filter[7]*(pixel7[1])+filter[8]*(pixel8[1]);
			img_updated[counter+2] = filter[0]*(pixel0[2])+filter[1]*(pixel1[2])+filter[2]*(pixel2[2])+filter[3]*(pixel3[2])+filter[4]*(pixel4[2])+filter[5]*(pixel5[2])+filter[6]*(pixel6[2])+filter[7]*(pixel7[2])+filter[8]*(pixel8[2]);
			counter = counter+3;
		}
	}

	// Sharpen the image.
	counter = 0;
	for(int j=1; j<height-1; j++){
		for(int i=1; i<width-1; i++){
			R2Pixel& pixel = Pixel(i,j);
			pixel[0] += (pixel[0]-img_updated[counter])*0.5;
			pixel[1] += (pixel[1]-img_updated[counter+1])*0.5;
			pixel[2] += (pixel[2]-img_updated[counter+2])*0.5;
			counter = counter+3;
		}
	}
	free(img_updated);
	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "Sharpen() not implemented\n");
}



void R2Image::
EdgeDetect(void)
{
	// Detect edges in an image.
	// Convolution over image. [haven't considered zero padding]
	// Updated values of pixels with edges.
	double* img_updated = (double*) calloc(width*height*3,sizeof(double));

	// Applied [[-1,-1,-1],
			//  [-1, 8,-1],
			//  [-1,-1,-1]] filter to find edges.
	int counter = 0;
	for(int j=1; j<height-1; j++){
		for(int i=1; i<width-1; i++){
			R2Pixel& pixel0 = Pixel(i-1,j-1);
			R2Pixel& pixel1 = Pixel(i,j-1);
			R2Pixel& pixel2 = Pixel(i+1,j-1);
			R2Pixel& pixel3 = Pixel(i-1,j);
			R2Pixel& pixel4 = Pixel(i,j);
			R2Pixel& pixel5 = Pixel(i+1,j);
			R2Pixel& pixel6 = Pixel(i-1,j+1);
			R2Pixel& pixel7 = Pixel(i,j+1);
			R2Pixel& pixel8 = Pixel(i+1,j+1);

			img_updated[counter] = -(pixel0[0])-(pixel1[0])-(pixel2[0])-(pixel3[0])+8*(pixel4[0])-(pixel5[0])-(pixel6[0])-(pixel7[0])-(pixel8[0]);
			img_updated[counter+1] = -(pixel0[1])-(pixel1[1])-(pixel2[1])-(pixel3[1])+8*(pixel4[1])-(pixel5[1])-(pixel6[1])-(pixel7[1])-(pixel8[1]);
			img_updated[counter+2] = -(pixel0[2])-(pixel1[2])-(pixel2[2])-(pixel3[2])+8*(pixel4[2])-(pixel5[2])-(pixel6[2])-(pixel7[2])-(pixel8[2]);
			counter = counter+3;
		}
	}

	// Update the image with absolute values obtained from convolution.
	counter = 0;
	for(int j=1; j<height-1; j++){
		for(int i=1; i<width-1; i++){
			R2Pixel& pixel = Pixel(i,j);
			pixel[0] = abs(img_updated[counter]);
			pixel[1] = abs(img_updated[counter+1]);
			pixel[2] = abs(img_updated[counter+2]);
			counter = counter+3;
		}
	}

	// free the memory address.
	free(img_updated);
	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "EdgeDetect() implemented\n");
}

// Resampling operations  ////////////////////////////////////////////////

void R2Image::
Scale(double sx, double sy, int sampling_method)
{
	// Scale an image in x by sx, and y by sy.
	int new_width = width*sx, new_height = height*sy;
	R2Image new_image(new_width, new_height);

	// Gaussian Sampling:
		// Gaussian blur (3x3 filter) + Sampling
	if(sampling_method==R2_IMAGE_GAUSSIAN_SAMPLING){
		int filter_size=3; double sigma = 1/sx;
		double* filter = GaussianFilter(filter_size, &sigma);
		for(int j=0; j<height; j++){
			for(int i=0; i<width; i++){
				Convolve(&i, &j, filter_size, filter);
			}
		}
		// Pixel Sampling
		for(int j=0; j<new_height; j++){
			for(int i=0; i<new_width; i++){
				R2Pixel& pixel = new_image.Pixel(i,j);
				R2Pixel& pixel1 = Pixel(round(i/sx),round(j/sy));
				pixel[0] = pixel1[0];
				pixel[1] = pixel1[1];
				pixel[2] = pixel1[2];
			}
		}
		*this = new_image;
	}
	// Point Sampling
	else if(sampling_method==R2_IMAGE_POINT_SAMPLING){
		for(int j=0; j<new_height; j++){
			for(int i=0; i<new_width; i++){
				R2Pixel& pixel = new_image.Pixel(i,j);
				R2Pixel& pixel1 = Pixel(round(i/sx),round(j/sy));
				pixel[0] = pixel1[0];
				pixel[1] = pixel1[1];
				pixel[2] = pixel1[2];
			}
		}
		*this = new_image;
	}
	// Bilinear Sampling
	else if(sampling_method==R2_IMAGE_BILINEAR_SAMPLING) {
		for(int j=0; j<new_height; j++){
			for(int i=0; i<new_width; i++){
				double x_i=i/sx, y_j=j/sy;
				int pixel_i = round(x_i), pixel_j = round(y_j);
				
				// Find four nearest pixels.
				R2Pixel& pixel1 = Pixel(pixel_i,pixel_j);
				R2Pixel& pixel2 = Pixel(pixel_i+1,pixel_j);
				R2Pixel& pixel3 = Pixel(pixel_i,pixel_j+1);
				R2Pixel& pixel4 = Pixel(pixel_i+1,pixel_j+1);

				// Find function value at given pixel in scaled image.
				double value0 = pixel3[0]*(pixel_i+1-x_i)*(pixel_j-y_j) + pixel4[0]*(x_i-pixel_i+1)*(pixel_j-y_j) + pixel1[0]*(pixel_i+1-x_i)*(y_j-pixel_j) + pixel2[0]*(x_i-pixel_i)*(y_j-pixel_j);
				double value1 = pixel3[1]*(pixel_i+1-x_i)*(pixel_j-y_j) + pixel4[1]*(x_i-pixel_i+1)*(pixel_j-y_j) + pixel1[1]*(pixel_i+1-x_i)*(y_j-pixel_j) + pixel2[1]*(x_i-pixel_i)*(y_j-pixel_j);
				double value2 = pixel3[2]*(pixel_i+1-x_i)*(pixel_j-y_j) + pixel4[2]*(x_i-pixel_i+1)*(pixel_j-y_j) + pixel1[2]*(pixel_i+1-x_i)*(y_j-pixel_j) + pixel2[2]*(x_i-pixel_i)*(y_j-pixel_j);

				R2Pixel& pixel = new_image.Pixel(i,j);
				pixel[0] = value0;
				pixel[1] = value1;
				pixel[2] = value2;
				pixel.Clamp();
			}
		}
		*this = new_image;
	}

	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "Scale(%g, %g, %d) not implemented\n", sx, sy, sampling_method);
}


void R2Image::
Composite(const R2Image& top, int operation)
{
	// Composite passed image on top of this one using operation (e.g., OVER)
	
	// Ref: https://en.wikipedia.org/wiki/Alpha_compositing
	for(int j=1; j<height; j++){
		for(int i=1; i<width; i++){
			const R2Pixel& pixel_top = top.Pixel(i,j);
			R2Pixel& pixel_bottom = Pixel(i,j);
			pixel_bottom[0] = (pixel_top[0]*pixel_top[3]+pixel_bottom[0]*pixel_bottom[3]*(1-pixel_top[3]))/(pixel_top[3]+pixel_bottom[3]*(1-pixel_top[3]));
			pixel_bottom[1] = (pixel_top[1]*pixel_top[3]+pixel_bottom[1]*pixel_bottom[3]*(1-pixel_top[3]))/(pixel_top[3]+pixel_bottom[3]*(1-pixel_top[3]));
			pixel_bottom[2] = (pixel_top[2]*pixel_top[3]+pixel_bottom[2]*pixel_bottom[3]*(1-pixel_top[3]))/(pixel_top[3]+pixel_bottom[3]*(1-pixel_top[3]));
			pixel_bottom[3] = pixel_top[3]+pixel_bottom[3]*(1-pixel_top[3]);
		}
	}
	// FILL IN IMPLEMENTATION HERE (REMOVE PRINT STATEMENT WHEN DONE)
	// fprintf(stderr, "Composite not implemented\n");
}


// Miscellaneous operations ////////////////////////////////////////////////

void R2Image::
ExtractChannel(int channel)
{
	// Extracts a channel of an image (e.g., R2_IMAGE_RED_CHANNEL).  
	// Leaves the specified channel intact, 
	// and sets all the other ones to zero.

	// Extract channel
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			R2Pixel& pixel = Pixel(i, j);
			for (int c = 0; c < R2_IMAGE_NUM_CHANNELS; c++) {
				if (c != channel) pixel[c] = 0.0;
			}
		}
	}
}



void R2Image::
CopyChannel(const R2Image& from_image, int from_channel, int to_channel)
{
	// Copies one channel of an image (e.g., R2_IMAGE_RED_CHANNEL).  
	// to another channel

	// Check consistency of image dimensions
	if ((from_image.Width() != Width()) || (from_image.Height() != Height())) {
		fprintf(stderr, "Invalid image dimensions in R2Image::CopyChannel\n");
		abort();
	}

	// Copy channel
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			R2Pixel& to_pixel = Pixel(i, j);
			const R2Pixel& from_pixel = from_image.Pixel(i, j);
			to_pixel[to_channel] = from_pixel[from_channel];
		}
	}
}


////////////////////////////////////////////////////////////////////////
// I/O Functions
////////////////////////////////////////////////////////////////////////

int R2Image::
Read(const char *filename)
{
	// Initialize everything
	if (pixels) { delete [] pixels; pixels = NULL; }
	npixels = width = height = 0;

	// Parse input filename extension
	char *input_extension;
	if (!(input_extension = (char*)strrchr(filename, '.'))) {
		fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
		return 0;
	}
	
	// Read file of appropriate type
	if (!strncmp(input_extension, ".bmp", 4)) return ReadBMP(filename);
	else if (!strncmp(input_extension, ".ppm", 4)) return ReadPPM(filename);
	else if (!strncmp(input_extension, ".jpg", 4)) return ReadJPEG(filename);
	else if (!strncmp(input_extension, ".jpeg", 5)) return ReadJPEG(filename);
	else if (!strncmp(input_extension, ".txt", 4)) return ReadTXT(filename);
	
	// Should never get here
	fprintf(stderr, "Unrecognized image file extension");
	return 0;
}



int R2Image::
Write(const char *filename) const
{
	// Parse input filename extension
	char *input_extension;
	if (!(input_extension = (char*)strrchr(filename, '.'))) {
		fprintf(stderr, "Input file has no extension (e.g., .jpg).\n");
		return 0;
	}
	
	// Write file of appropriate type
	if (!strncmp(input_extension, ".bmp", 4)) return WriteBMP(filename);
	else if (!strncmp(input_extension, ".ppm", 4)) return WritePPM(filename, 1);
	else if (!strncmp(input_extension, ".jpg", 4)) return WriteJPEG(filename);
	else if (!strncmp(input_extension, ".jpeg", 5)) return WriteJPEG(filename);
	else if (!strncmp(input_extension, ".txt", 4)) return WriteTXT(filename);

	// Should never get here
	fprintf(stderr, "Unrecognized image file extension");
	return 0;
}



////////////////////////////////////////////////////////////////////////
// BMP I/O
////////////////////////////////////////////////////////////////////////

#if !defined(_WIN32)

typedef struct tagBITMAPFILEHEADER {
	unsigned short int bfType;
	unsigned int bfSize;
	unsigned short int bfReserved1;
	unsigned short int bfReserved2;
	unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER {
	unsigned int biSize;
	int biWidth;
	int biHeight;
	unsigned short int biPlanes;
	unsigned short int biBitCount;
	unsigned int biCompression;
	unsigned int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	unsigned int biClrUsed;
	unsigned int biClrImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBTRIPLE {
	unsigned char rgbtBlue;
	unsigned char rgbtGreen;
	unsigned char rgbtRed;
} RGBTRIPLE;

typedef struct tagRGBQUAD {
	unsigned char rgbBlue;
	unsigned char rgbGreen;
	unsigned char rgbRed;
	unsigned char rgbReserved;
} RGBQUAD;

#endif

#define BI_RGB        0L
#define BI_RLE8       1L
#define BI_RLE4       2L
#define BI_BITFIELDS  3L

#define BMP_BF_TYPE 0x4D42 /* word BM */
#define BMP_BF_OFF_BITS 54 /* 14 for file header + 40 for info header (not sizeof(), but packed size) */
#define BMP_BI_SIZE 40 /* packed size of info header */


static unsigned short int WordReadLE(FILE *fp)
{
	// Read a unsigned short int from a file in little endian format 
	unsigned short int lsb, msb;
	lsb = getc(fp);
	msb = getc(fp);
	return (msb << 8) | lsb;
}



static void WordWriteLE(unsigned short int x, FILE *fp)
{
	// Write a unsigned short int to a file in little endian format
	unsigned char lsb = (unsigned char) (x & 0x00FF); putc(lsb, fp); 
	unsigned char msb = (unsigned char) (x >> 8); putc(msb, fp);
}



static unsigned int DWordReadLE(FILE *fp)
{
	// Read a unsigned int word from a file in little endian format 
	unsigned int b1 = getc(fp);
	unsigned int b2 = getc(fp);
	unsigned int b3 = getc(fp);
	unsigned int b4 = getc(fp);
	return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void DWordWriteLE(unsigned int x, FILE *fp)
{
	// Write a unsigned int to a file in little endian format 
	unsigned char b1 = (x & 0x000000FF); putc(b1, fp);
	unsigned char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
	unsigned char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
	unsigned char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



static int LongReadLE(FILE *fp)
{
	// Read a int word from a file in little endian format 
	int b1 = getc(fp);
	int b2 = getc(fp);
	int b3 = getc(fp);
	int b4 = getc(fp);
	return (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
}



static void LongWriteLE(int x, FILE *fp)
{
	// Write a int to a file in little endian format 
	char b1 = (x & 0x000000FF); putc(b1, fp);
	char b2 = ((x >> 8) & 0x000000FF); putc(b2, fp);
	char b3 = ((x >> 16) & 0x000000FF); putc(b3, fp);
	char b4 = ((x >> 24) & 0x000000FF); putc(b4, fp);
}



int R2Image::
ReadBMP(const char *filename)
{
	// Open file
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	/* Read file header */
	BITMAPFILEHEADER bmfh;
	bmfh.bfType = WordReadLE(fp);
	bmfh.bfSize = DWordReadLE(fp);
	bmfh.bfReserved1 = WordReadLE(fp);
	bmfh.bfReserved2 = WordReadLE(fp);
	bmfh.bfOffBits = DWordReadLE(fp);
	
	/* Check file header */
	assert(bmfh.bfType == BMP_BF_TYPE);
	/* ignore bmfh.bfSize */
	/* ignore bmfh.bfReserved1 */
	/* ignore bmfh.bfReserved2 */
	assert(bmfh.bfOffBits == BMP_BF_OFF_BITS);
	
	/* Read info header */
	BITMAPINFOHEADER bmih;
	bmih.biSize = DWordReadLE(fp);
	bmih.biWidth = LongReadLE(fp);
	bmih.biHeight = LongReadLE(fp);
	bmih.biPlanes = WordReadLE(fp);
	bmih.biBitCount = WordReadLE(fp);
	bmih.biCompression = DWordReadLE(fp);
	bmih.biSizeImage = DWordReadLE(fp);
	bmih.biXPelsPerMeter = LongReadLE(fp);
	bmih.biYPelsPerMeter = LongReadLE(fp);
	bmih.biClrUsed = DWordReadLE(fp);
	bmih.biClrImportant = DWordReadLE(fp);
	
	// Check info header 
	assert(bmih.biSize == BMP_BI_SIZE);
	assert(bmih.biWidth > 0);
	assert(bmih.biHeight > 0);
	assert(bmih.biPlanes == 1);
	assert(bmih.biBitCount == 24);  /* RGB */
	assert(bmih.biCompression == BI_RGB);   /* RGB */
	int lineLength = bmih.biWidth * 3;  /* RGB */
	if ((lineLength % 4) != 0) lineLength = (lineLength / 4 + 1) * 4;
	assert(bmih.biSizeImage == (unsigned int) lineLength * (unsigned int) bmih.biHeight);

	// Assign width, height, and number of pixels
	width = bmih.biWidth;
	height = bmih.biHeight;
	npixels = width * height;

	// Allocate unsigned char buffer for reading pixels
	int rowsize = 3 * width;
	if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
	int nbytes = bmih.biSizeImage;
	unsigned char *buffer = new unsigned char [nbytes];
	if (!buffer) {
		fprintf(stderr, "Unable to allocate temporary memory for BMP file");
		fclose(fp);
		return 0;
	}

	// Read buffer 
	fseek(fp, (long) bmfh.bfOffBits, SEEK_SET);
	if (fread(buffer, 1, bmih.biSizeImage, fp) != bmih.biSizeImage) {
		fprintf(stderr, "Error while reading BMP file %s", filename);
		return 0;
	}

	// Close file
	fclose(fp);

	// Allocate pixels for image
	pixels = new R2Pixel [ width * height ];
	if (!pixels) {
		fprintf(stderr, "Unable to allocate memory for BMP file");
		fclose(fp);
		return 0;
	}

	// Assign pixels
	for (int j = 0; j < height; j++) {
		unsigned char *p = &buffer[j * rowsize];
		for (int i = 0; i < width; i++) {
			double b = (double) *(p++) / 255;
			double g = (double) *(p++) / 255;
			double r = (double) *(p++) / 255;
			R2Pixel pixel(r, g, b, 1);
			SetPixel(i, j, pixel);
		}
	}

	// Free unsigned char buffer for reading pixels
	delete [] buffer;

	// Return success
	return 1;
}



int R2Image::
WriteBMP(const char *filename) const
{
	// Open file
	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Compute number of bytes in row
	int rowsize = 3 * width;
	if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;

	// Write file header 
	BITMAPFILEHEADER bmfh;
	bmfh.bfType = BMP_BF_TYPE;
	bmfh.bfSize = BMP_BF_OFF_BITS + rowsize * height;
	bmfh.bfReserved1 = 0;
	bmfh.bfReserved2 = 0;
	bmfh.bfOffBits = BMP_BF_OFF_BITS;
	WordWriteLE(bmfh.bfType, fp);
	DWordWriteLE(bmfh.bfSize, fp);
	WordWriteLE(bmfh.bfReserved1, fp);
	WordWriteLE(bmfh.bfReserved2, fp);
	DWordWriteLE(bmfh.bfOffBits, fp);

	// Write info header 
	BITMAPINFOHEADER bmih;
	bmih.biSize = BMP_BI_SIZE;
	bmih.biWidth = width;
	bmih.biHeight = height;
	bmih.biPlanes = 1;
	bmih.biBitCount = 24;       /* RGB */
	bmih.biCompression = BI_RGB;    /* RGB */
	bmih.biSizeImage = rowsize * (unsigned int) bmih.biHeight;  /* RGB */
	bmih.biXPelsPerMeter = 2925;
	bmih.biYPelsPerMeter = 2925;
	bmih.biClrUsed = 0;
	bmih.biClrImportant = 0;
	DWordWriteLE(bmih.biSize, fp);
	LongWriteLE(bmih.biWidth, fp);
	LongWriteLE(bmih.biHeight, fp);
	WordWriteLE(bmih.biPlanes, fp);
	WordWriteLE(bmih.biBitCount, fp);
	DWordWriteLE(bmih.biCompression, fp);
	DWordWriteLE(bmih.biSizeImage, fp);
	LongWriteLE(bmih.biXPelsPerMeter, fp);
	LongWriteLE(bmih.biYPelsPerMeter, fp);
	DWordWriteLE(bmih.biClrUsed, fp);
	DWordWriteLE(bmih.biClrImportant, fp);

	// Write image, swapping blue and red in each pixel
	int pad = rowsize - width * 3;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			const R2Pixel& pixel = (*this)[i][j];
			double r = 255.0 * pixel.Red();
			double g = 255.0 * pixel.Green();
			double b = 255.0 * pixel.Blue();
			if (r >= 255) r = 255;
			if (g >= 255) g = 255;
			if (b >= 255) b = 255;
			fputc((unsigned char) b, fp);
			fputc((unsigned char) g, fp);
			fputc((unsigned char) r, fp);
		}

		// Pad row
		for (int i = 0; i < pad; i++) fputc(0, fp);
	}
	
	// Close file
	fclose(fp);

	// Return success
	return 1;  
}



////////////////////////////////////////////////////////////////////////
// PPM I/O
////////////////////////////////////////////////////////////////////////

int R2Image::
ReadPPM(const char *filename)
{
	// Open file
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Read PPM file magic identifier
	char buffer[128];
	if (!fgets(buffer, 128, fp)) {
		fprintf(stderr, "Unable to read magic id in PPM file");
		fclose(fp);
		return 0;
	}

	// skip comments
	int c = getc(fp);
	while (c == '#') {
		while (c != '\n') c = getc(fp);
		c = getc(fp);
	}
	ungetc(c, fp);

	// Read width and height
	if (fscanf(fp, "%d%d", &width, &height) != 2) {
		fprintf(stderr, "Unable to read width and height in PPM file");
		fclose(fp);
		return 0;
	}

	npixels = width * height;
	
	// Read max value
	double max_value;
	if (fscanf(fp, "%lf", &max_value) != 1) {
		fprintf(stderr, "Unable to read max_value in PPM file");
		fclose(fp);
		return 0;
	}
	
	// Allocate image pixels
	pixels = new R2Pixel [ width * height ];
	if (!pixels) {
		fprintf(stderr, "Unable to allocate memory for PPM file");
		fclose(fp);
		return 0;
	}

	// Check if raw or ascii file
	if (!strcmp(buffer, "P6\n")) {
		// Read up to one character of whitespace (\n) after max_value
		int c = getc(fp);
		if (!isspace(c)) putc(c, fp);

		// Read raw image data 
		// First ppm pixel is top-left, so read in opposite scan-line order
		for (int j = height-1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				double r = (double) getc(fp) / max_value;
				double g = (double) getc(fp) / max_value;
				double b = (double) getc(fp) / max_value;
				R2Pixel pixel(r, g, b, 1);
				SetPixel(i, j, pixel);
			}
		}
	}
	else {
		// Read asci image data 
		// First ppm pixel is top-left, so read in opposite scan-line order
		for (int j = height-1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
	// Read pixel values
	int red, green, blue;
	if (fscanf(fp, "%d%d%d", &red, &green, &blue) != 3) {
		fprintf(stderr, "Unable to read data at (%d,%d) in PPM file", i, j);
		fclose(fp);
		return 0;
	}

	// Assign pixel values
	double r = (double) red / max_value;
	double g = (double) green / max_value;
	double b = (double) blue / max_value;
				R2Pixel pixel(r, g, b, 1);
				SetPixel(i, j, pixel);
			}
		}
	}

	// Close file
	fclose(fp);

	// Return success
	return 1;
}



int R2Image::
WritePPM(const char *filename, int ascii) const
{
	// Check type
	if (ascii) {
		// Open file
		FILE *fp = fopen(filename, "w");
		if (!fp) {
			fprintf(stderr, "Unable to open image file: %s\n", filename);
			return 0;
		}

		// Print PPM image file 
		// First ppm pixel is top-left, so write in opposite scan-line order
		fprintf(fp, "P3\n");
		fprintf(fp, "%d %d\n", width, height);
		fprintf(fp, "255\n");
		for (int j = height-1; j >= 0 ; j--) {
			for (int i = 0; i < width; i++) {
				const R2Pixel& p = (*this)[i][j];
				int r = (int) (255 * p.Red());
				int g = (int) (255 * p.Green());
				int b = (int) (255 * p.Blue());
				fprintf(fp, "%-3d %-3d %-3d  ", r, g, b);
				if (((i+1) % 4) == 0) fprintf(fp, "\n");
			}
			if ((width % 4) != 0) fprintf(fp, "\n");
		}
		fprintf(fp, "\n");

		// Close file
		fclose(fp);
	}
	else {
		// Open file
		FILE *fp = fopen(filename, "wb");
		if (!fp) {
			fprintf(stderr, "Unable to open image file: %s\n", filename);
			return 0;
		}
		
		// Print PPM image file 
		// First ppm pixel is top-left, so write in opposite scan-line order
		fprintf(fp, "P6\n");
		fprintf(fp, "%d %d\n", width, height);
		fprintf(fp, "255\n");
		for (int j = height-1; j >= 0 ; j--) {
			for (int i = 0; i < width; i++) {
				const R2Pixel& p = (*this)[i][j];
				int r = (int) (255 * p.Red());
				int g = (int) (255 * p.Green());
				int b = (int) (255 * p.Blue());
				fprintf(fp, "%c%c%c", r, g, b);
			}
		}
		
		// Close file
		fclose(fp);
	}

	// Return success
	return 1;  
}



////////////////////////////////////////////////////////////////////////
// JPEG I/O
////////////////////////////////////////////////////////////////////////


extern "C" { 
#   define XMD_H // Otherwise, a conflict with INT32
#   undef FAR // Otherwise, a conflict with windows.h
#   include "jpeg/jpeglib.h"
};



int R2Image::
ReadJPEG(const char *filename)
{
	// Open file
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Initialize decompression info
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, fp);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	// Remember image attributes
	width = cinfo.output_width;
	height = cinfo.output_height;
	npixels = width * height;
	int ncomponents = cinfo.output_components;

	// Allocate pixels for image
	pixels = new R2Pixel [ npixels ];
	if (!pixels) {
		fprintf(stderr, "Unable to allocate memory for BMP file");
		fclose(fp);
		return 0;
	}

	// Allocate unsigned char buffer for reading image
	int rowsize = ncomponents * width;
	if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
	int nbytes = rowsize * height;
	unsigned char *buffer = new unsigned char [nbytes];
	if (!buffer) {
		fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
		fclose(fp);
		return 0;
	}

	// Read scan lines 
	// First jpeg pixel is top-left, so read pixels in opposite scan-line order
	while (cinfo.output_scanline < cinfo.output_height) {
		int scanline = cinfo.output_height - cinfo.output_scanline - 1;
		unsigned char *row_pointer = &buffer[scanline * rowsize];
		jpeg_read_scanlines(&cinfo, &row_pointer, 1);
	}

	// Free everything
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	// Close file
	fclose(fp);

	// Assign pixels
	for (int j = 0; j < height; j++) {
		unsigned char *p = &buffer[j * rowsize];
		for (int i = 0; i < width; i++) {
			double r, g, b, a;
			if (ncomponents == 1) {
				r = g = b = (double) *(p++) / 255;
				a = 1;
			}
			else if (ncomponents == 1) {
				r = g = b = (double) *(p++) / 255;
				a = 1;
				p++;
			}
			else if (ncomponents == 3) {
				r = (double) *(p++) / 255;
				g = (double) *(p++) / 255;
				b = (double) *(p++) / 255;
				a = 1;
			}
			else if (ncomponents == 4) {
				r = (double) *(p++) / 255;
				g = (double) *(p++) / 255;
				b = (double) *(p++) / 255;
				a = (double) *(p++) / 255;
			}
			else {
				fprintf(stderr, "Unrecognized number of components in jpeg image: %d\n", ncomponents);
				return 0;
			}
			R2Pixel pixel(r, g, b, a);
			SetPixel(i, j, pixel);
		}
	}

	// Free unsigned char buffer for reading pixels
	delete [] buffer;

	// Return success
	return 1;
}


	

int R2Image::
WriteJPEG(const char *filename) const
{
	// Open file
	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Initialize compression info
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, fp);
	cinfo.image_width = width; 	/* image width and height, in pixels */
	cinfo.image_height = height;
	cinfo.input_components = 3;		/* # of color components per pixel */
	cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
	cinfo.dct_method = JDCT_ISLOW;
	jpeg_set_defaults(&cinfo);
	cinfo.optimize_coding = TRUE;
	jpeg_set_quality(&cinfo, 75, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	
	// Allocate unsigned char buffer for reading image
	int rowsize = 3 * width;
	if ((rowsize % 4) != 0) rowsize = (rowsize / 4 + 1) * 4;
	int nbytes = rowsize * height;
	unsigned char *buffer = new unsigned char [nbytes];
	if (!buffer) {
		fprintf(stderr, "Unable to allocate temporary memory for JPEG file");
		fclose(fp);
		return 0;
	}

	// Fill buffer with pixels
	for (int j = 0; j < height; j++) {
		unsigned char *p = &buffer[j * rowsize];
		for (int i = 0; i < width; i++) {
			const R2Pixel& pixel = (*this)[i][j];
			int r = (int) (255 * pixel.Red());
			int g = (int) (255 * pixel.Green());
			int b = (int) (255 * pixel.Blue());
			if (r > 255) r = 255;
			if (g > 255) g = 255;
			if (b > 255) b = 255;
			*(p++) = r;
			*(p++) = g;
			*(p++) = b;
		}
	}



	// Output scan lines
	// First jpeg pixel is top-left, so write in opposite scan-line order
	while (cinfo.next_scanline < cinfo.image_height) {
		int scanline = cinfo.image_height - cinfo.next_scanline - 1;
		unsigned char *row_pointer = &buffer[scanline * rowsize];
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}

	// Free everything
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	// Close file
	fclose(fp);

	// Free unsigned char buffer for reading pixels
	delete [] buffer;

	// Return number of bytes written
	return 1;
}



////////////////////////////////////////////////////////////////////////
// TXT I/O
////////////////////////////////////////////////////////////////////////

int R2Image::
ReadTXT(const char *filename)
{
	// Open file
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Read width, height, and nchannels
	int nchannels;
	if (fscanf(fp, "%d%d%d", &width, &height, &nchannels) != 3) {
		fprintf(stderr, "Unable to read width and height and nchannels in TXT file");
		fclose(fp);
		return 0;
	}

	// Check number of channels
	if ((nchannels == 0) || (nchannels > 4)) {
		fprintf(stderr, "Invalid number of channels (%d) in TXT image %s\n", nchannels, filename);
		fclose(fp);
		return 0;
	}
		
	// Compute number of pixels
	npixels = width * height;
	
	// Allocate image pixels
	pixels = new R2Pixel [ width * height ];
	if (!pixels) {
		fprintf(stderr, "Unable to allocate memory for TXT file");
		fclose(fp);
		return 0;
	}

	// Read asci image data 
	// First pixel is top-left, so read in opposite scan-line order
	for (int j = height-1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			// Read pixel values
			double rgba[4];
			for (int k = 0; k < nchannels; k++) {
				if (fscanf(fp, "%lf\n", &rgba[k]) != 1) {
					fprintf(stderr, "Unable to read data at (%d,%d) in TXT file", i, j);
					fclose(fp);
					return 0;
				}
			}

			// Rectify channels
			if (nchannels == 1) { rgba[3] = 1.0; rgba[1] = rgba[2] = rgba[0]; }
			else if (nchannels == 2) { rgba[3] = rgba[2]; rgba[1] = rgba[2] = rgba[0]; } 
			else if (nchannels == 3) { rgba[3] = 1.0; } 

			// Set pixel
			SetPixel(i, j, R2Pixel(rgba));
		}
	}

	// Close file
	fclose(fp);

	// Return success
	return 1;
}



int R2Image::
WriteTXT(const char *filename) const
{
	// Open file
	FILE *fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "Unable to open image file: %s\n", filename);
		return 0;
	}

	// Print width, height, and nchannels
	fprintf(fp, "%d %d %d\n", width, height, 4);

	// Print pixel values
	// First pixel is top-left, so write in opposite scan-line order
	for (int j = height-1; j >= 0 ; j--) {
		for (int i = 0; i < width; i++) {
			R2Pixel pixel = Pixel(i, j);
			fprintf(fp, "%g %g %g %g\n", pixel[0], pixel[1], pixel[2], pixel[3]);
		}
	}

	// Close file
	fclose(fp);

	// Return success
	return 1;  
}