#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

string winName = "meanshift";
int spatialRad, colorRad, maxPyrLevel;
Mat res, img;

// printing 2 images in single window
int print_both(Mat& image, Mat& image1, bool horizantal = 1)
{
	Mat out;
	namedWindow("Joined image", WINDOW_NORMAL);
	if (horizantal)
	{
		resizeWindow("Joined image", 1500, 500);         // viewable size in a normal laptop
		hconcat(image, image1, out);
	}
	else
	{
		resizeWindow("Joined image", 750, 1000);         // viewable size in a normal laptop
		vconcat(image, image1, out);
	}
	
	imshow("Joined image", out);
	waitKey(0);
	destroyWindow("Joined image");

	return 0;
}

// Laplacian of Gaussian on a Greyscale Image
int LoG(Mat& image, int std)
{
	Mat blur, dst, abs_dst;

	namedWindow("blur img");

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( image, blur, Size(0,0), std, std);
    imshow("blur img", image );
    waitKey(0);
    imshow("blur img", blur );
    waitKey(0);
 
    /// Apply Laplace function
    namedWindow("result");
    Laplacian( blur, dst, CV_32F, 3, 1, 1);
    imshow( "result", dst );
    waitKey(0);
    convertScaleAbs(dst, abs_dst );
    imshow( "result", abs_dst );
    waitKey(0);
    destroyAllWindows();

    return 0;
}

int meanshift(Mat& image){
    Mat msf;
    double spatialWindowRadius = 20;
    double colorWindowRadius = 50;
    pyrMeanShiftFiltering(image, msf, spatialWindowRadius,colorWindowRadius);
    namedWindow("Mean Shift Filtering");
    imshow("Mean Shift Filtering",msf);
    waitKey(0);
    destroyWindow("Mean Shift Filtering");
    return 0;
}

int harris(Mat& image)
{
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    Mat dest, dest_norm, dest_norm_scaled;
    dest = Mat::zeros( image.size(), CV_8UC1 );         

    /// Detecting corners
    cornerHarris(image, dest, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing
    normalize( dest, dest_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs(dest_norm, dest_norm_scaled);
    equalizeHist(dest_norm_scaled, dest_norm_scaled);

    namedWindow("Harris", WINDOW_NORMAL);
    imshow("Harris", dest_norm_scaled);
    waitKey(0);
    destroyWindow("Harris");

}

int le_gall(Mat& img, Mat& out)
{
    Mat noise(img.size(), img.type()), noisy_img;
    randn(noise, 0, 25);
    noisy_img = img+noise;
    imshow("noise", noisy_img);
    waitKey(0);
    destroyWindow("noise");

    Mat analyse_low = (Mat_<double>(1,5) << -0.125, 0.25, 0.75, 0.25, -0.125);
    Mat analyse_high = (Mat_<double>(1,3) << -0.5, 1, -0.5);
    Mat syn_low = (Mat_<double>(1, 3) << 0.5, 1, 0.5);
    Mat syn_high = (Mat_<double>(1, 5) << -0.125, 0.25, 0.75, 0.25, -0.125);

    Mat img_low, img_high, img_ll, img_lh, img_hl, img_hh;

    // low pass
    filter2D(noisy_img, img_low, -1, analyse_low);
    resize(img_low, img_low, Size(img_low.cols/2, img_low.rows));

    // high pass
    filter2D(noisy_img, img_high, -1, analyse_high);
    resize(img_high, img_high, Size(img_high.cols/2, img_high.rows));

    rotate(analyse_low, analyse_low, ROTATE_90_CLOCKWISE);
    rotate(analyse_high, analyse_high, ROTATE_90_CLOCKWISE);

    // LL
    filter2D(img_low, img_ll, -1, analyse_low);
    resize(img_ll, img_ll, Size(img_low.cols, img_low.rows/2));
    imshow("Low Low", img_ll);
    waitKey(0);

    //LH
    filter2D(img_low, img_lh, -1, analyse_high);
    resize(img_lh, img_lh, Size(img_low.cols, img_low.rows/2));
    imshow("Low High", img_lh);
    waitKey(0);

    // HL
    filter2D(img_high, img_hl, -1, analyse_low);
    resize(img_hl, img_hl, Size(img_high.cols, img_high.rows/2));
    imshow("High Low", img_hl);
    waitKey(0);

    //HH
    filter2D(img_high, img_hh, -1, analyse_high);
    resize(img_hh, img_hh, Size(img_high.cols, img_high.rows/2));
    imshow("High High", img_hh);
    waitKey(0);

    //upsampling
    resize(img_ll, img_ll, Size(img_ll.cols, img_ll.rows*2));
    resize(img_lh, img_lh, Size(img_lh.cols, img_lh.rows*2));
    resize(img_hl, img_hl, Size(img_hl.cols, img_hl.rows*2));
    resize(img_hh, img_hh, Size(img_hh.cols, img_hh.rows*2));

    rotate(syn_low, syn_low, ROTATE_90_CLOCKWISE);
    rotate(syn_high, syn_high, ROTATE_90_CLOCKWISE);

    // filter 1
    filter2D(img_ll, img_ll, -1, syn_low);
    filter2D(img_lh, img_lh, -1, syn_high);
    filter2D(img_hl, img_hl, -1, syn_low);
    filter2D(img_hh, img_hh, -1, syn_high);

    img_low = img_ll + img_lh;
    img_high = img_hl + img_hh;

    resize(img_low, img_low, Size(img_low.cols*2, img_low.rows));
    resize(img_high, img_high, Size(img_high.cols*2, img_high.rows));

    rotate(syn_low, syn_low, ROTATE_90_CLOCKWISE);
    rotate(syn_high, syn_high, ROTATE_90_CLOCKWISE);

    filter2D(img_low, img_low, -1, syn_low);
    filter2D(img_high, img_high, -1, syn_high);

    out = img_low + img_high;

    imshow("created output", out);
    waitKey(0);

    return 0;
}

int main()
{
	img = imread("Church.jpg");
	
	Mat grey;
	cvtColor(img, grey, COLOR_BGR2GRAY);
	cvtColor(grey, grey, COLOR_GRAY2BGR);
	
	print_both(img, grey, 0);

    Mat legall;

    le_gall(grey, legall);

    //cvtColor(grey, grey, COLOR_BGR2GRAY);

	//LoG(grey, 5);

	//meanshift(img);

    //harris(grey);

	return 0;
}