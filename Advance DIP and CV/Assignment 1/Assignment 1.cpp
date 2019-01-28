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
int part3(Mat& image, int std)
{
	Mat blur, dst, abs_dst;

	namedWindow("result", WINDOW_NORMAL);

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( image, blur, Size(0,0), std, std, BORDER_DEFAULT );
    imshow("result", image );
    waitKey(0);
    imshow("result", blur );
    waitKey(0);
 
    /// Apply Laplace function
    Laplacian( blur, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );
    imshow( "result", dst );
    waitKey(0);
    convertScaleAbs(dst, abs_dst );
    imshow( "result", abs_dst );
    waitKey(0);
    destroyWindow("result");

    return 0;
}

static void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}

static void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
         << "colorRad=" << colorRad << "; "
         << "maxPyrLevel=" << maxPyrLevel << endl;
    pyrMeanShiftFiltering( img, res, spatialRad, colorRad, maxPyrLevel );
    floodFillPostprocess( res, Scalar::all(2) );
    imshow( winName, res );
}

int part4()
{
	spatialRad = 20;
    colorRad = 50;
    maxPyrLevel = 1;

    namedWindow( winName, CV_WINDOW_AUTOSIZE );

    pyrMeanShiftFiltering( img, res, spatialRad, colorRad, maxPyrLevel );
    //floodFillPostprocess( res, Scalar::all(2) );
    imshow( winName, res );

    // createTrackbar( "spatialRad", winName, &spatialRad, 80, meanShiftSegmentation );
    // createTrackbar( "colorRad", winName, &colorRad, 60, meanShiftSegmentation );
    // createTrackbar( "maxPyrLevel", winName, &maxPyrLevel, 5, meanShiftSegmentation );

    meanShiftSegmentation(0, 0);
    waitKey(0);

    destroyWindow(winName);

    return 0;
}

int part5(Mat& image)
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

    namedWindow("Harris", WINDOW_NORMAL);
    imshow("Harris", dest_norm_scaled);
    waitKey(0);
    destroyWindow("Harris");

}

int main()
{
	img = imread("Church.jpg");
	
	Mat grey;
	cvtColor(img, grey, COLOR_BGR2GRAY);
	cvtColor(grey, grey, COLOR_GRAY2BGR);
	
	print_both(img, grey, 0);

    cvtColor(grey, grey, COLOR_BGR2GRAY);

	part3(grey, 2.5);

	part4();

    part5(grey);

	return 0;
}