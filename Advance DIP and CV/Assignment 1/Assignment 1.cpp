#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

string winName = "meanshift";
int spatialRad, colorRad, maxPyrLevel;
Mat res, img;

double getPSNR(Mat& I1, Mat& I2);
int print_both(Mat& image, Mat& image1, bool horizantal);
int LoG(Mat& image, int std);
int meanshift(Mat& image);
int harris(Mat& image);
int le_gall(Mat& img, Mat& out);
double getPSNR(Mat& I1, Mat& I2);

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
	Mat blur, LoG, abs_dst;

    /// Remove noise by blurring with a Gaussian filter
    GaussianBlur( image, blur, Size(0,0), std, std);
 
    /// Apply Laplace function
    Laplacian( blur, LoG, CV_32F, 3, 1, 1);
    imshow( "LoG", LoG );
    waitKey(0);
    //imwrite("images/log.jpg", LoG);

    GaussianBlur(LoG,LoG,Size(0,0),3,3);
    Mat zeroCross;
    LoG.convertTo(zeroCross,CV_8U);
    //cout<<zeroCross.depth()<<endl;
    int i=0,j=0;
    for(i=0;i<LoG.rows;i++){
        for(j=0;j<LoG.cols;j++){
            if(((LoG.at<float>(i-1,j)>0) && (LoG.at<float>(i,j)<0) ) ||((LoG.at<float>(i-1,j)<0) && (LoG.at<float>(i,j)>0)) ){
                zeroCross.at<uchar>(i,j)=0;
            }
            else if(((LoG.at<float>(i+1,j)>0) && (LoG.at<float>(i,j)<0) ) ||((LoG.at<float>(i+1,j)<0) && (LoG.at<float>(i,j)>0))){
                zeroCross.at<uchar>(i,j)=0;
            }
            else if(((LoG.at<float>(i,j-1)>0) && (LoG.at<float>(i,j)<0) ) ||((LoG.at<float>(i,j-1)<0) && (LoG.at<float>(i,j)>0))){
                zeroCross.at<uchar>(i,j)=0;
            }
            else if(((LoG.at<float>(i,j+1)>0) && (LoG.at<float>(i,j)<0) ) || ((LoG.at<float>(i,j+1)<0) && (LoG.at<float>(i,j)>0))){
                zeroCross.at<uchar>(i,j)=0;
            }
            else{
                zeroCross.at<uchar>(i,j)=255;
            }
            
        }
    }

    imshow( "Zero Cross", zeroCross );
    waitKey(0);
    //imwrite("images/zerocross.jpg", zeroCross);
    destroyAllWindows();

    return 0;
}

int meanshift(Mat& image)
{
    Mat msf;
    double spatialWindowRadius = 20;
    double colorWindowRadius = 50;
    pyrMeanShiftFiltering(image, msf, spatialWindowRadius,colorWindowRadius);
    namedWindow("Mean Shift Filtering");
    imshow("Mean Shift Filtering",msf);
    waitKey(0);
    //imwrite("images/meanshift.jpg", msf);
    destroyWindow("Mean Shift Filtering");
    return 0;
}

int harris(Mat& image)
{
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 120;

    Mat dest, dest_norm, dest_norm_scaled;
    dest = Mat::zeros( image.size(), CV_8UC1 );         

    /// Detecting corners
    cornerHarris(image, dest, blockSize, apertureSize, k, BORDER_DEFAULT);

    /// Normalizing
    normalize( dest, dest_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs(dest_norm, dest_norm_scaled);
    equalizeHist(dest_norm_scaled, dest_norm_scaled);

    for( int i = 0; i < dest_norm.rows ; i++ )
    {
        for( int j = 0; j < dest_norm.cols; j++ )
        {
            if( (int) dest_norm.at<float>(i,j) > thresh )
            {
                circle( dest_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }

    namedWindow("Harris", WINDOW_NORMAL);
    imshow("Harris", dest_norm_scaled);
    waitKey(0);
    //imwrite("images/harris.jpg", dest_norm_scaled);
    destroyWindow("Harris");

}

int le_gall(Mat& img, Mat& out)
{
    Mat noise(img.size(), img.type()), noisy_img;
    randn(noise, 0, 25);
    noisy_img = img+noise;
    imshow("noise", noisy_img);
    waitKey(0);
    //imwrite("images/noisy.jpg", noisy_img);
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

    //LH
    filter2D(img_low, img_lh, -1, analyse_high);
    resize(img_lh, img_lh, Size(img_low.cols, img_low.rows/2));

    // HL
    filter2D(img_high, img_hl, -1, analyse_low);
    resize(img_hl, img_hl, Size(img_high.cols, img_high.rows/2));;

    //HH
    filter2D(img_high, img_hh, -1, analyse_high);
    resize(img_hh, img_hh, Size(img_high.cols, img_high.rows/2));

    Mat temp1, temp2;
    hconcat(img_ll, img_lh, temp1);
    hconcat(img_hl, img_hh, temp2);
    vconcat(temp1, temp2, temp1);

    imshow("Features", temp1);
    waitKey(0);
    //imwrite("images/features.jpg", temp1);

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

    addWeighted( img_ll, 0.35, img_lh, 0.65, 0.0, img_low);
    //imshow("noise", img_low);
    //waitKey(0);
    //img_low = (img_ll + img_lh)/2;
    //img_high = (img_hl + img_hh)/2;
    addWeighted( img_hl, 0.1, img_hh, 0.9,0.0, img_high);
    //imshow("noise", img_high);
    //waitKey(0);

    resize(img_low, img_low, Size(img_low.cols*2, img_low.rows));
    resize(img_high, img_high, Size(img_high.cols*2, img_high.rows));

    rotate(syn_low, syn_low, ROTATE_90_CLOCKWISE);
    rotate(syn_high, syn_high, ROTATE_90_CLOCKWISE);

    filter2D(img_low, img_low, -1, syn_low);
    filter2D(img_high, img_high, -1, syn_high);

    //out = (img_low + img_high)/2;
    addWeighted( img_low, 0.82, img_high, 0.18, 0.0, out);

    //out.convertTo(out, CV_8UC1);
    //equalizeHist(out, out);
    //bitwise_not ( out, out );

    imshow("created output", out);
    waitKey(0);
    //imwrite("images/synthesis.jpg", out);

    Mat Gaussian_img, Median_img;
    GaussianBlur(noisy_img, Gaussian_img, Size(5,5), 0,0);
    medianBlur(noisy_img, Median_img, 5);

    imshow("Median Blur", Median_img);
    waitKey(0);
    //imwrite("images/Median.jpg", Median_img);
    imshow("Gaussina Blur", Gaussian_img);
    waitKey(0);
    //imwrite("images/Gaussian.jpg", Gaussian_img);

    cout << " PSNR for Median Blur and Grey\t" << getPSNR(Median_img, img) << endl;
    cout << " PSNR for Gaussian Blur and Grey\t" << getPSNR(Gaussian_img, img) << endl;
    cout << " PSNR for Final Image and Grey\t" << getPSNR(out, img) << endl;

    destroyAllWindows();
    return 0;
}

double getPSNR(Mat& I1, Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    //double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels for RBG

    double sse = s.val[0];

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}


int main()
{
	img = imread("Church.jpg");
	
	Mat grey;
	cvtColor(img, grey, COLOR_BGR2GRAY);
	cvtColor(grey, grey, COLOR_GRAY2BGR);
	
	print_both(img, grey, 0);
    //imwrite("images/greyscale.jpg", grey);

    Mat legall;

    le_gall(grey, legall);

    cvtColor(grey, grey, COLOR_BGR2GRAY);

	LoG(grey, 5);

	meanshift(img);

    harris(grey);

	return 0;
}