#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat Helipad, Palace;
int n=0;
vector<Point2f> imagePoint;// point coordinates, global variable;
vector<Point3f> realPoint;

void on_mouse(int event,int x,int y,int flags,void *ustc)
{
    Point pt;//mouse position;
    char coordinate[16];

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        pt = Point2f(x,y);
        cout<<x<<" "<<y<<endl;
        imagePoint.push_back(pt);
        n++;

        //imshow("org",org);

        if(n>=4)//only four points are needed;
        {
            //imshow("org",org);
            cvDestroyAllWindows();
        }
    }
}

int main()
{
	Helipad = imread("Helipad.jpg");

    namedWindow("Helipad");
    setMouseCallback("Helipad",on_mouse,0);//mouse callback function;

    imshow("Helipad",Helipad);
    waitKey(0);

    realPoint.push_back(Point3f(0,2,0));
    realPoint.push_back(Point3f(0,2,-4));
    realPoint.push_back(Point3f(0,0,-4));
    realPoint.push_back(Point3f(1.5,0,-4));

    vector<vector< Point2f> > imagepointdata;
    imagepointdata.push_back(imagePoint);

    vector<vector< Point3f> > realpointdata;
    realpointdata.push_back(realPoint);

    Mat cameraMatrix;
  	Mat distCoeffs;
  	vector<Mat> rvecs;
	vector<Mat> tvecs;

	calibrateCamera(realpointdata, imagepointdata, Helipad.size(), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS);

	cout << cameraMatrix << endl;
	//cout << distCoeffs << endl;
	//cout << rvecs.size()<< endl;
	cout << tvecs.size() << endl;

	//cout << rvecs[0] << endl;
	cout << tvecs[0].size() << endl;

	Mat test;

	Rodrigues(rvecs[0], test);
	
	
	//cout<<test<<endl;

	Mat cam, t;
	rotate(tvecs[0], t, ROTATE_90_CLOCKWISE);
	//t = tvecs[0].reshape(3,1);
	t.convertTo(t, cameraMatrix.type());

	Mat kr;
	Mat kt;

	kr = cameraMatrix*test;

	cout<< cameraMatrix<< endl;
	cout << t << endl;
	cout << t.size() << endl;

	kt = cameraMatrix*t.t();
	cout<<kt<<endl;
	kt.convertTo(kt, kr.type());


	//cout << "KR" << kr << endl;

	hconcat(kr, kt, cam);

	cout<< "cam" << cam << endl;
	Mat check;
	Mat p = (Mat_<double>(1, 4) << 0, 2, 0, 1);
	check = cam*p.t();
	cout<<check<<endl;
	cout<<imagePoint[0]<<endl;

	return 0;
}