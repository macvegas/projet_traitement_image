//////////////////////////////////////////////////////////////////////////
// Option Images
// Projet, s�ance 1
// th�me : premier pas en OpenCV
// contenu : charge, affiche, r�duction, calcul et affichage d'histogramme
// version : 17.1128
//////////////////////////////////////////////////////////////////////////


#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;
static void help()
{
	cout
		<< "\nThis program illustrates the use of findContours and drawContours\n"
		<< "The original image is put up along with the image of drawn contours\n"
		<< "Usage:\n"
		<< "./contours2\n"
		<< "\nA trackbar is put up which controls the contour level from -3 to 3\n"
		<< endl;
}
cv::Size w;
int levels = 3;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
static void on_trackbar(int, void*)
{
	Mat cnt_img = Mat::zeros(w, CV_8UC3);
	int _levels = levels - 3;
	drawContours(cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128, 255, 255),
		3, LINE_AA, hierarchy, std::abs(_levels));
	imshow("contours", cnt_img);
}
/*
int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help h||}");
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	
	string path = "c:/Users/sbeaulie/Desktop/Projet OpenCV-CMake/images/00000.png";
	Mat img2= imread(path);

	//Grayscale matrix
	cv::Mat grayscaleMat(img2.size(), CV_8U);

	w = img2.size();

	//Convert BGR to Gray
	cv::cvtColor(img2, grayscaleMat, CV_BGR2GRAY);

	//Binary image
	cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

	//Apply thresholding
	cv::threshold(grayscaleMat, binaryMat, 200, 255, cv::THRESH_BINARY);

	/*
	
	Mat img = Mat::zeros(w, w, CV_8UC1);
	//Draw 6 faces
	for (int i = 0; i < 6; i++)
	{
		int dx = (i % 2) * 250 - 30;
		int dy = (i / 2) * 150;
		const Scalar white = Scalar(255);
		const Scalar black = Scalar(0);
		if (i == 0)
		{
			for (int j = 0; j <= 10; j++)
			{
				double angle = (j + 5)*CV_PI / 21;
				line(img, Point(cvRound(dx + 100 + j * 10 - 80 * cos(angle)),
					cvRound(dy + 100 - 90 * sin(angle))),
					Point(cvRound(dx + 100 + j * 10 - 30 * cos(angle)),
						cvRound(dy + 100 - 30 * sin(angle))), white, 1, 8, 0);
			}
		}
		ellipse(img, Point(dx + 150, dy + 100), Size(100, 70), 0, 0, 360, white, -1, 8, 0);
		ellipse(img, Point(dx + 115, dy + 70), Size(30, 20), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 185, dy + 70), Size(30, 20), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 115, dy + 70), Size(15, 15), 0, 0, 360, white, -1, 8, 0);
		ellipse(img, Point(dx + 185, dy + 70), Size(15, 15), 0, 0, 360, white, -1, 8, 0);
		ellipse(img, Point(dx + 115, dy + 70), Size(5, 5), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 185, dy + 70), Size(5, 5), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 150, dy + 100), Size(10, 5), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 150, dy + 150), Size(40, 10), 0, 0, 360, black, -1, 8, 0);
		ellipse(img, Point(dx + 27, dy + 100), Size(20, 35), 0, 0, 360, white, -1, 8, 0);
		ellipse(img, Point(dx + 273, dy + 100), Size(20, 35), 0, 0, 360, white, -1, 8, 0);
	}
	//show the faces
	namedWindow("image", WINDOW_NORMAL);
	//cv::resizeWindow("image", , HEIGHT);
	imshow("image", binaryMat);
	//Extract the contours so that
	vector<vector<Point> > contours0;
	findContours(binaryMat, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	contours.resize(contours0.size());
	for (size_t k = 0; k < contours0.size(); k++)
		approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

	namedWindow("contours", WINDOW_NORMAL);
	createTrackbar("levels+3", "contours", &levels, 7, on_trackbar);
	on_trackbar(0, 0);
	waitKey();
	return 0;
}*/
