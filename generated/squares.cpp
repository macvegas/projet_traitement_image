// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <cstdio>
#include <numeric>

using namespace cv;
using namespace std;

static void help()
{
	cout <<
		"\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
		"memory storage to find squares in a list of images\n"
		"Returns sequence of squares detected on the image.\n"
		"the sequence is stored in the specified memory storage\n"
		"Call:\n"
		"./squares\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 5;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	//pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
	//pyrUp(pyr, timg, image.size());


	// blur will enhance edge detection
	
	Mat timg(image);
	//medianBlur(image, timg, 9);
	Mat gray0(timg.size(), CV_8U), gray;
	
	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 5, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}


// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares, cv::Scalar color)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];

		int n = (int)squares[i].size();
		//dont detect the border
		if (p->x > 3 && p->y > 3)
			polylines(image, &p, &n, 1, true, color, 3, LINE_AA);
	}

	imshow(wndname, image);
}

typedef vector<Point> square_t;

// returns true if pointA < pointB
// orders points by row first, then column
bool compare_points(Point& pointA, Point& pointB, float proximity_tolerance) {
	if (pointA.y < pointB.y - proximity_tolerance) return true;
	if (pointA.y > pointB.y + proximity_tolerance) return false;
	if (pointA.x < pointB.x - proximity_tolerance) return true;
	return false;
}

// returns true if the upper left corner of a is < ulc of b
bool compare_square_t(square_t& a, square_t& b) {
	Point pointA = a[0], pointB = b[0];
	return compare_points(pointA, pointB, 50);
}

int upperLeft(square_t& sq) {
	int idx = 0;
	Point min = sq[0];
	for (int i = 1; i < sq.size(); i++) {
		if (sq[i].x <= min.x && abs(sq[i].y - min.y) < 20 ) {
			min = sq[i];
			idx = i;
		}
	}
	return idx;
}


void printSquares(vector<square_t>& squares, string filename) {
	ofstream myfile2;
	myfile2.open("C:/Users/sbeaulie/Desktop/" + filename);

	//Le rotate ne marchait pas
	for (square_t& sq : squares) {
		for (int i = 0; i < sq.size(); i++) {
			myfile2 << sq[i] << endl;
		}
		myfile2 << endl;
	}

	myfile2.close();
}

void filterBySize(vector<square_t>& in, vector<square_t>& out) {
	
	for (int i = 0; i < in.size(); i++)
	{
		//def des points
		Point p1 = in[i][0];
		Point p2 = in[i][1];
		Point p3 = in[i][2];
		Point p4 = in[i][3];

		//calcul de taille du carré
		int distancex = (p2.x - p1.x) ^ 2;
		int distancey = (p2.y - p1.y) ^ 2;

		double width = sqrt(abs(distancex - distancey));

		if (width > 15.5 && width<17) {
			vector<Point> points;
			points.push_back(p1);
			points.push_back(p2);
			points.push_back(p3);
			points.push_back(p4);
			out.push_back(points);
		}
	}
}

// rotate each square so that the upper left corner is the first point
void rotateSquares(vector<square_t>& squaresBis) {
	for (square_t& sq : squaresBis) {
		int mouv = upperLeft(sq);
		vector<Point> inter = sq;
		sq[(0 + mouv) % 4] = inter[0];
		sq[(1 + mouv) % 4] = inter[1];
		sq[(2 + mouv) % 4] = inter[2];
		sq[(3 + mouv) % 4] = inter[3];
	}
}


bool squaresOverlap(square_t& a, square_t& b, float tolX, float tolY) {
	Point pointA = a[0], pointB = b[0];
	return abs(pointB.x - pointA.x) < 160 && abs(pointB.y - pointA.y) < 320;
}

void filterOverlappingSquares(vector<square_t>& in, vector<square_t>& out, float tolX, float tolY) {
	std::sort(in.begin(), in.end(), compare_square_t);

	for (auto sq : in) {
		if (out.size() == 0
			|| !squaresOverlap(out[out.size() - 1], sq, tolX, tolY)) {
			out.push_back(sq);
		}
	}
}


bool areSameRow(square_t a, square_t b, float tolY) {
	return std::abs(a[0].y - b[0].y) < tolY;
}

vector<vector<square_t>> groupByRow(vector<square_t>& squares) {
	//Trier les carrés par lignes
	vector<vector<square_t>> lignes;

	std::sort(squares.begin(), squares.end(), compare_square_t);

	vector<square_t> currentRow;

	currentRow.push_back(squares[0]);
	for (int i = 1; i < squares.size(); i++) {
		cout << "i: " << i << endl;
		if (!areSameRow(squares[i], squares[i - 1], 150)) {
			cout << "changeRow " << i << endl;
			lignes.push_back(currentRow);
			currentRow = vector<square_t>();
		}
		currentRow.push_back(squares[i]);
	}
	return lignes;
}


int main(int /*argc*/, char** /*argv*/)
{
	const string numero = "00104";
	const string imgPath = "c:/Users/sbeaulie/Desktop/Projet OpenCV-CMake/images";
	string alt = imgPath + "/" + numero + ".png";

	char * cstr = new char[alt.length() + 1];
	std::strcpy(cstr, alt.c_str());

	static const char* names[] = { cstr ,0 };
	
	help();
	namedWindow(wndname, WINDOW_NORMAL);
	vector<square_t> squares;

	for (int i = 0; names[i] != 0; i++)
	{
		Mat image = imread(names[i], 1);
		if (image.empty())
		{
			cout << "Couldn't load " << names[i] << endl;
			continue;
		}

		findSquares(image, squares);
	

		vector<square_t> squaresBis;
		filterBySize(squares, squaresBis);
		rotateSquares(squaresBis);

		vector<square_t> filtered;
		filterOverlappingSquares(squaresBis, filtered, 160, 320);

		
		ofstream myfile;
		myfile.open("C:/Users/sbeaulie/Desktop/example.txt");
		for (int i = 0; i < filtered.size();i++) {
			if (i % 7 == 0) myfile << endl;
			myfile << filtered[i] << endl;
		}
		myfile.close();

	

		cout << filtered.size()<< endl;

		//drawSquares(image, filtered);
		
		auto lignes = groupByRow(filtered);

		drawSquares(image, lignes[3], Scalar(0, 255, 0));
		
		

		//imwrite( "out", image );
		int c = waitKey();
		if ((char)c == 27)
			break;
	}
	delete[] cstr;
	return 0;
}