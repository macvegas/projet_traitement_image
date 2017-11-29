//////////////////////////////////////////////////////////////////////////
// Option Images
// Projet, séance 1
// thème : premier pas en OpenCV
// contenu : charge, affiche, réduction, calcul et affichage d'histogramme
// version : 17.1128
//////////////////////////////////////////////////////////////////////////


#include <iostream>
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

#include "histogram.hpp"


int main (void) {

	//charge et affiche l'image (à MODIFIER) :
	string imName = "C:/Users/sbeaulie/Desktop/Projet OpenCV-CMake/images/Lenna.png";
	Mat im = imread(imName);
	if(im.data == nullptr){
		cerr << "Image not found: "<< imName << endl;
		waitKey(0);
		//system("pause");
		exit(EXIT_FAILURE);
	}
	imshow("exemple1", im);

	//applique une reduction de taille d'un facteur 5
	//ici modifier pour ne reduire qu'a l'affichage 
	//comme demande dans l'enonce
	int reduction = 5;
	Size tailleReduite(im.cols/reduction, im.rows/reduction);
	Mat imreduite = Mat(tailleReduite,CV_8UC3); //cree une image à 3 canaux de profondeur 8 bits chacuns
	resize(im,imreduite,tailleReduite);
	imshow("image reduite", imreduite);

	computeHistogram("histogramme", im);

	//termine le programme lorsqu'une touche est frappee
	waitKey(0);
	//system("pause");
	return EXIT_SUCCESS;
}
