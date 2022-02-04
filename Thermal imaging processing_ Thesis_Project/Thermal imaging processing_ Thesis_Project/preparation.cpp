/* UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE*/
/*ALGORITMO DE PREPARACION DE DATOS DE ENTRENAMIENTIO PARTIR DE IMAGENES TERMICAS*/
/*AUTORES: -BYRON JIMENEZ E INTI TOALOMBO*/
/*SEPTIEMBRE 2017*/
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;
int numeroimagenes = 0;
int ancho = 32;
String rutaimagenes;
String rutadirectorio_p = "E:/ESPE/TESIS/MUESTRAS/I2+/";
String rutadirectorio_n = "E:/ESPE/TESIS/MUESTRAS/I2-/";
String salidaentrenamiento = "C:/xml/entrenamientoi4.ocv";
int mattoarray[1024];
int aux;
Mat img;
Mat imgescalada(ancho, ancho, CV_32F);
Mat recortada;
int arreglo[300][2];
void recortarimagen();
int main()
{
	fstream fs(salidaentrenamiento, ios::out);
	namedWindow("Original", WINDOW_AUTOSIZE);
	namedWindow("Escalada", WINDOW_AUTOSIZE);
	numeroimagenes = 48;//imagenes binarias positivas
	for (int i = 1; i <= numeroimagenes; i++) {
		rutaimagenes = rutadirectorio_p + "p (" + to_string(i) + ").png";
		cout << rutaimagenes << endl;
		img = imread(rutaimagenes, IMREAD_GRAYSCALE);
		//imshow("original",img);
		recortarimagen();
		resize(recortada, imgescalada, Size(ancho, ancho));
		imshow("Escalada", imgescalada);
		int j = 0;
		for (int x = 0; x < ancho; x++)
		{
			for (int y = 0; y < ancho; y++)//scar los pixeles
			{
				mattoarray[j] = (imgescalada.at<uchar>(x, y) > 100) ? 1 : 0;
				if (x == 31 & y == 31) {
					fs << mattoarray[j] << ",1";
				}
				else fs << mattoarray[j] << ",";
				j++;
			}
		}
		fs << "\n";
	}
	numeroimagenes = 24;// numero de imagenes binarias negativas
	for (int i = 1; i <= numeroimagenes; i++) {
		rutaimagenes = rutadirectorio_n + "n (" + to_string(i) + ").png";
		cout << rutaimagenes << endl;
		img = imread(rutaimagenes, IMREAD_GRAYSCALE);
		if (img.empty()) cout << "No hay Imagen" << endl;
		recortarimagen();
		resize(recortada, imgescalada, Size(ancho, ancho));
		imshow("Escalada", imgescalada);
		int j = 0;
		for (int x = 0; x < ancho; x++)
		{
			for (int y = 0; y < ancho; y++)
			{
				mattoarray[j] = (imgescalada.at<uchar>(x, y) > 100) ? 1 : 0;
				if (x == 31 & y == 31) {
					fs << mattoarray[j] << ",0";
					fs << "\n";
				}
				else fs << mattoarray[j] << ",";
				j++;
			}
		}
	}
	cout << "Generacion Completada de los Datos de Entrenamiento, revise su carpeta de destino" << endl;
	fs.close();
	return 0;
}
void recortarimagen() {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat imagen = img;
	findContours(imagen, contours, RETR_TREE, CHAIN_APPROX_NONE);
	imshow("Original", img);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	cout << "Contornos: " + to_string(contours.size()) << endl;
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		arreglo[i][0] = boundRect[i].width;
		arreglo[i][1] = i;
	}
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 1; j < contours.size(); j++) {
			if (arreglo[j][0] >= arreglo[j - 1][0]) {
				aux = arreglo[j][0];
				arreglo[j][0] = arreglo[j - 1][0];
				arreglo[j - 1][0] = aux;
				aux = arreglo[j][1];
				arreglo[j][1] = arreglo[j - 1][1];
				arreglo[j - 1][1] = aux;
			}
		}
	}
	// cout << "Ordenamiento i: "; cout<<arreglo[0][1] << endl;
	// cout << "Ordenamiento heigh: "; cout<<arreglo[0][0] << endl;
	cout << "Recortando" << endl;
	int posicion = arreglo[0][1];
	recortada = img(boundRect[posicion]);
	if (contours.size() == 0) recortada = img;
	namedWindow("Recortada", WINDOW_AUTOSIZE);
	imshow("Recortada", recortada);
	waitKey(0);
}
