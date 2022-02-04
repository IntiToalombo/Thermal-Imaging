/* UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE*/
/*ALGORITMO DE ENTRENAMIENTO DEL PERCEPTRON MULTICAPA CON MUESTRAS OBTENIDAS A PARTIR DE IMAGENES TERMICAS*/
/*AUTORES: -BYRON JIMENEZ E INTI TOALOMBO*/
/*SEPTIEMBRE 2017*/
#include <iostream>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;
int mattoarray[1024];
int numeroimagenes = 48;//numero de imagenes positivas
int ancho = 32;
int elemento;
Mat img;
Mat imgescalada(ancho, ancho, CV_32F);
int acertados = 0;
String rutaimagenes;
String rutadirectorio_p = "E:/ESPE/TESIS/MUESTRAS/I2+/";
float arrayentrenamiento[1][1024];
Mat recortada;
int arreglo[300][2];
int aux;
void recortarimagen(Mat& img) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
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
	int posicion = arreglo[0][1];
	if (contours.size() > 0)recortada = img(boundRect[posicion]);
	else recortada = img;
}
int main(int argc, char* argv[])
{
	namedWindow("IMG Real", WINDOW_AUTOSIZE);
	namedWindow("IMG Escalada", WINDOW_AUTOSIZE);
	Ptr<ANN_MLP> nnetwork = ANN_MLP::create();
	cout << "Leyendo Datos" << endl;
	Ptr<TrainData> datos = TrainData::loadFromCSV("C:/xml/entrenamientoi4.ocv", ROW_SAMPLE, 1024, 1025);
	vector<int> layerSizes = { 1024, //Numero de Entradas
	8, //Capa Oculta
	1 //Capa de Salida, para ver si es maiz o no
	};
	nnetwork->setLayerSizes(layerSizes);
	nnetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	//Entrenamiento
	nnetwork->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.000001));
	nnetwork->setTrainMethod(ANN_MLP::BACKPROP);
	cout << "Creando la RNA" << endl;
	nnetwork->train(datos);
	printf("Entrenado \n");
	FileStorage fs("C:/xml/entrenamientoi4.xml", FileStorage::WRITE);
	nnetwork->write(fs);
	fs.release();
	cout << "Escribiendo XML" << endl;
	Mat resultados;
	for (int i = 1; i <= numeroimagenes; i++) {
		Mat inputTrainingData(1, 1024, CV_32F);;
		rutaimagenes = rutadirectorio_p + "p (" + to_string(i) + ").png";
		cout << rutaimagenes << endl;
		img = imread(rutaimagenes, IMREAD_GRAYSCALE);
		if (!img.data) cout << "IMG no Encontrada" << endl;
		recortarimagen(img);
		resize(recortada, imgescalada, Size(ancho, ancho));
		int j = 0;
		for (int x = 0; x < ancho; x++)
		{
			for (int y = 0; y < ancho; y++)
			{
				if (imgescalada.at<uchar>(x, y) > 1) arrayentrenamiento[0][j] = 1.0;
				else arrayentrenamiento[0][j] = 0.0;
				j += 1;
			}
		}
		inputTrainingData = Mat(1, 1024, CV_32F, arrayentrenamiento, StatModel::UPDATE_MODEL);
		Mat respuesta;
		cout << respuesta << endl;
		nnetwork->predict(inputTrainingData, resultados);
		cout << "=============================" << endl;
		cout << "Resultado entrenamiento : IMGs"; cout << to_string(i); cout << " : "; cout << resultados << endl;
		cout << "=============================" << endl;
		if (resultados.at<float>(0) > 0) {
			putText(img, "Persona", Point(img.cols / 2, 30), FONT_ITALIC, 1.0, Scalar(120), 3, LINE_8);
			acertados += 1;
		}
		imshow("IMG Escalada", imgescalada);
		imshow("IMG Real", img);
		waitKey(0);
	}
	numeroimagenes = 24;//numero de muestras negativas
	String rutadirectorio_n = "E:/ESPE/TESIS/MUESTRAS/I-/";
	for (int i = 1; i <= numeroimagenes; i++) {
		Mat inputTrainingData(1, 1024, CV_32F);;
		rutaimagenes = rutadirectorio_n + "n (" + to_string(i) + ").png";
		cout << rutaimagenes << endl;
		img = imread(rutaimagenes, IMREAD_GRAYSCALE);
		if (!img.data) cout << "IMG no Encontrada" << endl;
		recortarimagen(img);
		resize(recortada, imgescalada, Size(ancho, ancho));
		int j = 0;
		for (int x = 0; x < ancho; x++)
		{
			for (int y = 0; y < ancho; y++)
			{
				if (imgescalada.at<uchar>(x, y) > 1) arrayentrenamiento[0][j] = 1.0;
				else arrayentrenamiento[0][j] = 0.0;
				j += 1;
			}
		}
		inputTrainingData = Mat(1, 1024, CV_32F, arrayentrenamiento, StatModel::UPDATE_MODEL);
		Mat respuesta;
		cout << respuesta << endl;
		nnetwork->predict(inputTrainingData, resultados);
		cout << "=============================" << endl;
		cout << "Resultado entrenamiento : IMGs"; cout << to_string(i); cout << " : "; cout << resultados << endl;
		cout << "=============================" << endl;
		if (resultados.at<float>(0) <= 0.5) {
			putText(img, "No Persona", Point(img.cols / 2, 30), FONT_ITALIC, 1.0, Scalar(120), 3, LINE_8);
			acertados += 1;
		}
		imshow("IMG Escalada", imgescalada);
		imshow("IMG Real", img);
		waitKey(0);
	}
	cout << "Acertados"; cout << acertados << endl;
}