/*UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE*/
/*ALGORITMO DE DETECCION Y CONTEO DE PERSONAS A PARTIR DE IMAGENES TERMICAS*/
/*AUTORES: -BYRON JIMENEZ E INTI TOALOMBO*/
/*SEPTIEMBRE 2017*/
//LIBRERIAS
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/videoio.hpp>
#include<opencv2\video\video.hpp>
#include<opencv2\ml\ml.hpp>
#include "jackylib.h" // Esta libreria es usada para convertir string2char, int2string & Mat2Bmp
#include <stdio.h>
#include <iostream>
#include <conio.h>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
bool aplastarcontar = false; //Se activa con el boton contar, Permite que el proceso contar se ejecuta si es verdadero entonces se empieza a contar, si es falso no cuenta
bool procesar = false; //Se activa con el boton procesar y si esta activado procesa, sino no procesa
bool abrir = false; //Abrir le utiizo para cuando se haya abierto correctamente una camara o un video
bool abrirg = false;
bool reproducir = false; //Se activa con el boton reproducir y si esta activado reproduce el video, sino entonces pausa la reproduccion de video
bool coincidencia = false; //Cuando se ejecuta la busqueda de patrones coincidentes, este se activa, sino se activa, considera que es una nueva persona y cuenta
bool resetear = false; //Se activa al aplastar el Boton resetear y sirve para resetear los objetos VideoCapture y Videowriter y sirve para volver al estado inicial las variables y botones
bool terminarprocesamiento = false; //Se activa cuando se aplasta el Boton Detener Procesamiento o el VIdeo se acaba y termina el procesamiento para mostrar los resultados en el VIdeoWriter y en el TXT
int area = 0, contp = 1; //Area, saca el area del objeto que se esta analizando, contp es el contador de persona y asigna el id a los objetos
double framestotal = 0, frameactual = 0; //Variables progressbar, framestotal obtiene el numero total de cuadros que tiene el video, framesactual, saca el cuadro actual que se se esta procesando
float fps = 0.0; //FPS obtiene la tasa de frames por segundo del video y sirve para cuadrar los valores de tiempo cuando se muestra los resultados.
int contador = 0;
//Variable para comprobar el error si la otra detección corresponde a la misma persona o no
float porcentajeacumulado = 0.0; //Variable donde se almacena el porcentaje acumulado de la probabilidad que corresponda a una persona detectada anteriormente
float probx = 0.0; //Saca el error de posicion en el ejex
float proby = 0.0; //Saca el error de posicion en el ejey
float probaltura = 0.0; //Saca el error de posicion en el probaltura
const double porcentajeaceptable = 0.85; //Porcentaje acumulado minimo para que una persona o figura sea considerada como una persona anterior
float maxarea = 0; //Variable para sacar el maximo valor de pixeles negros acumulados en la imagen
const float maxareainvividual = 1500; //Con mas de 6000 pixeles, se borra el contorno que lo sobrepasa
std::string rutaposible; //Ruta cargada del video
std::string rutaposibleg;
std::string rutaguardar; //Ruta para guardar video
std::string rutamuestra;
#using <System.dll>
#using <System.Windows.Forms.dll>
#using <System.Drawing.dll>
namespace interface1 {
	using namespace System;
	using namespace System::Threading;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Runtime::InteropServices;
	using namespace std;
	using namespace cv;
	using namespace jacky_lib;
	//Variables para poner un ancho y alto de frames
	const int FRAME_WIDTH = 320;
	const int FRAME_HEIGHT = 240;
	Mat src, srcImg, logo, gps;
	Mat imagenpresentacion;// Video que se presenta en la interface
	Mat coordenadas;
	Mat procesado, infraGris, temp, resultados; // inicializar mat necesarios
	int cuadros = 0;
	string valor = "";
	mat2picture mat2bmp; //Transforma de objeto Mat a objeto BMP para que se pueda mostrar en la Interfaz de Visual Studio
	VideoCapture video; //Obtiene video de la camara o de un archivo de video
	VideoCapture videog;
	VideoWriter writer; //Objeto que permite escribir el video de los resultados
	//Se tiene se cargar solo una vez, al principio del programa, proque va a ser la misma
	FileStorage ffs("C:\\xml\\entrenamientolag.xml", FileStorage::READ);// cargar arhchivo .xml que contiene todas las caracterisitcas de la RNA entrenada
	Ptr<cv::ml::ANN_MLP> nn = Algorithm::read<cv::ml::ANN_MLP>(ffs.root());// Creacion del objeto del MLP Multilayer Perceptron
	//fin cargar red
	//pasar por red
	float arrayentrenamiento[1][1024];//vector columna de todos los objetos del Mat del escalado
	vector<vector<cv::Point>> contours;//almacenar los puntos que da el findcontourns
	vector<Vec4i> hierarchy;
	Mat contornounico, contornounicodimensionado, muestra; //Sacar los objetos individuales y hacer una redimensionamiento para entrar a la red
	Rect recta; //Rectangulo que almacena las propiedades de la recta que corresponde al contorno que se esta procesando
	Mat inputTrainingData(1, 1024, CV_32F);// crera Mat con los datos de entrenamiento
	//Este objeto permite almacenar las propiedes de las figuras que entran o no a la red, para analizar si corresponde a la misma persona o no.
	struct persona {
		Rect rectafigura; //Rectangulo que almacena el x,y, altura, ancho.
		int id = 0; //Variable para identificar a cada persona reconocida
		float probabilidadfigura = 0.0; //Probabilidad que pertenezca a un objeto reconocido por la Red o no
		int idfigura = 0.0; //me sirve para imprimir, porque almacena el valor del contorno reconocido
		int frameinicio = 0; //Primer frame donde aparece la persona
		int framefin = 0; //Frame ultimo que se guardo hasta donde aparecio la persona
	};
	//Creo los vectores anterior y actual para guardar los personas que han sido reconocidas en el cuadro anterior y en el cuadro anterior
	vector<persona> personaanterior; //Todas las personas reconocidas en el Cuadro Anterior
	vector<persona> personaactual; //Personas que se han añadido o que se han agregado del vector anterior en el Cuadro Actual
	persona caracteristicasfiguras; //Creo una estructura para guardar las propiedades de cada contorno procesado.
	//Esta estructura almacena las propiedades de las personas reconocidas a lo largo de todo el video
	struct personageotiquetada {
		int id = 0; //ID de la persona encontrada
		int frameinicio = 0; //Frame donde fue la primera deteccion
		int framefin = 0; //Frame donde fue la ultima deteccion
	};
	//Creamos una estructura, pera guardar los datos para ingresar al vector
	personageotiquetada personareconocida;
	vector<personageotiquetada> conteopersonas; //Vector mas importante, contiene las personas contadas y su ubicacion en el tiempo del video.
	/// <summary>
	/// Summary for MainForm
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::ProgressBar^ progressBar1;
	private:
	public: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::TextBox^ textBoxCargarVideo;
	public:
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog2;
	private: System::Windows::Forms::TextBox^ textBox2;
	public: System::Windows::Forms::Button^ btnDetenerProcesamiento;
	private:
	private: System::Windows::Forms::SaveFileDialog^ saveFileDialog1;
	private: System::Windows::Forms::ComboBox^ comboBoxFuenteVideo;
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::Label^ label8;
	private: System::Windows::Forms::Button^ btnResetear;
	private: System::Windows::Forms::Button^ btnSalir;
	private: System::Windows::Forms::GroupBox^ grupoArchivovideo;
	private: System::Windows::Forms::Label^ label9;
	private: System::Windows::Forms::GroupBox^ GrupoCamara;
	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::ComboBox^ comboBoxCamara;
	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	private: System::Windows::Forms::StatusStrip^ statusStrip1;
	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::Label^ label12;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::GroupBox^ groupBox3;
	public: System::Windows::Forms::PictureBox^ ptbLogoEspe;
	private:
	public: System::Windows::Forms::Label^ label4;
	public: System::Windows::Forms::PictureBox^ ptbLogoMecatronica;
	public: System::Windows::Forms::PictureBox^ pictureBox1;
	public: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::PictureBox^ ptbcoordenadas;
	private: System::Windows::Forms::GroupBox^ Reporte;
	private: System::Windows::Forms::Label^ label13;
	private: System::Windows::Forms::TextBox^ textBoxGPS;
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::GroupBox^ groupBox4; private: System::ComponentModel::BackgroundWorker^ backgroundWorker2; //Crea un thread para realizar procesamiento paralelo para procesar dos cosas al mismo tiempo
//Primer hilo corresponde al procesamiento de la interfaz, el segundo hilo corresponde al procesamiento del video
	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}
	public: System::Windows::Forms::Button^ btnBrowse;
	public: System::Windows::Forms::Button^ btnProcess;
	public: System::Windows::Forms::PictureBox^ ptbSource;
	public: System::Windows::Forms::Label^ label1;
	public: System::Windows::Forms::PictureBox^ ptbProcess;
	public: System::Windows::Forms::Label^ label2;
	public: System::Windows::Forms::Button^ btnContar;
	public: System::Windows::Forms::Label^ label5;
	public: System::Windows::Forms::Button^ btnAbrirVideo;
	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;
#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MainForm::typeid));
			this->btnBrowse = (gcnew System::Windows::Forms::Button());
			this->btnProcess = (gcnew System::Windows::Forms::Button());
			this->ptbSource = (gcnew System::Windows::Forms::PictureBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->ptbProcess = (gcnew System::Windows::Forms::PictureBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->btnContar = (gcnew System::Windows::Forms::Button());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->btnAbrirVideo = (gcnew System::Windows::Forms::Button());
			this->backgroundWorker2 = (gcnew System::ComponentModel::BackgroundWorker());
			this->progressBar1 = (gcnew System::Windows::Forms::ProgressBar());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->textBoxCargarVideo = (gcnew System::Windows::Forms::TextBox());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->openFileDialog2 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->btnDetenerProcesamiento = (gcnew System::Windows::Forms::Button());
			this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			this->comboBoxFuenteVideo = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->btnResetear = (gcnew System::Windows::Forms::Button());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->btnSalir = (gcnew System::Windows::Forms::Button());
			this->grupoArchivovideo = (gcnew System::Windows::Forms::GroupBox());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->GrupoCamara = (gcnew System::Windows::Forms::GroupBox());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->comboBoxCamara = (gcnew System::Windows::Forms::ComboBox());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->statusStrip1 = (gcnew System::Windows::Forms::StatusStrip());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->ptbLogoEspe = (gcnew System::Windows::Forms::PictureBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->ptbLogoMecatronica = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->ptbcoordenadas = (gcnew System::Windows::Forms::PictureBox());
			this->Reporte = (gcnew System::Windows::Forms::GroupBox());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->textBoxGPS = (gcnew System::Windows::Forms::TextBox());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbSource))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbProcess))->BeginInit();
			this->groupBox1->SuspendLayout();
			this->grupoArchivovideo->SuspendLayout();
			this->GrupoCamara->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbLogoEspe))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbLogoMecatronica))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbcoordenadas))->BeginInit();
			this->Reporte->SuspendLayout();
			this->groupBox4->SuspendLayout();
			this->SuspendLayout();
			//
			// btnBrowse
			//
			this->btnBrowse->Enabled = false;
			this->btnBrowse->Location = System::Drawing::Point(16, 21);
			this->btnBrowse->Name = L"btnBrowse";
			this->btnBrowse->Size = System::Drawing::Size(149, 45);
			this->btnBrowse->TabIndex = 0;
			this->btnBrowse->Text = L"Reproducir Video";
			this->btnBrowse->UseVisualStyleBackColor = true;
			this->btnBrowse->Click += gcnew System::EventHandler(this, &MainForm::btnBrowse_Click);
			//
			// btnProcess
			//
			this->btnProcess->Enabled = false;
			this->btnProcess->Location = System::Drawing::Point(16, 79);
			this->btnProcess->Name = L"btnProcess";
			this->btnProcess->Size = System::Drawing::Size(149, 40);
			this->btnProcess->TabIndex = 1;
			this->btnProcess->Text = L"Procesar Vidéo";
			this->btnProcess->UseVisualStyleBackColor = true;
			this->btnProcess->Click += gcnew System::EventHandler(this, &MainForm::btnProcess_Click);
			//
			// ptbSource
			//
			this->ptbSource->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->ptbSource->Location = System::Drawing::Point(337, 303);
			this->ptbSource->Name = L"ptbSource";
			this->ptbSource->Size = System::Drawing::Size(431, 275);
			this->ptbSource->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->ptbSource->TabIndex = 2;
			this->ptbSource->TabStop = false;
			this->ptbSource->Click += gcnew System::EventHandler(this, &MainForm::ptbSource_Click);
			//
			// label1
			//
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label1->Location = System::Drawing::Point(422, 275);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(215, 25);
			this->label1->TabIndex = 3;
			this->label1->Text = L"Toma Aérea Térmica";
			this->label1->Click += gcnew System::EventHandler(this, &MainForm::label1_Click);
			//
			// ptbProcess
			//
			this->ptbProcess->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->ptbProcess->Location = System::Drawing::Point(776, 303);
			this->ptbProcess->Name = L"ptbProcess";
			this->ptbProcess->Size = System::Drawing::Size(443, 275);
			this->ptbProcess->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->ptbProcess->TabIndex = 4;
			this->ptbProcess->TabStop = false;
			//
			// label2
			//
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label2->Location = System::Drawing::Point(910, 275);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(196, 25);
			this->label2->TabIndex = 5;
			this->label2->Text = L"Video Segmentado";
			//
			// btnContar
			//
			this->btnContar->Enabled = false;
			this->btnContar->Location = System::Drawing::Point(16, 176);
			this->btnContar->Name = L"btnContar";
			this->btnContar->Size = System::Drawing::Size(149, 40);
			this->btnContar->TabIndex = 12;
			this->btnContar->Text = L"Contar Personas";
			this->btnContar->UseVisualStyleBackColor = true;
			this->btnContar->Click += gcnew System::EventHandler(this, &MainForm::btnContar_Click);
			//
			// label5
			//
			this->label5->AutoSize = true;
			this->label5->Font = (gcnew System::Drawing::Font(L"Modern No. 20", 25.8F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->label5->Location = System::Drawing::Point(33, 382);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(151, 45);
			this->label5->TabIndex = 15;
			this->label5->Text = L"MENÚ";
			this->label5->Click += gcnew System::EventHandler(this, &MainForm::label5_Click);
			//
			// btnAbrirVideo
			//
			this->btnAbrirVideo->Enabled = false;
			this->btnAbrirVideo->Location = System::Drawing::Point(563, 18);
			this->btnAbrirVideo->Name = L"btnAbrirVideo";
			this->btnAbrirVideo->Size = System::Drawing::Size(149, 35);
			this->btnAbrirVideo->TabIndex = 16;
			this->btnAbrirVideo->Text = L"Abrir";
			this->btnAbrirVideo->UseVisualStyleBackColor = true;
			this->btnAbrirVideo->Click += gcnew System::EventHandler(this, &MainForm::btnAbrir_Click);
			//
			// backgroundWorker2
			//
			this->backgroundWorker2->WorkerReportsProgress = true;
			this->backgroundWorker2->WorkerSupportsCancellation = true;
			this->backgroundWorker2->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &MainForm::backgroundWorker2_DoWork);
			this->backgroundWorker2->ProgressChanged += gcnew System::ComponentModel::ProgressChangedEventHandler(this, &MainForm::backgroundWorker2_ProgressChanged);
			//
			// progressBar1
			//
			this->progressBar1->Location = System::Drawing::Point(460, 238);
			this->progressBar1->Name = L"progressBar1";
			this->progressBar1->Size = System::Drawing::Size(596, 23);
			this->progressBar1->TabIndex = 17;
			//
			// label6
			//
			this->label6->AutoSize = true;
			this->label6->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 7.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label6->Location = System::Drawing::Point(310, 238);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(132, 17);
			this->label6->TabIndex = 18;
			this->label6->Text = L"Porcentaje Video";
			this->label6->Click += gcnew System::EventHandler(this, &MainForm::label6_Click);
			//
			// label7
			//
			this->label7->AutoSize = true;
			this->label7->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 16.2F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label7->Location = System::Drawing::Point(75, 63);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(25, 32);
			this->label7->TabIndex = 19;
			this->label7->Text = L"-";
			//
			// textBoxCargarVideo
			//
			this->textBoxCargarVideo->Enabled = false;
			this->textBoxCargarVideo->Location = System::Drawing::Point(19, 24);
			this->textBoxCargarVideo->Name = L"textBoxCargarVideo";
			this->textBoxCargarVideo->Size = System::Drawing::Size(520, 22);
			this->textBoxCargarVideo->TabIndex = 22;
			this->textBoxCargarVideo->TextChanged += gcnew System::EventHandler(this, &MainForm::textBox1_TextChanged);
			//
			// openFileDialog1
			//
			this->openFileDialog1->FileName = L"openFileDialog1";
			this->openFileDialog1->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MainForm::openFileDialog1_FileOk);
			//
			// openFileDialog2
			//
			this->openFileDialog2->FileName = L"openFileDialog2";
			this->openFileDialog2->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MainForm::openFileDialog2_FileOk);
			//
			// textBox2
			//
			this->textBox2->Location = System::Drawing::Point(710, 250);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(194, 22);
			this->textBox2->TabIndex = 24;
			//
			// btnDetenerProcesamiento
			//
			this->btnDetenerProcesamiento->Enabled = false;
			this->btnDetenerProcesamiento->Location = System::Drawing::Point(16, 125);
			this->btnDetenerProcesamiento->Name = L"btnDetenerProcesamiento";
			this->btnDetenerProcesamiento->Size = System::Drawing::Size(149, 45);
			this->btnDetenerProcesamiento->TabIndex = 23;
			this->btnDetenerProcesamiento->Text = L"Stop Procesamiento";
			this->btnDetenerProcesamiento->UseVisualStyleBackColor = true;
			this->btnDetenerProcesamiento->Visible = false;
			this->btnDetenerProcesamiento->Click += gcnew System::EventHandler(this, &MainForm::btnDetenerProcesamiento_Click);
			//
			// saveFileDialog1
			//
			this->saveFileDialog1->Filter = L"Text files (*.txt)|*.txt|All files (*.*)|*.*";
			this->saveFileDialog1->Title = L"Guardar datos procesamiento";
			this->saveFileDialog1->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MainForm::saveFileDialog1_FileOk);
			//
			// comboBoxFuenteVideo
			//
			this->comboBoxFuenteVideo->FormattingEnabled = true;
			this->comboBoxFuenteVideo->Items->AddRange(gcnew cli::array< System::Object^ >(2) { L"Cámara", L"Video" });
			this->comboBoxFuenteVideo->Location = System::Drawing::Point(25, 186);
			this->comboBoxFuenteVideo->Name = L"comboBoxFuenteVideo";
			this->comboBoxFuenteVideo->Size = System::Drawing::Size(207, 24);
			this->comboBoxFuenteVideo->TabIndex = 27;
			this->comboBoxFuenteVideo->SelectedIndexChanged += gcnew System::EventHandler(this, &MainForm::comboBox1_SelectedIndexChanged);
			//
			// groupBox1
			//
			this->groupBox1->Controls->Add(this->btnContar);
			this->groupBox1->Controls->Add(this->btnProcess);
			this->groupBox1->Controls->Add(this->btnBrowse);
			this->groupBox1->Controls->Add(this->btnDetenerProcesamiento);
			this->groupBox1->Controls->Add(this->btnResetear);
			this->groupBox1->Location = System::Drawing::Point(25, 437);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(180, 284);
			this->groupBox1->TabIndex = 28;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Controles";
			//
			// btnResetear
			//
			this->btnResetear->AccessibleDescription = L"Permite";
			this->btnResetear->Location = System::Drawing::Point(16, 222);
			this->btnResetear->Name = L"btnResetear";
			this->btnResetear->Size = System::Drawing::Size(149, 45);
			this->btnResetear->TabIndex = 30;
			this->btnResetear->Text = L"Resetear";
			this->btnResetear->UseVisualStyleBackColor = true;
			this->btnResetear->Click += gcnew System::EventHandler(this, &MainForm::btnResetear_Click);
			//
			// label8
//
			this->label8->AutoSize = true;
			this->label8->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 7.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label8->Location = System::Drawing::Point(721, 221);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(123, 17);
			this->label8->TabIndex = 29;
			this->label8->Text = L"Salida de Datos";
			//
			// btnSalir
			//
			this->btnSalir->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10.2F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnSalir->Location = System::Drawing::Point(710, 302);
			this->btnSalir->Name = L"btnSalir";
			this->btnSalir->Size = System::Drawing::Size(194, 45);
			this->btnSalir->TabIndex = 31;
			this->btnSalir->Text = L"Salir";
			this->btnSalir->UseVisualStyleBackColor = true;
			this->btnSalir->Click += gcnew System::EventHandler(this, &MainForm::btnSalir_Click);
			//
			// grupoArchivovideo
			//
			this->grupoArchivovideo->Controls->Add(this->btnAbrirVideo);
			this->grupoArchivovideo->Controls->Add(this->textBoxCargarVideo);
			this->grupoArchivovideo->Location = System::Drawing::Point(294, 157);
			this->grupoArchivovideo->Name = L"grupoArchivovideo";
			this->grupoArchivovideo->Size = System::Drawing::Size(762, 69);
			this->grupoArchivovideo->TabIndex = 32;
			this->grupoArchivovideo->TabStop = false;
			this->grupoArchivovideo->Text = L"Cargar Video";
			//
			// label9
			//
			this->label9->AutoSize = true;
			this->label9->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 7.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label9->Location = System::Drawing::Point(22, 157);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(183, 17);
			this->label9->TabIndex = 33;
			this->label9->Text = L"Seleccion entrada video";
			//
			// GrupoCamara
			//
			this->GrupoCamara->Controls->Add(this->label10);
			this->GrupoCamara->Controls->Add(this->comboBoxCamara);
			this->GrupoCamara->Location = System::Drawing::Point(25, 224);
			this->GrupoCamara->Name = L"GrupoCamara";
			this->GrupoCamara->Size = System::Drawing::Size(207, 59);
			this->GrupoCamara->TabIndex = 34;
			this->GrupoCamara->TabStop = false;
			this->GrupoCamara->Text = L"Cargar Camara";
			//
			// label10
			//
			this->label10->AutoSize = true;
			this->label10->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 7.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label10->Location = System::Drawing::Point(19, 22);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(91, 17);
			this->label10->TabIndex = 37;
			this->label10->Text = L"ID camara: ";
			this->label10->Click += gcnew System::EventHandler(this, &MainForm::label10_Click);
			//
			// comboBoxCamara
			//
			this->comboBoxCamara->Enabled = false;
			this->comboBoxCamara->FormattingEnabled = true;
			this->comboBoxCamara->Items->AddRange(gcnew cli::array< System::Object^ >(3) { L"0", L"1", L"2" });
			this->comboBoxCamara->Location = System::Drawing::Point(132, 19);
			this->comboBoxCamara->Name = L"comboBoxCamara";
			this->comboBoxCamara->Size = System::Drawing::Size(51, 24);
			this->comboBoxCamara->TabIndex = 0;
			this->comboBoxCamara->SelectedIndexChanged += gcnew System::EventHandler(this, &MainForm::comboBoxCamara_SelectedIndexChanged);
			//
			// menuStrip1
			//
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1248, 24);
			this->menuStrip1->TabIndex = 35;
			this->menuStrip1->Text = L"menuStrip1";
			//
			// statusStrip1
			//
			this->statusStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->statusStrip1->Location = System::Drawing::Point(0, 1015);
			this->statusStrip1->Name = L"statusStrip1";
			this->statusStrip1->Size = System::Drawing::Size(1248, 22);
			this->statusStrip1->TabIndex = 36;
			this->statusStrip1->Text = L"statusStrip1";
			//
			// label11
			//
			this->label11->AutoSize = true;
			this->label11->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10.2F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->label11->Location = System::Drawing::Point(15, 24);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(134, 20);
			this->label11->TabIndex = 37;
			this->label11->Text = L"Byron Jiménez";
			this->label11->Click += gcnew System::EventHandler(this, &MainForm::label11_Click);
			//
			// label12
			//
			this->label12->AutoSize = true;
			this->label12->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10.2F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			this->label12->Location = System::Drawing::Point(15, 44);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(122, 20);
			this->label12->TabIndex = 38;
			this->label12->Text = L"Inti Toalombo";
			//
			// groupBox2
			//
			this->groupBox2->Controls->Add(this->label7);
			this->groupBox2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->groupBox2->Location = System::Drawing::Point(710, 48);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(194, 149);
			this->groupBox2->TabIndex = 39;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Num Personas";
			this->groupBox2->Enter += gcnew System::EventHandler(this, &MainForm::groupBox2_Enter);
			//
			// groupBox3
			//
			this->groupBox3->Controls->Add(this->label12);
			this->groupBox3->Controls->Add(this->label11);
			this->groupBox3->Location = System::Drawing::Point(25, 857);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(180, 98);
			this->groupBox3->TabIndex = 40;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"Autores";
			//
			// ptbLogoEspe
			//
			this->ptbLogoEspe->BackColor = System::Drawing::SystemColors::ControlLight;
			this->ptbLogoEspe->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->ptbLogoEspe->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"ptbLogoEspe.Image")));
			this->ptbLogoEspe->Location = System::Drawing::Point(12, 23);
			this->ptbLogoEspe->Name = L"ptbLogoEspe";
			this->ptbLogoEspe->Size = System::Drawing::Size(280, 117);
			this->ptbLogoEspe->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->ptbLogoEspe->TabIndex = 7;
			this->ptbLogoEspe->TabStop = false;
			this->ptbLogoEspe->Click += gcnew System::EventHandler(this, &MainForm::ptbLogo_Click);
			//
			// label4
			//
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"XOUMEG S57", 19.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label4->Location = System::Drawing::Point(331, 43);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(560, 41);
			this->label4->TabIndex = 8;
			this->label4->Text = L"PROTOTIPO DE SISTEMA PARA CONTEO ";
			//
			// ptbLogoMecatronica
			//
			this->ptbLogoMecatronica->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->ptbLogoMecatronica->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"ptbLogoMecatronica.Image")));
			this->ptbLogoMecatronica->Location = System::Drawing::Point(1106, 27);
			this->ptbLogoMecatronica->Name = L"ptbLogoMecatronica";
			this->ptbLogoMecatronica->Size = System::Drawing::Size(122, 117);
			this->ptbLogoMecatronica->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->ptbLogoMecatronica->TabIndex = 9;
			this->ptbLogoMecatronica->TabStop = false;
			//
			// pictureBox1
			//
			this->pictureBox1->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pictureBox1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"pictureBox1.Image")));
			this->pictureBox1->Location = System::Drawing::Point(1108, 150);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(122, 123);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox1->TabIndex = 10;
			this->pictureBox1->TabStop = false;
			//
			// label3
			//
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"XOUMEG S57", 19.8F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label3->Location = System::Drawing::Point(497, 84);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(271, 41);
			this->label3->TabIndex = 41;
			this->label3->Text = L"Y GEOETIQUETADO";
			//
			// ptbcoordenadas
			//
			this->ptbcoordenadas->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->ptbcoordenadas->Location = System::Drawing::Point(0, 35);
			this->ptbcoordenadas->Name = L"ptbcoordenadas";
			this->ptbcoordenadas->Size = System::Drawing::Size(639, 312);
			this->ptbcoordenadas->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->ptbcoordenadas->TabIndex = 42;
			this->ptbcoordenadas->TabStop = false;
			//
			// Reporte
			//
			this->Reporte->Controls->Add(this->groupBox2);
			this->Reporte->Controls->Add(this->btnSalir);
			this->Reporte->Controls->Add(this->ptbcoordenadas);
			this->Reporte->Controls->Add(this->label8);
			this->Reporte->Controls->Add(this->textBox2);
			this->Reporte->Location = System::Drawing::Point(315, 588);
			this->Reporte->Name = L"Reporte";
			this->Reporte->Size = System::Drawing::Size(914, 381);
			this->Reporte->TabIndex = 43;
			this->Reporte->TabStop = false;
			this->Reporte->Text = L"Reporte de Datos";
			//
			// label13
			//
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(9, 24);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(0, 17);
			this->label13->TabIndex = 45;
			//
			// textBoxGPS
			//
			this->textBoxGPS->Location = System::Drawing::Point(12, 58);
			this->textBoxGPS->Name = L"textBoxGPS";
			this->textBoxGPS->Size = System::Drawing::Size(195, 22);
			this->textBoxGPS->TabIndex = 46;
			this->textBoxGPS->TextChanged += gcnew System::EventHandler(this, &MainForm::textBoxGPS_TextChanged);
			//
			// button1
			//
			this->button1->Location = System::Drawing::Point(68, 24);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(124, 30);
			this->button1->TabIndex = 47;
			this->button1->Text = L"Abrir";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MainForm::button1_Click_2);
			//
			// groupBox4
//
			this->groupBox4->Controls->Add(this->button1);
			this->groupBox4->Controls->Add(this->textBoxGPS);
			this->groupBox4->Controls->Add(this->label13);
			this->groupBox4->Location = System::Drawing::Point(12, 289);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Size = System::Drawing::Size(219, 93);
			this->groupBox4->TabIndex = 48;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"Video GPS";
			//
			// MainForm
			//
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::SystemColors::Control;
			this->ClientSize = System::Drawing::Size(1248, 1037);
			this->Controls->Add(this->groupBox4);
			this->Controls->Add(this->Reporte);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->ptbProcess);
			this->Controls->Add(this->groupBox3);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->statusStrip1);
			this->Controls->Add(this->GrupoCamara);
			this->Controls->Add(this->label9);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->comboBoxFuenteVideo);
			this->Controls->Add(this->label6);
			this->Controls->Add(this->progressBar1);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->ptbLogoMecatronica);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->ptbLogoEspe);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->ptbSource);
			this->Controls->Add(this->grupoArchivovideo);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MainForm";
			this->Text = L"Conteo y Geoetiquetado";
			this->Load += gcnew System::EventHandler(this, &MainForm::MainForm_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbSource))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbProcess))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->grupoArchivovideo->ResumeLayout(false);
			this->grupoArchivovideo->PerformLayout();
			this->GrupoCamara->ResumeLayout(false);
			this->GrupoCamara->PerformLayout();
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbLogoEspe))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbLogoMecatronica))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ptbcoordenadas))->EndInit();
			this->Reporte->ResumeLayout(false);
			this->Reporte->PerformLayout();
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();
}
#pragma endregion
//Inicio del evento Reproducir Video -> btnBrowse, se ejecuta una vez al aplastar el Boton Reproducir
private: System::Void btnBrowse_Click(System::Object^ sender, System::EventArgs^ e) {
	//Este evento activa o desactiva la marca reproducir, que reproduce el video
	if (!reproducir) { //Si la marca de reproducir no esta activa, entonces activa la marca
		this->btnBrowse->Text = L"Pausar Video"; //Cambia el texto en el Boton
		this->btnBrowse->Refresh();
		reproducir = true; //Activa la marca
		if (!backgroundWorker2->IsBusy) this->backgroundWorker2->RunWorkerAsync(); /*Arranca todo el procesamiento*/ //Si el Backgroundworker no esta haciendo nada, inicializa el Hilo del Backgroundworker para que empiece a procesar el video
	}
	else { //Si la marca de reproducir esta activa, entonces desactiva la marca
		this->btnBrowse->Text = L"Reproducir Video";
		this->btnBrowse->Refresh();
		reproducir = false;
	}
	btnDetenerProcesamiento->Visible = true;
	btnDetenerProcesamiento->Enabled = true; //Habilita secuencialmente los botones que tienen que hacer las acciones subsecuentes
}
	   //Fin del evento Browse ->btnBrowse
	   //Evento al aplastar el boton abrir, este evento solo se activa cuando haz elegido un video en el selector
private: System::Void btnAbrir_Click(System::Object^ sender, System::EventArgs^ e) {
	bool errorcarga = false;
	abrir = true;
	if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) //Este objeto abre un dialogo para seleccionar un video, si aplasta el Boton Aceptar, entonces inicia la siguiente accion
	{
		textBoxCargarVideo->Text = openFileDialog1->FileName; //Poner la ruta que se obtuvo con el openFileDialog1
		rutaposible = managedStrToNative(openFileDialog1->FileName); //Esta linea convierte de la Variable String^(De Visual Studio) a std::string de la libreria std y guarda en rutaposible
		video.open(rutaposible); //Abrimos esa ruta, con el metodo .open de VideoCapture
		video.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);// Establecer el ancho del frame del video
		video.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//Establecer el alto del frame del video
		fps = video.get(CV_CAP_PROP_FPS); //Obtener los FPS para calcular del Video
		//Esta parte es del VideoWriter, se activan al momento
		int codec = CV_FOURCC('M', 'J', 'P', 'G'); //Selecciones el codec, para poder guardar en distintos formatos del video
		//Para que no se repita el nombre del video procesado, pongo la fecha y la hora al nombre del video
		time_t now = time(0); //Obtiene el tiempo actual para cambiar el nombre del video
		tm* ltm = localtime(&now); //Convierte a un formato de tiempo local
		rutaguardar = "";
		std::string rutadefectovideo = managedStrToNative(Environment::GetFolderPath(Environment::SpecialFolder::MyVideos)); //Obtener la ruta por defecto donde se guardan los videos en Windows
		rutaguardar += "./";
		//System::Diagnostics::Debug::WriteLine("xxxxxxxxxxxxxxxxxxxxxxxxx");
		rutaguardar += to_string(1900 + ltm->tm_year) + "-" + to_string(1 + ltm->tm_mon)
			+ "-" + to_string(ltm->tm_mday) + "-" + to_string(ltm->tm_hour) + ":" + to_string(1 + ltm->tm_min) + ":"
			+ to_string(1 + ltm->tm_sec) + ".avi"; //Guarda el video en la carpeta donde se ejecuta el programa con el siguiente formato video-aaaa-mm-dd-hh:mm::ss.avi
		video >> srcImg; //Saco un cuendro del video para obtener las caracteristicas
		//System::Diagnostics::Debug::WriteLine(gcnew System::String(rutaguardar.c_str()));
		writer.open(rutaguardar, codec, fps, srcImg.size(), true); //Escribo un video con las mismas caracteristicas del video capturado
		if (video.isOpened() && video.get(CV_CAP_PROP_FRAME_COUNT) > 1) { //If nos sirve para cargar solo videos, porque el programa puede cargar imagenenes tambien
			MessageBox::Show("Video agregado correctamente");
			btnBrowse->Enabled = true; //Si es video, se habilita para reproducir
			btnProcess->Enabled = true;
			btnAbrirVideo->Enabled = false; //Desactivo el Boton Abrir para que no poder abrir otro video
			framestotal = video.get(CV_CAP_PROP_FRAME_COUNT); //Obtengo los framestotal para poder visualizar en el Progressbar
		}
		else {
			MessageBox::Show("El Archivo no es un video, revise la ruta!");
		}
	}
}
	   //Inicio del evento Process -> btnProcess es el evento que procesa el video
private: System::Void btnProcess_Click(System::Object^ sender, System::EventArgs^ e) {
	procesar = true; //Habilita la marca par procesar y procesa el video
	btnContar->Enabled = true; //Una vez que se procesa el video, entonces se habilita para que cuente
}
	   //Fin del evento Process ->btnProcess
private: System::Void ptbLogo_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void label1_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void ptbSource_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void MainForm_Load(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
}
	   //Proceso del Boton Contar
public: System::Void btnContar_Click(System::Object^ sender, System::EventArgs^ e) {
	if (aplastarcontar) { //Cuenta cuando se activa la marca aplastar contar, sino no cuenta
		aplastarcontar = false;
		btnContar->Text = "Contar Personas";
	}
	else {
		aplastarcontar = true;
		btnContar->Text = "Dejar contar";
	}
}
private: System::Void ptbContar_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void label3_Click(System::Object^ sender, System::EventArgs^ e) {
}
public: System::Void button1_Click_1(System::Object^ sender, System::EventArgs^ e) {
	if (aplastarcontar) aplastarcontar = false;
	else aplastarcontar = true;
}
private: System::Void label5_Click(System::Object^ sender, System::EventArgs^ e) {
}
	   //Este metodo bgWorker2_DoWork se activa cuando backgroundworker se le pone RunAsync() y empieza el procesamiento del video
private: System::Void backgroundWorker2_DoWork(System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e) {
	if (reproducir) { //Si esta activada, hace algo, sino no hace nada
	//int morph_operator = 0;
		for (;;/*size_t i = 0; i < framestotal; i++*/) { //Inicio del ciclo infinito en caso de video inicio de ciclo de frames para video normal
		// int operation = morph_operator + 7;
			video >> srcImg;
			videog >> gps;
			frameactual = video.get(CV_CAP_PROP_POS_FRAMES);
			if (srcImg.empty()) {
				break;
			}
			srcImg.copyTo(imagenpresentacion);// copiar imagen termica original en Mat imagenpresentacion para que se muestre
			if (procesar) { //Si la marca procesar esta activada, entonces empieza a procesar
			//Inicio Procesamiento del la imagen termica
				medianBlur(srcImg, procesado, 11);
				cvtColor(procesado, infraGris, COLOR_BGR2GRAY);
				threshold(infraGris, temp, 90, 225, THRESH_BINARY | THRESH_OTSU);//buscando el color entre 0 a 80 que tenga negro en escala de grises
				/*medianBlur(srcImg, procesado,5);
				cvtColor(procesado, infraGris, COLOR_BGR2GRAY);//transformar de colores a escala de grises
				//threshold(infraGris, temp, 90, 225, THRESH_BINARY ||THRESH_OTSU);//buscando el color entre 0 a 80 que tenga negro en escala de grises
				threshold(infraGris, temp, 90, 225, THRESH_BINARY & THRESH_OTSU);//buscando el color entre 0 a 80 que tenga negro en escala de grises
				//adaptiveThreshold(infraGris,temp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 5 & THRESH_OTSU);
				//bitwise_not(temp,temp);
				//dilate(temp,temp, MORPH_CROSS);
				Mat element = getStructuringElement(2, cv::Size(2 *7 + 1, 2 * 7 + 1), cv::Point(7, 7));
				morphologyEx(temp,temp, 3, element);
				medianBlur(temp,temp,11);*/
				findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, cv::Point(0, 0));// encontrar contornos
				vector<vector<cv::Point>> contours_poly(contours.size());//para sacar el rectangulo que encierra al controno
				//Fin procesamiento de la imagen termica
				//System::Diagnostics::Debug::WriteLineLine("<<<=============================>>>");
				//imshow("tresh Binary & otsu", temp);
				for (size_t a = 0; a < contours.size(); a++) { //Iterar por los contornos encontrados
					area = contourArea(contours[a]); //Encuentra la mayor area para ver si es una mancha grande o no
					maxarea += (float)area;
					/*System::Diagnostics::Debug::Write("Area Contorno[");
					System::Diagnostics::Debug::Write(a);
					System::Diagnostics::Debug::Write("]: ");
					System::Diagnostics::Debug::WriteLine(area);*/
					if (area >= (int)maxareainvividual) {
						contours.erase(contours.begin() + a);
						//System::Diagnostics::Debug::WriteLine("Contorno Eliminado");
					}
				}
				/*System::Diagnostics::Debug::Write("Max Area: ");
				System::Diagnostics::Debug::WriteLine(maxarea);
				System::Diagnostics::Debug::Write("Area max admisisble del Cuadro: ");
				System::Diagnostics::Debug::WriteLine(0.6*srcImg.rows* srcImg.cols);*/
				if (maxarea >= (float)0.6 * srcImg.rows * srcImg.cols & !aplastarcontar) { //Si la mancha es mayor a las 70% del total del area de la imagen, pone la imagen en negro
					contours.clear(); //Limpia el vector de contornos para que no haga ningun procesamiento
				}
				temp = Mat(srcImg.rows, srcImg.cols, CV_8UC1, Scalar(0)); //Carga todo la matriz temp que tiene los valores segmentados con 0
				/*System::Diagnostics::Debug::Write("Numero Contornos: ");
				System::Diagnostics::Debug::WriteLine(contours.size());*/
				maxarea = 0; //Devuelvo a cero el max area para que vuelva a hacer otra iteracion
				for (size_t a = 0; a < contours.size(); a++) { //Redibujar los contornos encontrados
					drawContours(temp, contours, a, Scalar(255), -1, LINE_AA);// dibujar contornos
				}
				//System::Diagnostics::Debug::WriteLine("<<<=============================>>>");
				//Si se activa la marca aplastar contar
				if (aplastarcontar) {
					for (size_t a = 0; a < contours.size(); a++) { //Iterar por los contornos encontrados
						Mat dibujarcontornos(temp.rows, temp.cols, CV_8UC1, Scalar(0));
						drawContours(dibujarcontornos, contours, a, Scalar(255), -1, LINE_AA);// dibujar contornos
						approxPolyDP(Mat(contours[a]), contours_poly[a], 3, true);//puntos mas extremos
						area = contourArea(contours[a]);//encontrar area de contorno
						recta = boundingRect(Mat(contours_poly[a]));//encuentra la recta a partir de los puntos
						if (((float)recta.width / recta.height <= 0.67) & (recta.height > 5) & (recta.width > 5) & ((float)recta.width / recta.height >= 0.2) & (recta.x + recta.width <= srcImg.cols - 20)
							& (recta.x >= 20) & (recta.y >= 10) & (recta.y + recta.height <= srcImg.rows - 10)) {// condiciones para que inicie el procesamiento
							contornounico = dibujarcontornos(recta);
							resize(contornounico, contornounicodimensionado, cv::Size(32, 32));// redimensionar el contorno encontrado
							int j = 0;
							for (int x = 0; x < 32; x++)
							{
								for (int y = 0; y < 32; y++)
								{
									if (contornounicodimensionado.at<uchar>(x, y) > 1) arrayentrenamiento[0][j] = 1.0;
									else arrayentrenamiento[0][j] = 0.0;
									j += 1;
								}
							}
							inputTrainingData = Mat(1, 1024, CV_32F, arrayentrenamiento, ml::StatModel::UPDATE_MODEL);//ingresar los datos
							nn->predict(inputTrainingData, resultados);// prediccion para el reconocimiento
							caracteristicasfiguras.rectafigura = recta; //Carga la Recta del Objeto
							caracteristicasfiguras.probabilidadfigura = resultados.at<float>(0);
							if (personaanterior.size() > 0) { //Se llena cuando reconoce a una persona, no cuando pasa el primer cuador //En el primer cuadro, el vector persona anterior esta vacio. entonces si esta vacio empieza a llenar, sino empieza a procesar //Comprueba que ya se haya dado una deteccion en el sistema
								if (resultados.at<float>(0) >= 0.60) { //Probamos si se reconocio otra persona 0.80 lag 0.75 illuchi
								//Codigo para verificar si la persona detectada corresponde a una persona anteriormente detectada
									for (size_t p = 0; p < personaanterior.size(); p++) { //Persona Anterior
									////System::Diagnostics::Debug::WriteLine("p: ");
									////System::Diagnostics::Debug::WriteLineLine(p);
										if (personaanterior.at(p).rectafigura.x > caracteristicasfiguras.rectafigura.x) { //Calcula una probabilidad menor a 1 de que la persona corresponda al mismo punto en x de la persona anterior
											probx = (float)caracteristicasfiguras.rectafigura.x / personaanterior.at(p).rectafigura.x;
										}
										else {
											probx = (float)personaanterior.at(p).rectafigura.x / caracteristicasfiguras.rectafigura.x;
										}
										if (personaanterior.at(p).rectafigura.y > caracteristicasfiguras.rectafigura.y) { //Calcula una probabilidad menor a 1 de que la persona corresponda al mismo punto en y de la persona anterior
											proby = (float)caracteristicasfiguras.rectafigura.y / personaanterior.at(p).rectafigura.y;
										}
										else {
											proby = (float)personaanterior.at(p).rectafigura.y / caracteristicasfiguras.rectafigura.y;
										}
										if (personaanterior.at(p).rectafigura.height > caracteristicasfiguras.rectafigura.height) { //Calcula una probabilidad menor a 1 de que la persona corresponda a la misma altura de la persona anterior
											probaltura = (float)caracteristicasfiguras.rectafigura.height / personaanterior.at(p).rectafigura.height;
										}
										else {
											probaltura = (float)personaanterior.at(p).rectafigura.height / caracteristicasfiguras.rectafigura.height;
										}
										porcentajeacumulado = 0.35 * probx + 0.25 * proby + 0.15 * probaltura + 0.25; //0.25 se asigna porque es detectada como persona
										if (porcentajeacumulado >= porcentajeaceptable) {
											////System::Diagnostics::Debug::WriteLineLine("Persona Encontrada en ID: ");
											//System::Diagnostics::Debug::WriteLineLine(personaanterior.at(p).id);
											//System::Diagnostics::Debug::WriteLine("Recta.height: ");
											//System::Diagnostics::Debug::WriteLineLine(recta.height);
											//System::Diagnostics::Debug::WriteLine("Recta.widht: ");
											//System::Diagnostics::Debug::WriteLineLine(recta.width);
											//System::Diagnostics::Debug::WriteLine("Proporcion");
											//System::Diagnostics::Debug::WriteLineLine((float)recta.width/recta.height);
											//System::Diagnostics::Debug::WriteLine("Numero de contorno: ");
											//System::Diagnostics::Debug::WriteLineLine(a);
											caracteristicasfiguras.idfigura = a; //Primero guarda el numero de contorno donde esta la persona para imprimir;
											caracteristicasfiguras.id = personaanterior.at(p).id; //Copia el ID para que se identifique con la misma persona
											caracteristicasfiguras.framefin = frameactual; //Carga la ultima posicion conocida
											personaactual.push_back(caracteristicasfiguras); //Cargar en el vector persona actual porque es la misma persona que anterior
											////System::Diagnostics::Debug::WriteLine("Vector Persona Actual: ");
											////System::Diagnostics::Debug::WriteLineLine(personaactual.size());
											personaanterior.erase(personaanterior.begin() + p); //Elimina el elemento del vector, para que no cuente de nuevo
											conteopersonas.at(caracteristicasfiguras.id - 1).framefin = frameactual;
											coincidencia = true; //Coincidencia se activa cuando: Una persona en el frame actual corresponde con una persona del frame anterior
											break;
										}
									}//Fin del For de busqueda de persona anterior
									if (!coincidencia) { //Crea un nuevo objeto de persona debido a que no hubo coincidencias en el paso anterior
										caracteristicasfiguras.idfigura = a; //Cargamos el numero de contorno para imprimir
										caracteristicasfiguras.id = contp; //Asignamos un nuevo valor de ID a la persona
										caracteristicasfiguras.framefin = frameactual;
										personareconocida.frameinicio = frameactual; //Primera vez que le cuenta
										personareconocida.id = contp; //Cargar el ID
										conteopersonas.push_back(personareconocida); //Metemos al vector
										contp += 1; //Sumamos el contador de personas
									}
									coincidencia = false;//Coincidencia = false, se reinicia ese valor, para permitir que vuelva a hacerse otro ciclo
								}
								else { //Cuando no entra a la red, el objeto todavia puede ser una persona, por eso se procede a hacer otra comprobacion
								////System::Diagnostics::Debug::WriteLineLine("No persona, verificando la figura si corresponde a una persona ");
									for (size_t p = 0; p < personaanterior.size(); p++) { //Iteramos en el Vector persona anterior para ver coincidencias
									////System::Diagnostics::Debug::WriteLine("p: ");
									////System::Diagnostics::Debug::WriteLineLine(p);
										if (personaanterior.at(p).rectafigura.x > caracteristicasfiguras.rectafigura.x) { //Calcula una probabilidad menor a 1 de que la persona corresponda al mismo punto en x de la persona anterior
											probx = (float)caracteristicasfiguras.rectafigura.x / personaanterior.at(p).rectafigura.x;
										}
										else {
											probx = (float)personaanterior.at(p).rectafigura.x / caracteristicasfiguras.rectafigura.x;
										}
										if (personaanterior.at(p).rectafigura.y > caracteristicasfiguras.rectafigura.y) { //Calcula una probabilidad menor a 1 de que la persona corresponda al mismo punto en y de la persona anterior
											proby = (float)caracteristicasfiguras.rectafigura.y / personaanterior.at(p).rectafigura.y;
										}
										else {
											proby = (float)personaanterior.at(p).rectafigura.y / caracteristicasfiguras.rectafigura.y;
										}
										if (personaanterior.at(p).rectafigura.height > caracteristicasfiguras.rectafigura.height) { //Calcula una probabilidad menor a 1 de que la persona corresponda a la misma altura de la persona anterior
											probaltura = (float)caracteristicasfiguras.rectafigura.height / personaanterior.at(p).rectafigura.height;
										}
										else {
											probaltura = (float)personaanterior.at(p).rectafigura.height / caracteristicasfiguras.rectafigura.height;
										}
										porcentajeacumulado = 0.4 * probx + 0.4 * proby + 0.2 * probaltura;
										if (porcentajeacumulado >= porcentajeaceptable) { //Hacer la misma comprobacion de arriba
											caracteristicasfiguras.idfigura = a; //Carga id de la figura para imprmir
											//System::Diagnostics::Debug::WriteLine("Persona Encontrada en ID: ");
											//System::Diagnostics::Debug::WriteLineLine(personaanterior.at(p).id);
											caracteristicasfiguras.id = personaanterior.at(p).id; //Guardamos los mismos valores
											caracteristicasfiguras.framefin = frameactual; //Carga la ultima posicion conocida
											personaactual.push_back(caracteristicasfiguras); //Añadimos al vector
											personaanterior.erase(personaanterior.begin() + p); //Elimina el elemento del vector persona anterior, para que no cuente de nuevo
											conteopersonas.at(caracteristicasfiguras.id - 1).framefin = frameactual;
											break;
										}
									}//Fin del For de busqueda de persona anterior
								}
							}
							else { //Carga por primera vez el vector de personas si hubo deteccion al principio, entonces empieza a llenar el vector
							////System::Diagnostics::Debug::WriteLineLine("Vector personaanterior vacio, llenando");
								if (resultados.at<float>(0) >= 0.60) {
									////System::Diagnostics::Debug::WriteLine("Persona Encontrada, ID: ");
									////System::Diagnostics::Debug::WriteLineLine(contp);
									caracteristicasfiguras.id = contp;
									caracteristicasfiguras.frameinicio = frameactual;
									personaactual.push_back(caracteristicasfiguras);
									////System::Diagnostics::Debug::WriteLine("Persona Actual tamaño: ");
									////System::Diagnostics::Debug::WriteLineLine(personaactual.size());
									personareconocida.frameinicio = frameactual;
									personareconocida.id = contp;
									conteopersonas.push_back(personareconocida);
									contp += 1;
									vector<int> compression_params;
									compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
									compression_params.push_back(9);
									rutamuestra = "E:\\ESPE\\TESIS\\MUESTRAS\\UBICACION\\";
									imwrite(rutamuestra + "persona" + to_string(contp - 1) + ".png", gps, compression_params);
								}
							}
						}
					}//Fin del For Busqueda de Contornos
					//Imprimir las personas encontradas
					for (size_t j = 0; j < personaactual.size(); j++) {
						//System::Diagnostics::Debug::WriteLine("Persona Encontrada en ID: ");
						try {
							drawContours(imagenpresentacion, contours, personaactual.at(j).idfigura, Scalar(0, 0, 255), 3, LINE_8);// Dibujamos el contorno exterior para presentar
						}
						catch (cv::Exception errorcv) {
							//System::Diagnostics::Debug::WriteLineLine("xxxxxxxxxxxxxxxxxxxxxxxxx");
							//System::Diagnostics::Debug::WriteLineLine(errorcv.code);
							std::string codigodeerror = errorcv.msg;
							std::cout << codigodeerror << std::endl;
							System::String^ str2 = gcnew System::String(codigodeerror.c_str()); //Se ejectuta si hay error
							//System::Diagnostics::Debug::WriteLineLine(str2);}
							//cv::putText(imagenpresentacion, "ID:" + to_string(personaactual.at(j).id), cv::Point(personaactual.at(j).rectafigura.x, personaactual.at(j).rectafigura.y), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 1);
						}
						personaanterior.assign(personaactual.begin(), personaactual.end()); //Hago que se cargue el vector Persona Anterior con los datos del vector persona actual para comprobar con el siguiente ciclo
						personaactual.clear(); //Reseteo el Vector de Persona Actual
					}//Fin evento contar
}
//Escribir los cuadros en un video
writer.write(imagenpresentacion);
//Report Progress sirve para actualizar el contenido de la Mat, bmp, botones
backgroundWorker2->ReportProgress(1);
System::Threading::Thread::Sleep(60); //Tiempo de espera para el procesamiento
if (terminarprocesamiento) { //Cuando se aplasta el boton Detener procesamiento se activa y sale del ciclo
	break;
}
}//Acaba de procesar el video
writer.release(); //Libera el objeto de VideoWriter para que pueda escribir otra vez
if (!resetear)backgroundWorker2->ReportProgress(80); //Mandamos a finalizar el proceso
}
}
//Este se activa cuando se ejecuta )backgroundWorker2->ReportProgress(80);
private: System::Void backgroundWorker2_ProgressChanged(System::Object^ sender, System::ComponentModel::ProgressChangedEventArgs^ e) {
	if (reproducir) {
		if (!imagenpresentacion.empty()) {
			ptbSource->Image = mat2bmp.Mat2Bimap(imagenpresentacion); //Muestra la imagen en el PictureBox
			ptbSource->Refresh();
		}
		if (!gps.empty()) {
			ptbcoordenadas->Image = mat2bmp.Mat2Bimap(gps); //Muestra la imagen en el PictureBox
			ptbcoordenadas->Refresh();
		}
		progressBar1->Value = 100 * frameactual / framestotal; //Cargamos el progress bar
		if (procesar & !temp.empty()) {
			if (!temp.empty()) {
				ptbProcess->Image = mat2bmp.Mat2Bimap(temp); //Cargamos el picturebox
				ptbProcess->Refresh();
				int codec = CV_FOURCC('M', 'J', 'P', 'G');
				VideoWriter("E:\\ESPE\\TESIS\\MUESTRAS\\TEMP\\", codec, 5.0, temp.size(), true);
			}
		}
		if (aplastarcontar) {
			label7->Text = System::Convert::ToString(conteopersonas.size()); //Mostrar en el Label el texto de numero de personas
			label7->Refresh();
		}
	}
	//Cuando es 80 el numero de parametro que se envia, entonces hace esto.
	if (e->ProgressPercentage == 80 & aplastarcontar) {
		MessageBox::Show("Finalizo procesamiento de video, se procede a guardar el resultado");
		saveFileDialog1->ShowDialog(); //Mostrar el dialogo para guardar un txt
		if (saveFileDialog1->FileName != "") { //Cuando elegimos un objeto para guardar
			int minuto, segundo; //Minuto y segundo del procesamiento
			MessageBox::Show(saveFileDialog1->FileName); //Muestro donde se va a guardar el archivo de texto
			std::ofstream ofs(managedStrToNative(saveFileDialog1->FileName), std::ofstream::out); //Creo un objeto ofstream para sacar valores en un archivo de text
			ofs << "Geoconteo y Etiquetado de Personas" << "\n"; //Enviar al archivo de texto el siguiente texto
			ofs << "Numero Total de Personas: " << conteopersonas.size() << "\n"; //Imprima el numero de personas identificadas
			ofs << " ID " << " " << "Inicio" << " " << "Final" << "\n"; //Imprime la cabaecera
			for (size_t m = 0; m < conteopersonas.size(); m++) {
				segundo = (int)conteopersonas.at(m).frameinicio / fps; //Calculo los segundos de acuerdo a los fps
				if (segundo > 60) {
					minuto = (int)segundo / 60;
				}
				else {
					minuto = 0;
				}
				ofs << conteopersonas.at(m).id << " " << minuto << ":" << segundo;
				segundo = (int)conteopersonas.at(m).framefin / fps;
				if (segundo > 60) {
					minuto = (int)segundo / 60;
				}
				else {
					minuto = 0;
				}
				ofs << " " << minuto << ":" << segundo << "\n";
			}
			ofs.close(); //Cerramos el objeto ofs ofstream para que nose se quede abierto y poder guardar nuevamente
			MessageBox::Show("Revise la Ruta de Guardado!");
		}
	}
}
private: System::Void label6_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void textBox1_TextChanged(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void folderBrowserDialog1_HelpRequest(System::Object^ sender, System::EventArgs^ e) {
}
	   //Convertir String^ to std::string
	   std::string managedStrToNative(System::String^ sysstr)
	   {
		   using System::IntPtr;
		   using System::Runtime::InteropServices::Marshal;
		   IntPtr ip = Marshal::StringToHGlobalAnsi(sysstr);
		   std::string outString = static_cast<const char*>(ip.ToPointer());
		   Marshal::FreeHGlobal(ip);
		   return outString;
	   }
private: System::Void saveFileDialog1_FileOk(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e) {
}
private: System::Void label10_Click(System::Object^ sender, System::EventArgs^ e) {
}
	   //Esta parte coge los valores del Combobox, donde se elige la camara o la fuente de video
private: System::Void comboBox1_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
	resetear = false;
	if (this->comboBoxFuenteVideo->GetItemText(this->comboBoxFuenteVideo->SelectedItem) == "Cámara") { //Si eligo camara, habilito los controles de camara
		comboBoxCamara->Enabled = true;
		comboBoxFuenteVideo->Enabled = false;
	}
	else if (this->comboBoxFuenteVideo->GetItemText(this->comboBoxFuenteVideo->SelectedItem) == "Video") { //Si eligo video, habilito los controles de video
		btnAbrirVideo->Enabled = true;
		textBoxCargarVideo->Enabled = true;
		comboBoxFuenteVideo->Enabled = false;
	}
}
private: System::Void btnSalir_Click(System::Object^ sender, System::EventArgs^ e) {
	video.release();
	writer.release();
	Application::Exit();
}
	   //Combobox cuando seleccionamos la camara
private: System::Void comboBoxCamara_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
	rutaposible = "";
	rutaposibleg = "";
	if (this->comboBoxCamara->GetItemText(this->comboBoxCamara->SelectedItem) == "0") {
		video.open(0);
		video.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);// ancho del frame del video
		video.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//alto del frame del video
		fps = video.get(CV_CAP_PROP_FPS);
		if (video.isOpened()) {
			MessageBox::Show("Camara 0 agregada correctamente");
			btnBrowse->Enabled = true;
			btnProcess->Enabled = true;
			btnAbrirVideo->Enabled = false;
			framestotal = video.get(CV_CAP_PROP_FRAME_COUNT);
			comboBoxCamara->Enabled = false;
		}
		else {
			MessageBox::Show("Camara 0 no se pudo agregar, revise la conexion o el ID correcto de camara");
		}
	}
	else if (this->comboBoxCamara->GetItemText(this->comboBoxCamara->SelectedItem) == "1") {
		video.open(1);
		video.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);// ancho del frame del video
		video.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//alto del frame del video
		fps = video.get(CV_CAP_PROP_FPS);
		if (video.isOpened()) {
			MessageBox::Show("Camara 1 agregada correctamente");
			btnBrowse->Enabled = true;
			btnProcess->Enabled = true;
			btnAbrirVideo->Enabled = false;
			framestotal = video.get(CV_CAP_PROP_FRAME_COUNT);
			comboBoxCamara->Enabled = false;
		}
		else {
			MessageBox::Show("Camara 1 no se pudo agregar, revise la conexion o el ID correcto de camara");
		}
	}
	else if (this->comboBoxCamara->GetItemText(this->comboBoxCamara->SelectedItem) == "2") {
		video.open(2);
		video.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);// ancho del frame del video
		video.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//alto del frame del video
		fps = video.get(CV_CAP_PROP_FPS);
		if (video.isOpened()) {
			MessageBox::Show("Camara 2 agregada correctamente");
			btnBrowse->Enabled = true;
			btnProcess->Enabled = true;
			btnAbrirVideo->Enabled = false;
			framestotal = video.get(CV_CAP_PROP_FRAME_COUNT);
			comboBoxCamara->Enabled = false;
		}
		else {
			MessageBox::Show("Camara 2 no se pudo agregar, revise la conexion o el ID correcto de camara");
		}
	}
}
private: System::Void btnResetear_Click(System::Object^ sender, System::EventArgs^ e) {
	this->backgroundWorker2->CancelAsync();
	contp = 1;
	video.release();
	writer.release();
	this->btnBrowse->Enabled = false;
	this->btnProcess->Enabled = false;
	this->btnContar->Enabled = false;
	this->btnAbrirVideo->Enabled = false;
	this->textBoxCargarVideo->Enabled = false;
	this->btnDetenerProcesamiento->Enabled = false;
	this->comboBoxCamara->Enabled = false;
	this->comboBoxFuenteVideo->Enabled = true;
	this->comboBoxCamara->ResetText();
	this->comboBoxFuenteVideo->ResetText();
	this->progressBar1->Value = 0;
	this->btnDetenerProcesamiento->Visible = false;
	this->btnDetenerProcesamiento->Enabled = false;
	this->label7->Text = "-";
	aplastarcontar = false;
	conteopersonas.clear();
	personaanterior.clear();
	personaactual.clear();
	procesar = false;
	abrir = false;
	reproducir = false;
	coincidencia = false; //Encontrar o no coincidencia para evitar contar
	ptbSource->Image = nullptr;
	ptbSource->Refresh();
	ptbProcess->Image = nullptr;
	ptbProcess->Refresh();
	ptbcoordenadas->Image = nullptr;
	ptbcoordenadas->Refresh();
	this->statusStrip1->Text = "Control Reseteado";
	this->statusStrip1->Refresh();
	this->btnBrowse->Text = L"Reproducir Video";
}
private: System::Void openFileDialog1_FileOk(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e) {
}
private: System::Void openFileDialog2_FileOk(System::Object^ sender, System::ComponentModel::CancelEventArgs^ e) {
}
private: System::Void btnDetenerProcesamiento_Click(System::Object^ sender, System::EventArgs^ e) {
	terminarprocesamiento = true;
}
private: System::Void label11_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void groupBox2_Enter(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void button1_Click_2(System::Object^ sender, System::EventArgs^ e) {
	bool errorcarga1 = false;
	abrirg = true;
	if (openFileDialog2->ShowDialog() == System::Windows::Forms::DialogResult::OK) //Este objeto abre un dialogo para seleccionar un video, si aplasta el Boton Aceptar, entonces inicia la siguiente accion
	{
		textBoxGPS->Text = openFileDialog2->FileName; //Poner la ruta que se obtuvo con el openFileDialog1
		rutaposibleg = managedStrToNative(openFileDialog2->FileName); //Esta linea convierte de la Variable String^(De Visual Studio) a std::string de la libreria std y guarda en rutaposible
		videog.open(rutaposibleg); //Abrimos esa ruta, con el metodo .open de VideoCapture
		videog.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);// Establecer el ancho del frame del video
		videog.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);//Establecer el alto del frame del video
		fps = videog.get(CV_CAP_PROP_FPS); //Obtener los FPS para calcular del Video
		//Esta parte es del VideoWriter, se activan al momento
		int codec = CV_FOURCC('M', 'J', 'P', 'G'); //Selecciones el codec, para poder guardar en distintos formatos del video
		//Para que no se repita el nombre del video procesado, pongo la fecha y la hora al nombre del video
		time_t now = time(0); //Obtiene el tiempo actual para cambiar el nombre del video
		tm* ltm = localtime(&now); //Convierte a un formato de tiempo local
		videog >> gps; //Saco un cuendro del video para obtener las caracteristicas
//System::Diagnostics::Debug::WriteLine(gcnew System::String(rutaguardar.c_str()));
		if (videog.isOpened() && videog.get(CV_CAP_PROP_FRAME_COUNT) > 1) { //If nos sirve para cargar solo videos, porque el programa puede cargar imagenenes tambien
			MessageBox::Show("Video agregado correctamente");
			btnBrowse->Enabled = true; //Si es video, se habilita para reproducir
			button1->Enabled = false; //Desactivo el Boton Abrir para que no poder abrir otro video
			//framestotal = videog.get(CV_CAP_PROP_FRAME_COUNT); //Obtengo los framestotal para poder visualizar en el Progressbar
		}
		else {
			MessageBox::Show("El Archivo no es un video, revise la ruta!");
		}
	}
}
private: System::Void textBoxGPS_TextChanged(System::Object^ sender, System::EventArgs^ e) {
}
};
}
