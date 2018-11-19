// Código prueba para extracción de mediciones en video
// C++ , Open CV
// 2018-2019 No Budget Animation S de RL de CV.

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/plot.hpp>
#include "metrics.h"

using namespace cv;
using namespace std;

String movieName;
String movieExt;
int totalFrames;
int noFrames;

int main(){
    
movieName = "DCSPYREM";
movieExt = ".mp4";
    
String Moviefile = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/01_Movies/"+movieName+movieExt;

    cout << "OpenCV Version " << CV_VERSION << endl;
    
    VideoCapture cap(Moviefile);
    cap.set(CV_CAP_PROP_FPS,12);
    
    if (!cap.isOpened()){
        cout << "No se pudo abrir el archivo de video. \n";
    return -1;
    }
    
    noFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); // numero total de cuadros
    totalFrames = noFrames;
    
    double fps = cap.get(CV_CAP_PROP_FPS); // cuadros por segundo
    
    int iLowH = 0;
    int iHighH = 128;
    
    int iLowS = 0;
    int iHighS = 128;
    
    int iLowV = 0;
    int iHighV = 128;
   
    cout << "Corriendo a: " << fps << " cuadros por segundo \n";
    cout << "El video contiene: " << totalFrames << " cuadros \n";
    
    setColorMapMat(totalFrames/3);
    setPlotMat();
    setShotMapMat (350);
    
    while(1){

        Mat frame;
        Mat frameA;
        Mat frameB;
        
        if (!cap.read(frameA)){
            cout << "No se puede leer correctamente el archivo de video. O el video ha terminado \n";
            break;
        }
        
        frameA = frameA;
        cap.read(frameB);
        
       if (!cap.read(frameB)){
            cout << "No se puede leer correctamente el archivo de video. O el video ha terminado \n";
            waitKey(0);
            break;
        }

        int actFrame = cap.get(CV_CAP_PROP_POS_FRAMES); // frame actual
        cout << "El cuadro actual es: " << actFrame << " \n";
        
        int resizeFactor = 4;
        Size sSize(frameA.cols/resizeFactor, frameA.rows/resizeFactor);
        Mat sframe;
        
        resize(frameA,sframe,sSize);
        
        String savePath ="/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/02_ImageFrames/"+movieName+"/frame_"+ to_string(actFrame) +".jpg";
        
        imwrite(savePath, sframe);
        
        Mat imgHSV;
        cvtColor(frameA, imgHSV, COLOR_BGR2HSV);
        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

        
    //namedWindow("Video Original",WINDOW_OPENGL); //crea una ventana
    //imshow("Video Original", frame);
    //resizeWindow("Video Original", 480, 320);
    //imshow("Imagen en Threshold", imgThresholded); //
        
        if(waitKey(30) == 27){
            cout << "EXIT \n";
            break;
        }
        
        //getClusteredColors(actFrame, noFrames);
        detectShot(frameA, frameB, actFrame, noFrames);
        //detectShot_m2(frameA, frameB, actFrame);
        //ShotFramesCount();
    }
    
    return 0;
}

// Fin del programa

