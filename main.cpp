// Extracción de mediciones en video (Cinemetrics)
// C++ , Open CV
// 2018-2020 No Budget Animation S de RL de CV.

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <core.hpp>
#include <ocl.hpp>
#include <plot.hpp>
#include "metrics.h"
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

String movieName;
String movieExt;
String corpus;
String mainLocation;
String dataFolder;
int totalFrames;
int noFrames;
//EVERY JumpFactor SECOND: 1 = 1 Second , 0 = each couple of frame
int jumpFactor = 1;
// –––––––––––––––––––––––
int noSamples;

int main(){
    mainLocation = "/Volumes/ORION HD/PROJECTS/PROGRAMAS";
    movieName = "Conavim";
    movieExt = ".mp4";
    corpus = "/Corpus/Misc/";
    dataFolder = mainLocation+"/Data/";
    
    String Moviefile = mainLocation+corpus+movieName+movieExt;
    
    
    /* CREATE DIRECTORIES FOR DATA*/
    
    int dir01;
    String dir01Loc = dataFolder+"00_Data_Output/02_ColorPaletes/"+movieName;
    dir01 = mkdir(dir01Loc.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    int dir02;
    String dir02Loc=dataFolder+"02_ImageFrames/"+movieName;
    dir02 = mkdir(dir02Loc.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    cout << "OpenCV Version " << CV_VERSION << endl;
    
    VideoCapture cap(Moviefile);
    int jumpFrame = 1;
    cap.set(CAP_PROP_POS_FRAMES,jumpFrame);
    
    if (!cap.isOpened()){
        cout << "No se pudo abrir el archivo de video. \n";
        return -1;
    }
    
    noFrames = cap.get(cv::CAP_PROP_FRAME_COUNT); // numero total de cuadros
    totalFrames = noFrames;
    
    double fps = cap.get(cv::CAP_PROP_FPS); // cuadros por segundo
    
    int iLowH = 0;
    int iHighH = 128;
    
    int iLowS = 0;
    int iHighS = 128;
    
    int iLowV = 0;
    int iHighV = 128;
    
    
    cout << "Corriendo a: " << fps << " cuadros por segundo \n";
    cout << "El video contiene: " << totalFrames << " cuadros \n";
    if (jumpFactor ==0){
        noSamples = totalFrames;
        
    }
    else{
        noSamples = totalFrames / fps / jumpFactor;
        
    }
    
    setColorMapMat(noSamples);
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
        //        cap.read(frameB);
        
        if (!cap.read(frameB)){
            cout << "No se puede leer correctamente el archivo de video. O el video ha terminado \n";
            waitKey(0);
            break;
        }
        
        int actFrame = cap.get(cv::CAP_PROP_POS_FRAMES); // frame actual
        cout << "El cuadro actual es: " << actFrame << " \n";
        
        int resizeFactor = 4;
        Size sSize(frameA.cols/resizeFactor, frameA.rows/resizeFactor);
        Mat sframe;
        
        resize(frameA,sframe,sSize);
        
        String savePath = dataFolder+"02_ImageFrames/"+movieName+"/frame_"+ to_string(actFrame) +".jpg";
        
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
        
        getClusteredColors(actFrame, noSamples);
        //detectShot(frameA, frameB, actFrame, noFrames);
        //detectShot_m2(frameA, frameB, actFrame);
        //ShotFramesCount();
        if (jumpFactor ==0){
            jumpFrame = jumpFrame + 1;
        }
        else{
            jumpFrame = jumpFrame + (fps*jumpFactor);
        }
        
        cout << "jumpFrame:  "<< jumpFrame << "\n";
        cap.set(CAP_PROP_POS_FRAMES,jumpFrame);
    }
    
    return 0;
}

// Fin del programa

