//
//  time.cpp
//  OpenCVTest
//
//  Created by Mipapamedijo on 25/03/18.
//  2018 No Budget Animation S de RL de CV.
//

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

String Shot_Dir, movie_Path;
Mat shotDistPlot;
int font = FONT_HERSHEY_COMPLEX_SMALL;
double fontSize = 0.5;
int fontThickness = 0.1;

Mat setShotMapMat (int n){
    
    int colorMapMatCols = n;
    Mat mSH(350, colorMapMatCols,CV_8UC3, Scalar::all(0));
    cout << "m rows: "<< mSH.rows<<"\n";
    cout << "m cols: "<< mSH.cols<<"\n";
    mSH.copyTo(shotDistPlot);
    return mSH;
}

void drawShotDistribution( vector <int> shotsList){
    
    size_t shotListLenght = shotsList.size();
    cout << "SH LENGHT: "<< shotListLenght<<"\n";
    
    // MOSTRAR LABELS DE TEXTO: 0, MAX, vector ShotsListLenght

    putText(shotDistPlot,"0", Point(0,10), font, fontSize, Scalar::all(255),fontThickness,0);
    putText(shotDistPlot,"max", Point(shotDistPlot.cols-30,10), font, fontSize, Scalar::all(255),fontThickness,0);
    putText(shotDistPlot,to_string(shotListLenght), Point(0,shotDistPlot.rows-10), font, fontSize, Scalar::all(255),fontThickness,0);
    
    
    
    for (int i = 0; i<shotListLenght; i++) {

        int posY = 20+(i*2);
        Point Pi(0,posY);
        Point Pf(shotsList.at(i),posY);
        
        //Point textP(0,i*2);
        
        int mod = i%255;
        Scalar c(255, 255, 255);
        Scalar c2(255,255,255);

        if (i%2 >0){
            rectangle(shotDistPlot,
                      Pi,
                      Pf,
                      c,CV_FILLED);
        }
        else {
            rectangle(shotDistPlot,
                      Pi,
                      Pf,
                      c2,CV_FILLED);
        }

        //putText(shotDistPlot, to_string(shotsList.at(i)), textP, font, fontSize, Scalar::all(255),fontThickness,0);
    }
    String ShotDistWin = "ShotDist - "+movieName;
    
    imshow(ShotDistWin, shotDistPlot);
    String shotsMapPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/03_Shots/"+movieName+"_dist_map.jpg";
    imwrite(shotsMapPath, shotDistPlot);
    
}

void ShotFramesCount(){
    
    int shNoFrames;
    double fps;
    vector <int> shotFramesVec;
    vector <string> ShotFiles;
    
    Shot_Dir = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/03_Shots/"+movieName+"_list.csv";

    
    cout << "\n" << "TIME METHOD LOADED! \n";

    
    ifstream inputfile(Shot_Dir);
    string current_line;
    
    while(getline(inputfile, current_line)){
        
        stringstream temp(current_line);
        string single_value;
        while(getline(temp,single_value,',')){
    
            ShotFiles.push_back(single_value.c_str());

        }
    }
    
    for (int i = 0; i<ShotFiles.size(); i++){
        movie_Path=ShotFiles.at(i);
        VideoCapture cap(movie_Path);
        shNoFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); // numero total de cuadros
        fps = cap.get(CV_CAP_PROP_FPS); // cuadros por segundo
        
        cout << "EN SHOT #: "<<i<<" ;NO FRAMES: " <<  shNoFrames << "\n";
        shotFramesVec.push_back(shNoFrames);
    }

    
    String csvPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/03_Shots/"+movieName+"_time.csv";
    cout<<"\n Escribiendo en: "<<csvPath<<" \n";
    ofstream tvsFile;
    try{
        tvsFile.open(csvPath, fstream::in | fstream::out );
        if (tvsFile.is_open()){
            cout<<"\n CSV FILE OPEN! \n";
            for (int j=0; j<shotFramesVec.size(); j++){
                tvsFile<< shotFramesVec.at(j) <<"\n";
            }
        }
        else  cout << "No se pudo abrir el CSV:_time.  \n";
    }
    catch (runtime_error& ex){
        cout << "Error al escribir la información en CSV.  \n";
    }
    
    // DIBUJAR LA DISTRIBUCIÓN DE CORTES **************************************************
    
    drawShotDistribution(shotFramesVec);

}



