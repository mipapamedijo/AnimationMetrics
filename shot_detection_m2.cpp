//
//  shot_detection_m2.cpp
//  OpenCVTest
//
//  Created by Mipapamedijo on 05/09/18.
//  2018 No Budget Animation S de RL de CV.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/core/ocl.hpp>
#include "metrics.h"

using namespace cv;
using namespace std;

vector<double> sum_m2;
double umbralCambio = 10;

string sFilePath_m2 = "/Users/mipapamedijo/Projects/PROGRAMAS/AnimationMetrics/03_Shots/"+movieName+"/";
string shotsInfoPath_m2;
string sxmlPath_m2 = "/Users/mipapamedijo/Projects/PROGRAMAS/AnimationMetrics/00_Data_Output/"+movieName+"/Shots.xml";

string winShot_m2 = "Shot Detection";

void detectShot_m2(Mat frameA, Mat frameB, int actFrame){
    
    shotsInfoPath_m2 = "/Users/mipapamedijo/Projects/PROGRAMAS/AnimationMetrics/03_Shots/frame_"+to_string(actFrame)+".jpg";
    
    if (actFrame<=3){
         imwrite(shotsInfoPath_m2, frameB);
    }
    
    Mat differenceMat, minMat, frameGrayA, frameGrayB;
    vector<Mat> bgr;
    Mat empty;
    empty = Mat::zeros(frameA.rows, frameA.cols, CV_8UC1);
    Mat resultBlue(frameA.rows, frameA.cols, CV_8UC3);
    Mat resultGreen(frameA.rows, frameA.cols, CV_8UC3);
    Mat resultRed(frameA.rows, frameA.cols, CV_8UC3);
    
    cvtColor(frameA, frameGrayA, CV_BGR2GRAY);
    cvtColor(frameB, frameGrayB, CV_BGR2GRAY);
    int width, height;
    width = frameGrayA.cols;
    height = frameGrayA.rows;
    

    //RESTAR SOLO IMAGENES EN GRISES
    subtract(frameGrayA, frameGrayB, differenceMat);
    
    //RESTAR IMAGENES POR CANAL [3]
    Mat bDiff, gDiff, rDiff;
    Mat subMat;
    
    subtract(frameA, frameB, subMat);
    
    split(subMat, bgr);
    
    
    Mat in1[] = {bgr[0], empty, empty};
    int from_to1[] = {0,0,1,1,2,2};
    mixChannels(in1, 3, &resultBlue, 1, from_to1, 3);
    
    Mat in2[] = {empty, bgr[1], empty};
    int from_to2[] = {0,0,1,1,2,2};
    mixChannels(in2, 3, &resultGreen, 1, from_to2, 3);
    
    Mat in3[] = {empty, empty, bgr[2]};
    int from_to3[] = {0,0,1,1,2,2};
    mixChannels(in3, 3, &resultRed, 1, from_to3, 3);
    
    imshow("BLUE",resultBlue);
    imshow("GREEN",resultGreen);
    imshow("RED",resultRed);
    
    bDiff = bgr[0];
    gDiff = bgr[1];
    rDiff = bgr[2];

    double bDiff_double = (double)countNonZero(bDiff)/bDiff.total()*100;
    cout << "\n bDiff : "<< bDiff_double <<" EN : "<< actFrame << " \n";
    double gDiff_double = (double)countNonZero(gDiff)/gDiff.total()*100;
    cout << "\n gDiff : "<< bDiff_double <<" EN : "<< actFrame << " \n";
    double rDiff_double = (double)countNonZero(rDiff)/rDiff.total()*100;
    cout << "\n rDiff : "<< bDiff_double <<" EN : "<< actFrame << " \n";
    
    //differenceMat = frameB - frameA;
    //absdiff(frameB, frameA, differenceMat);
    //min(frameB, frameA, minMat);
    
    Mat compareFrames_m2(width*2, height, CV_8UC3);
    Mat win_m2(width*2, height*2, CV_8UC3);
    Size r(width, height/2);
    hconcat(frameA, frameB, compareFrames_m2);
    hconcat(compareFrames_m2, subMat, win_m2);
    resize(win_m2, win_m2, r);
    imshow("Diferencia M2:", win_m2);
    //imshow("A_M2", frameA);
    //imshow("B_M2", frameB);
    
    //threshold(differenceMat, differenceMat, 10, 255, CV_THRESH_BINARY);
    double d_gray, d_color = 0;
    double sumNoDiferencias = 0;

    // USANDO UN SOLO CANAL GRIS:
    d_gray = (double)countNonZero(differenceMat)/differenceMat.total()*100;
    //sum_m2.push_back(d_gray);
    
    // USANDO SUMATORIA DE 3 CANALES:
    d_color = bDiff_double + gDiff_double + rDiff_double;
    cout << "\n  d_color [3] : "<< d_color <<" EN : "<< actFrame << " \n";
    sum_m2.push_back(d_color);
    
    size_t sumSize = sum_m2.size();
    cout << "\n  LENGHT : "<< sumSize <<" EN : "<< actFrame << " \n";
    
    if (sumSize>2){
        
        sumNoDiferencias = 0;
        sumNoDiferencias = abs(sum_m2.at(sumSize-1) - sum_m2.at(sumSize-2));
        cout << "\n  ENTRANDO EN IF : "<< actFrame <<" VALOR!!!!!!................. SUM  : "<< sumNoDiferencias << " \n";
    }
    
    if (sumNoDiferencias == 0 ){
         cout << "\n # NO HAY DIFERENCIAS USANDO M2, "<< sumNoDiferencias << " FRAMES IGUALES \n";
    }
    else{
        cout << "\n # EL NUMERO DE DIFERENCIAS USANDO M2, ES: "<< sumNoDiferencias << "\n";
        if (sumNoDiferencias > umbralCambio){
            
            imwrite(shotsInfoPath_m2, frameB);
            
            cout << "\n *************************************************** \n CORTE EN: "<< actFrame << "\n ******************************************* POR "<< sumNoDiferencias <<"\n";
        }
    }
}

