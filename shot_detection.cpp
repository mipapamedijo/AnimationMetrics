//
//  shot_detection.cpp
//  Encontrar la variación de cambio de shot entre cuadros.
//  C++ , Open CV
//  2018 No Budget Animation S de RL de CV.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/core/ocl.hpp>
#include "metrics.h"


using namespace cv;
using namespace cv::img_hash;
using namespace std;

template <typename T>

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
int threshold_type =  1;

//////- > CUT TRANSITION PARAMETERS
int const max_value = 100;
int const max_type = 4;
int threshold_value = 23;
int threshold_value_msd = 23;
int const max_value_msd = 60;
string trackbar_value = "Trackbar / Shot";
string trackbar_value_msd = "Trackbar / MSD";

int const max_BINARY_value = 255;
bool DebugMode = false;

double** imd;
vector<double> msdVector;
vector<double> rightGradientVector;
vector <double> averageVector;
vector <double> binVector;
double diffVal;
double tmpMsd;
double gradientDifference;
double MSE;
int height;
int width;

Mat frameGrayA,frameGrayB;
Mat histTotal(400, 900, CV_8UC3, Scalar::all(255));

string sFilePath = "/Users/mipapamedijo/PROJECTS_local/PROGRAMAS/AnimationMetrics/03_Shots/"+movieName+"/";
string shotsInfoPath;
string sxmlPath = "/Users/mipapamedijo/PROJECTS_local/PROGRAMAS/AnimationMetrics/00_Data_Output/"+movieName+"/Shots.xml";

FileStorage sxmlFile;
string winShot = "Shot Detection";

void detectShot(Mat frameA, Mat frameB, int frameAct, int totalFrames){

    height = frameA.rows;
    width = frameA.cols;
    
    namedWindow(winShot, CV_WINDOW_AUTOSIZE);
    createTrackbar(trackbar_value, winShot, &threshold_value, max_value);
    createTrackbar(trackbar_value_msd, winShot, &threshold_value_msd, max_value_msd);
    
    imd = new double*[height];
    for (int i=0; i<height; i++){
        imd[i] = new double[width];
    }
    
    //bool startTracking = false;
    
    cvtColor(frameA, frameGrayA, CV_BGR2GRAY);
    cvtColor(frameB, frameGrayB, CV_BGR2GRAY);
    
    
    // **********************************************************************
    //Método 1. Cálculo de la desviación cuadrática media (MSD) entre A y B
    
    diffVal = 0;
    for (size_t i = 0; i < height; i++){
        for (size_t j = 0; j < width; j++){
            
            diffVal = (int)frameGrayA.at<uchar>(i , j ) - (int)frameGrayB.at<uchar>(i, j);
            diffVal = diffVal*diffVal;
            imd[i][j] = diffVal;
        }
    }
    double Tot = 0;
    for (size_t i = 0; i < height; i++){
        for (size_t j = 0; j < width; j++){
            Tot = Tot + imd[i][j];
        }
    }
    
    double MSD_m1 = sqrt(Tot / frameA.total());
    
    cout << "\n # Método 1 - MSD : "<< MSD_m1 << " en el frame:" << frameAct << "\n";
    
    tmpMsd += MSD_m1;
    msdVector.push_back(MSD_m1);
    
    //  FIN DE MÉTODO 1
    
    // **********************************************************************
    // Método 2. Cálculo del error absoluto diferencial (Desviación cuadrática - MSE) y la media (MSD) entre A y B (absdiff)
    
    Mat dif;
    absdiff(frameGrayA, frameGrayB, dif);
    dif.convertTo(dif, CV_32F);
    dif = dif.mul(dif);
    Scalar s = sum(dif);
    
    double sse = s.val[0] + s.val[1] + s.val[2];
    
    if (sse <= 1e-05){
        MSE = 0;
    }
    else{
        MSE = sse / (double)(frameGrayA.channels() * frameGrayA.total());
    }
    
    cout << "\n # Método 2 - MSE : "<< MSE << " en el frame:" << frameAct << "\n";
     //msdVector.push_back(MSE);
    
     //  FIN DE MÉTODO 2
    
    // **********************************************************************
    // Método 3. Cálculo del la desviación cuadrática media por método de normalización (norm) de las imágenes A y B
    
    double MSD_m2 = norm(frameGrayA, frameGrayB);
    MSD_m2 = sqrt(MSD_m2);
    
    cout << "\n # Método 3 - MSD : "<< MSD_m2 << " en el frame:" << frameAct << "\n";
    
    //msdVector.push_back(MSD_m2);
    
    //  FIN DE MÉTODO 3
    
    // **********************************************************************
    // Método 4. Cálculo de los promedios (mean) entre las imágenes A y B
    
    Scalar meanA = mean(frameGrayA);
    Scalar meanB = mean(frameGrayB);
    double meanA_value = (double)meanA[0];
    double meanB_value = (double)meanB[0];
    
    double meanDiff = abs(meanB_value - meanA_value);
    
    cout << "\n # Método 4 - MEAN : "<< meanDiff << " en el frame:" << frameAct << "\n";
    
    //msdVector.push_back(meanDiff);
    
    // FIN DE MÉTODO 4
    
    // Cálculo de material de imagen en threshold binario
    
    Mat diff;
    absdiff(frameGrayA, frameGrayB, diff);
    threshold(diff, diff, 10, 255, CV_THRESH_BINARY);
    double d_color = 0;
    d_color += (double)countNonZero(diff)/frameA.total();
    //msdVector.push_back(d_color);
    cout << "\n En: "<< frameAct << " d_color : "<< d_color << " \n";
    
    // **********************************************************************
    // Método 5. Cálculo del Hash de A vs B
    
    ocl::setUseOpenCL(false);
    compPHash(frameGrayA, frameGrayB, frameAct);
    
    // FIN DE MÉTODO 5
    
    // **********************************************************************
    // Método 6. Cálculo de cross-correlation de A y B
    
    // ImageCrossCorrelation(frameGrayA, frameGrayB, frameAct);
    
    // FIN DE MÉTODO 6
    
    // Llamar métodos de representación (drawChart ; drawHist; histCompare)
    
    drawChart(msdVector);
    drawHist(height, width, frameA, frameAct, totalFrames);
    //histCompare<PHash>(frameA, frameB, frameAct);
    
    size_t sizeAr = msdVector.size();

    if (sizeAr > 3){
        
        double rightGradient = abs(msdVector.at(sizeAr - 3) - msdVector.at(sizeAr - 2));
        double leftGradient = abs(msdVector.at(sizeAr - 2) - msdVector.at(sizeAr - 1));
        
        rightGradientVector.push_back(rightGradient);
        int size = rightGradientVector.size();
        
        /*if (size > 4) {
            int rGV_3 = (int)(rightGradientVector.at(size - 3));
            int rGV_2 = (int)(rightGradientVector.at(size - 2));
            int rGV_1 = (int)(rightGradientVector.at(size - 1));
            
            //cout << "rGV_3: " << rGV_3 << "\n";
            //cout << "rGV_2: " << rGV_2 << "\n";
            //cout << "rGV_1: " << rGV_1 << "\n";
            
            int deltaL = abs(rGV_3 - rGV_2);
            int deltaR = abs(rGV_2 - rGV_1);
            //cout << "(int)rightGradientVector.at(size - 3) :" << (int)rightGradientVector.at(size - 3) << "\n";
            //cout << "(int)rightGradientVector.at(size - 2) :" << (int)rightGradientVector.at(size - 2) << "\n";
            //cout << "(int)rightGradientVector.at(size - 1) :" << (int)rightGradientVector.at(size - 1) << "\n";
            
            if ((rGV_3 < rGV_2)&&(rGV_2 < rGV_1)){
                cout << "\n …………………………………………………………………………………………………………………………………………PASS !\n";
                cout << "DeltaL:" << deltaL << "\n";
                cout << "DeltaR:" << deltaR << "\n";
                cout << "RightGradient:" << rightGradient << "\n";
                cout << "LeftGradient:" << leftGradient << "\n";
                
                
                if((rightGradient > threshold_value) && (leftGradient > threshold_value)){
                    cout << " \n ************************** \n";
                    cout << "G > TH - CORTE en " << frameAct << "\n";
                    cout << "************************** \n";
                    shotsInfoPath = "/Users/mipapamedijo/Projects/PROGRAMAS/AnimationMetrics/03_Shots/frame_"+to_string(frameAct)+".jpg";
                    imwrite(shotsInfoPath, frameB);
                    try{
                        sxmlFile.open(sxmlPath, FileStorage::APPEND);
                        if (sxmlFile.isOpened()){
                            sxmlFile << "Shot" <<  frameAct;
                            sxmlFile.release();
                        }
                        else{
                            cout << "No se pudo abrir el XML.  \n";
                        }
                    }
                    catch(runtime_error& ex){
                        cout << "Error al escribir la información de color en XML.  \n";
                    }
                }
            }
        }*/
        
        gradientDifference = rightGradient - leftGradient;
        cout << "Gradient Difference : "<< gradientDifference << "\n";
        cout << "\n En : "<< frameAct <<" R  Gradient  : "<< rightGradient << " -------- " << "L  Gradient  : "<< leftGradient << "\n";
        
        
       if ((rightGradient > threshold_value) && ( leftGradient  > threshold_value)){
            cout << "************************** \n";
            cout << "G > TH - CORTE en " << frameAct << "\n";
           shotsInfoPath = "/Users/mipapamedijo/PROJECTS_local/PROGRAMAS/AnimationMetrics/03_Shots/frame_"+to_string(frameAct)+".jpg";
           imwrite(shotsInfoPath, frameB);
            cout << "************************** \n";
            
        }
            else if (( rightGradient > threshold_value ) && ( leftGradient > ( threshold_value/4) )) {
                cout << "************************** \n";
                cout << "RG > TH && LG > TH/4 - CORTE en " << frameAct << "\n";
                shotsInfoPath = "/Users/mipapamedijo/PROJECTS_local/PROGRAMAS/AnimationMetrics/03_Shots/frame_"+to_string(frameAct)+".jpg";
                imwrite(shotsInfoPath, frameB);
                cout << "************************** \n";
            }
      }

    Mat compareFrames(width*2, height, CV_8UC3);
    Size r(width/2, height/4);
    hconcat(frameGrayA, frameGrayB, compareFrames);
    resize(compareFrames, compareFrames, r);
    
    imshow(winShot, compareFrames);
    moveWindow(winShot, width/4, 410);
    //imshow(winShot, frameGrayB);
    //imshow(winShot, frameGrayA);
  }

void drawChart (vector<double> msdA){
    int width = msdA.size();
    int height = 300;
    Mat hist(height, width, CV_8UC3, Scalar(255,255,255));
    Mat histMSDr;
    Size r(500,300);
    int scaleFactor = 5;
    int font = FONT_HERSHEY_COMPLEX_SMALL;
    double fontSize = 0.5;
    int fontThickness = 1;
    
    for (int i = 0; i < width; i++ ){
        line(hist, Point(i,height), Point(i,height - scaleFactor*msdA.at(i)), Scalar::all(0), 2, 8, 0);
    }
    Point textOrg(1, height/1.2);
    Point textFin(width-30,height/1.2);
    putText(hist, "2", textOrg, font, fontSize, Scalar::all(255),fontThickness,8);
    putText(hist, to_string(width*2), textFin, font, fontSize, Scalar::all(255),fontThickness,8);
    
    resize(hist, histMSDr, r);
    
    string wName = "Diagrama de valores de MSD";
    namedWindow(wName, CV_WINDOW_AUTOSIZE);
    moveWindow(wName, 830, 0);
    imshow(wName, histMSDr);
}

void drawHist (int height2, int width2, Mat img2, int frame, int totalFrames){
    
    int maxPixel = 256;
    int hBin[maxPixel];
    int binH[maxPixel];
    double average;
    double prev = 0;
    double curr = 0;
    
    double prob[maxPixel];
    double cProb[maxPixel];
    int nA[maxPixel];
    int sumBins;
    
    for (int n=0; n< maxPixel; n++){
        hBin[n] = 0;
        binH[n] = 0;
        prob[n] = 0;
        cProb[n] = 0;
        nA[n] = 0;
    }
    
    int m = 0;
    average = 0;
    
    for (int i=0; i<height2; i++){
        for (int j = 0; j<width2; j++){
            m = (int)img2.at<uchar>(i,j);
            hBin[m] = hBin[m] + 1;
            average = (average + hBin[m]);
            average = average*average;
            average = sqrt(average/img2.total());
            sumBins = sumBins + hBin[m];
            //average = mean(img2)[0];

        }
        
    }
    sumBins = sumBins / img2.total();
    averageVector.push_back(average);
    binVector.push_back(sumBins);

    if (binVector.size()>4) {
        prev = binVector.at(binVector.size()-2);
        curr = binVector.at(binVector.size()-1);
    }
        double diff = abs(curr-prev);

    cout << "\n # Promedio de histograma : "<< average << " en el frame:" << frame << "\n";
    cout << "\n # Suma de de histograma : "<< sumBins << " en el frame:" << frame << "\n";
    cout << "\n Tamaño del Vector (average): "<< averageVector.size() << "\n";
    cout << "\n Variación: "<< diff << "\n";
    //cout << "\n # Promedio de histograma <Vector> : "<< averageVector.at((frame/2)-1) << " en el frame:" << frame << "\n";
    
    double max = 0.00;
        
        for (int h=0; h<maxPixel; h++) {
            float binVal = hBin[h];
            if (max < binVal){
                max = binVal;
            }
        }
        
        int scaleFactor = 400;
        for (int h = 0; h <maxPixel; h++){
            binH[h] = cvRound((double)hBin[h] / max * scaleFactor);
        }
        
        int hist_w = 512;
        int hist_h = scaleFactor;
        int bin_w = cvRound((double)hist_w / maxPixel);
        
        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar::all(0));
        
        for (int i =0; i<maxPixel; i++){
            line(histImage, Point(bin_w*(i-1), hist_h),
                 Point(bin_w*(i-1), hist_h - binH[i-1]),
                 Scalar::all(250), 2,8,0 );
        }
    

        int scaleFactorTotal = 10;
    
    if (averageVector.size()>3){
        line(histTotal,
             Point(frame, hist_h),
             Point(frame, hist_h - scaleFactorTotal*averageVector.at(averageVector.size()-1)),
             Scalar::all(128), 2,8,0);
    }
    
        
        
        String wName = "Histograma";
        namedWindow(wName, CV_WINDOW_AUTOSIZE);
        imshow(wName, histImage);
        moveWindow(wName, width/4, 0);
    
        
        String histTotalWin = "Promedio lineal de histogramas";
        //namedWindow(histTotalWin, CV_WINDOW_AUTOSIZE);
        //imshow(histTotalWin, histTotal);
        
    }

double histCompare (Mat a, Mat b, int noFrame){
    
    cvtColor(a,a, CV_BGR2RGB);
    cvtColor(b,b, CV_BGR2RGB);
    
    double result;
    double diffA_B = 0;
    Mat comparisson;
    Mat aHSV;
    Mat bHSV;
    Mat HSV_half_down;
    cvtColor(a, aHSV, COLOR_RGB2HSV);
    cvtColor(b, bHSV, COLOR_RGB2HSV);
    HSV_half_down = aHSV( Range(aHSV.rows/2, aHSV.rows-1), Range(0, aHSV.cols-1) );
    int h_bins = 50;
    int s_bins = 60;
    int hSize[] = {h_bins, s_bins};
    int histSize = 256;
    
    float h_ranges[] = {0, 256};
    float s_ranges[] = {0, 256};
    
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = { 0, 1 };
    
    MatND hist_half_down;
    MatND hist_A;
    MatND hist_B;
    
    calcHist(&HSV_half_down, 1, channels, Mat(), hist_half_down, 2, hSize, ranges, true, false);
    
    calcHist(&aHSV, 1, channels, Mat(), hist_A, 2, hSize, ranges, true, false);
    
    calcHist(&bHSV, 1, channels, Mat(), hist_B, 2, hSize, ranges, true, false);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImageA( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );
    Mat histImageB( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );
    
    normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize(hist_A, hist_A, 0, histImageA.rows, NORM_MINMAX, -1, Mat() );
    normalize(hist_B, hist_B, 0, histImageB.rows, NORM_MINMAX, -1, Mat() );
    
    Mat calcHist(width*2, height, CV_8UC1);
    Size r2(width/2, height/2);
    hconcat(hist_A,hist_B, calcHist);
    resize(calcHist, calcHist, r2);
    namedWindow("HIST_CALC",WINDOW_AUTOSIZE);
    imshow("HIST_CALC", calcHist);
    
    
    //double diffA_half_down = compareHist(hist_half_down, hist_A, compareMethod);
    diffA_B = compareHist(hist_A, hist_B, CV_COMP_CHISQR );
    cout << "\n DIFF HIST DE A A B: " << diffA_B << " EN : " << noFrame <<"\n";
    //cout << "\n DIFF HIST DE A A HALF_DOWN: " << diffA_half_down << "\n";

    for (int i=1; i < histSize; i++){
        line(histImageA, Point(bin_w*(i-1), hist_h - cvRound(hist_A.at<float>(i-1)) ),
                        Point(bin_w*(i), hist_h - cvRound(hist_A.at<float>(i)) ),
                        Scalar::all(255), 2, 8 , 0 );
        line(histImageB, Point(bin_w*(i-1), hist_h - cvRound(hist_B.at<float>(i-1)) ),
             Point(bin_w*(i), hist_h - cvRound(hist_B.at<float>(i)) ),
             Scalar::all(255), 2, 8 , 0 );
    }
    
    
    
    Mat compareHist(width*2, height, CV_8UC3);
    Size r(width/2, height/2);
    hconcat(histImageA,histImageB, compareHist);
    resize(compareHist, compareHist, r);
    namedWindow("HIST_COMP",WINDOW_AUTOSIZE);
    imshow("HIST_COMP", compareHist);
    
    result = diffA_B;
    return result;
}

void compPHash (Mat a, Mat b, int frame){

    TickMeter tick;
    Mat aHash;
    Mat bHash;
    Size r(a.cols/4, a.rows/4);
    Ptr<ImgHashBase> func;
    func = img_hash::PHash::create();
    
    resize(a, a, r);
    resize(b, b, r);
    
    //imshow("For Phash A", a);
    
    tick.reset(); tick.start();
    func->compute(a, aHash);
    tick.stop();
    
     cout << "\n ! HASH_A: " << tick.getTimeMilli() << "ms \n";
    
    tick.reset(); tick.start();
    func->compute(b, bHash);
    tick.stop();
    
    cout << "\n ! HASH_B: " << tick.getTimeMilli() << "ms \n";
    
    double comparisson = func->compare(aHash,bHash);
    
    cout << "\n ! COMPARACIÓN DE HASH PERCEPTUAL: " << comparisson << " EN : " << frame <<"\n";

}

void ImageCrossCorrelation (Mat a, Mat b, int frame){
    
    Mat a_float, b_float;
    
    a.convertTo(a_float, CV_32FC1);
    b.convertTo(b_float, CV_32FC1);
    
    Mat aComplex[2] = {a_float, Mat::zeros(a_float.size(), CV_32F)};
    Mat bComplex[2] = {b_float, Mat::zeros(b_float.size(), CV_32F)};
    
    Mat aDFT, bDFT;
    merge(aComplex, 2, aDFT);
    merge(bComplex, 2, bDFT);
    
    Mat a_result, b_result;
    
    dft(aDFT, a_result, DFT_COMPLEX_OUTPUT);
    dft(aDFT, b_result, DFT_COMPLEX_OUTPUT);
    
    Mat splitArray_a[2] = {Mat::zeros(a_result.size(), CV_32F), Mat::zeros(a_result.size(), CV_32F)};
    Mat splitArray_b[2] = {Mat::zeros(b_result.size(), CV_32F), Mat::zeros(b_result.size(), CV_32F)};
    
    split(a_result, splitArray_a);
    split(b_result, splitArray_b);
    
    Mat dftMagnitude_a, dftMagnitude_b;
    
    magnitude(splitArray_a[0], splitArray_a[1], dftMagnitude_a);
    magnitude(splitArray_b[0], splitArray_b[1], dftMagnitude_b);
    
    dftMagnitude_a += Scalar::all(1);
    dftMagnitude_b += Scalar::all(1);
    
    log(dftMagnitude_a, dftMagnitude_a);
    log(dftMagnitude_b, dftMagnitude_b);
    
    normalize(dftMagnitude_a, dftMagnitude_a, 0, 1, CV_MINMAX);
    normalize(dftMagnitude_b, dftMagnitude_b, 0, 1, CV_MINMAX);
    
    int centerX = dftMagnitude_a.cols/2;
    int centerY = dftMagnitude_a.rows/2;
    
    Mat q1(dftMagnitude_a, Rect(0,0, centerX, centerY));
    Mat q2(dftMagnitude_a, Rect(centerX,0, centerX, centerY));
    Mat q3(dftMagnitude_a, Rect(0,centerY, centerX, centerY));
    Mat q4(dftMagnitude_a, Rect(centerX,centerY, centerX, centerY));
    
    Mat swapMap;
    
    q1.copyTo(swapMap);
    q4.copyTo(q1);
    swapMap.copyTo(q4);
    
    q2.copyTo(swapMap);
    q3.copyTo(q2);
    swapMap.copyTo(q3);
    
    Size r(dftMagnitude_a.cols/4, dftMagnitude_a.rows/4);
    
    resize(dftMagnitude_a, dftMagnitude_a, r);
    
    //INVERTIR DFT
    
    Mat inverse_a;
    
    dft(dftMagnitude_a, inverse_a, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    
    imshow("Cross Correlation DFT", dftMagnitude_a);
    
}

// FIN
