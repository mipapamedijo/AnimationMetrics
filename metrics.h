//
//  metrics.h
//  OpenCVTest
//
//  Created by Mipapamedijo on 06/02/18.
//  2018 No Budget Animation S de RL de CV.
//

#ifndef METRICS_H
#define METRICS_H
#include <opencv2/core.hpp>

using namespace cv;

extern String movieName;
extern String movieExt;
extern String corpus;
extern String mainLocation;
extern String dataFolder;
extern int totalFrames;

//Declaration

cv::Mat drawColors(cv::Scalar color[], int limite);
cv::Mat setColorMapMat (int m);
cv::Mat setPlotMat ();
cv::Mat setShotMapMat (int n);

void mapColorDistances(cv::Scalar color[]);
void colorMap(cv::Scalar color[], int limite);
void getClusteredColors(int i, int totalFrames);
void detectShot(cv::Mat frameA, cv::Mat frameB, int frameAct, int totalFrames);
void drawChart (std::vector<double> msdA);
void drawHist (int height2, int width2, cv::Mat img2, int frame, int totalFrames);
double histCompare (cv::Mat a, cv::Mat b, int noFrame);
inline void compPHash (cv::Mat a, cv::Mat b, int frame);
void ImageCrossCorrelation (cv::Mat a, cv::Mat b, int frame);
void ShotFramesCount();
void detectShot_m2(cv::Mat frameA, cv::Mat frameB, int frame);
void drawShotDistribution( std::vector<int>);

#endif /* metrics_h */
