//
//  clustering.cpp
//  Clusters de color (3 canales) a partir de una imagen, usando K-Means Clustering.
//  AnimationMetrics
//
//  Created by Mipapamedijo on 05/02/18.
//  2018 No Budget Animation S de RL de CV.


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/plot.hpp>
#include "metrics.h"

using namespace cv;
using namespace std;

String sColorsPath;
String imgPath;
String csvPath;
String xmlPath;
FileStorage xmlFile;

Mat m, p;
Mat colorMapMat;
Mat plotMat;

Vec3f hsvColor;

vector<Scalar> colors;
vector<double> distanceColor;

void polarTrans(Mat colorMap){
    Mat polarColorMap;
    Point2f center((colorMap.cols/2, colorMap.rows/2),colorMap.rows/2);
    double mVal = colorMap.cols / log(180);
    logPolar(colorMap, polarColorMap, center, mVal, CV_INTER_LINEAR + WARP_FILL_OUTLIERS);
    
    imshow("COLOR POLAR MAP", polarColorMap);
}

Mat setColorMapMat (int n){
    
    int colorMapMatCols = n;
    Mat m(300, colorMapMatCols,CV_8UC3, Scalar::all(255));
    cout << "totalFrames Cluster info: "<< colorMapMatCols <<"\n";
    cout << "m rows: "<< m.rows<<"\n";
    cout << "m cols: "<< m.cols<<"\n";
    m.copyTo(colorMapMat);
    return m;
}


// ********************************************************************************************************

// INGRESAR VALOR ENTERO PARA K (K means clustering)

int clusterCount = 1;

// ********************************************************************************************************

vector <Scalar> allColors;

Vec3f RgbToHsv(Scalar rgb) {
    
    Vec3f hsv(0,0,0);
    
    int r,g,b;
    r=rgb[2];
    g=rgb[1];
    b=rgb[0];
   float min, max, delta;
    
    min = std::min(std::min(b, g), r);
    max = std::max(std::max(b, g), r);
    
    cout << "\n MIN: " << min << " & MAX: " << max <<  "\n";
    
    hsv[2] = (max/255)*100;
    delta = max-min;
    cout << "\n DELTA: " << delta <<  "\n";
    if ( max!=0){
        hsv[1] = (delta / max)*100; //cout << "\n SAT: " << hsv[1] <<  "\n";
    }
    else {
        hsv[1] = 0;
        hsv[0] = -1;
    }
    if (r == max){
        hsv[0] = ((g-b)/delta);
    }
    else if (g == max){
        hsv[0] =  (2 + ( b - r ) / delta);
    }
    else{
        hsv[0] =  (4 + ( r - g ) / delta);
    }
    hsv[0] *= 60;
    
    if (hsv[0] < 0){
        hsv[0] += 360;
    }
    cout << "\n RGB TO HSV: " << hsv << "\n";
    
    String csvPathHSV = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/00_Data_Output/CSV/"+movieName+"_ColorData-HSV.csv";
    
    ofstream tvsFile;
    try{
        tvsFile.open(csvPathHSV, fstream::in | fstream::app);
        if (tvsFile.is_open()){
            //cout<<"\n CSV FILE OPEN! \n";
            //tvsFile<< "NUMERO_DE_CLUSTERS: "<< clusterCount <<"\n";
            //cout <<"SIZE of "<<sizeof(colorArray[0])<<"\n";
            tvsFile<< hsv <<"\n";
        }
        else  cout << "no se pudo abrir el CSV.  \n";
    }
    catch (runtime_error& ex){
        cout << "Error al escribir la información de color en CSV.  \n";
    }
    
    return hsv;
}

Scalar HsvToBgr (Vec3f hsv){
    
    double h = hsv[0];
    double s = hsv[1]/100;
    double v = hsv[2]/100;
    
    double b, g, r;
    double c =0, m =0, x=0;
    
    c = v * s;
    x = c * (1 - fabs(fmod(h / 60,2) - 1));
    m = v - c;
    
    if (h >=0 && h<60){
        r = c+m;
        g = x+m;
        b = m;
    }
    else if (h >=60 && h<120){
        r = x+m;
        g = c+m;
        b = m;
    }
    else if (h >=120 && h<180){
        r = m;
        g = c+m;
        b = x+m;
    }
    else if (h >=180 && h<240){
        r = m;
        g = x+m;
        b = c+m;
    }
    else if (h >=240 && h<300){
        r = x+m;
        g = m;
        b = c+m;
    }
    else if (h >=300 && h<360){
        r = c+m;
        g = m;
        b = x+m;
    }
    else{
        r = m;
        g = m;
        b = m;
    }
    
    Scalar BgrFromHsv(b*255,g*255,r*255);
    //cout << "\n CONVERSION DE HSV: "<< hsv << " A BGR: " << BgrFromHsv << "\n";
    return BgrFromHsv;
}

vector<Point> plotHSVDistances (Vec3f hsv){
    
    int posX, posY;
    //float max;
    double posV, posS;
    vector<Point> center;
    
    posX = hsv[0];
    posS = hsv[1];
    posV = hsv[2];
    
    if (posX < 0){
        posX = 0;
    }
        posV = 100+posV;

    
    cout << "\n POSX: "<< posX <<", POSS: "<< posS <<", POSV: "<< posV <<"\n";
    
    center.push_back(Point(posX, posS));
    center.push_back(Point(posX, posV));
    
    return center;
}

Mat setPlotMat(){
    
    // Dibujar espectro cromática
    
    Mat chromaHSV(200,360,CV_8UC3, Scalar::all(0));
    
    int div = 1;

    for (int i=0; i<360; i++){
        for (int j=0; j<100; j++){
            
            
            Vec3f c(i,j,100);
            Scalar color = HsvToBgr(c);
            
            //cout << "\n COLOR HSV A RGB:" << color << " ! \n";
            rectangle(chromaHSV, Point( div*i, j ), Point((div*i)+div, j+1), color,CV_FILLED,10);
            
                }
            }
    
        for (int i=0; i<360; i++){
            for (int k=0; k<100; k++){
             
        Vec3f c(i,100,k);
        Scalar color = HsvToBgr(c);
        //cout << "\n COLOR HSV A RGB:" << color << " ! \n";
            rectangle(chromaHSV, Point( div*i, 100+k ), Point((div*i)+div, (100+k)+1), color,CV_FILLED,10);
            }
        }
    
    chromaHSV.copyTo(p);
    return p;
}

void updatePlot(){
    
    Point centerS = plotHSVDistances(hsvColor).at(0);
    Point centerV = plotHSVDistances(hsvColor).at(1);
    //cout <<"\n"<< "CENTER: "<< center <<"\n";
    int radius = 2;
    
    circle(p, centerS, radius, Scalar::all(0),1,8);
    circle(p, centerV, radius, Scalar::all(255),1,8);
    
    imshow("POSICION EN HSV DE LOS CLUSTERS", p);
    
    //polarTrans(p);
    
    String plotPath ="/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/04_ColorMaps/"+movieName+"_plot.jpg";
    
    imwrite(plotPath, p);
}

Mat drawColors(Scalar color[], int limite) {
    int x = 300;
    int y = 20;
    int i=0;
    Mat array = Mat::ones(y,x,CV_8UC3);
    int div = (x/limite);

    for (i=0; i<limite; i++){
    cout <<"Color recibido: "<< color[i] << ". \n";
    //cout <<"Color en arreglo: "<< array << ". \n";
    cout <<"Dibujando color número "<< i+1 << " de "<<limite<<" en "<<div*i<<" . \n";
    rectangle(array, Point( div*i, 0 ), Point((div*i)+div, 20), color[i],CV_FILLED,10);
    }
 /*   try{
    imwrite(sImagePath, array);
    }
    catch (runtime_error& ex){
        cout << "Error al escribir la información de color.  \n";
        return array;
    }
  */
    return array;
}

void colorMap(Scalar color[], int limite){
    
    //double div = (colorMapMat.cols/limite);
    
    cout << "\n COLOR MAP MAT SIZE: " << colorMapMat.cols << " X " << colorMapMat.rows << "\n";
    cout << "\n LIMITE DE MAPA: " << limite << "\n";
    
    allColors.push_back(color[0]);
    int i = allColors.size();
    
    cout << "\n VECTOR ALL COLORS: " << i << "\n";

        for (int j = 0; j<allColors.size(); j++){
            rectangle(colorMapMat, Point( (i), 0 ), Point( ((i)+1), colorMapMat.rows), allColors.at(j),CV_FILLED);
        }
    
    int colorMapSizeW = 1920;
    double scale = limite/colorMapSizeW;
    
    Mat colorMapResult;
    Size r(colorMapSizeW, colorMapMat.rows);
    cout << "\n R: " << r << "\n";
    resize(colorMapMat,colorMapResult,r);
    string wName = "MAPA DE COLOR";
    namedWindow(wName, CV_WINDOW_AUTOSIZE);
    imshow(wName, colorMapResult);
    
    String colorMapSavePath ="/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/04_ColorMaps/"+movieName+".jpg";
    
    imwrite(colorMapSavePath, colorMapResult);
    
}

void mapColorDistances(Scalar color[]){
    
    // Medir las distancias entre los clusters obtenidos para saber qué tipo de paleta de color teórica se presenta en el video.
    
    colors.push_back(color[0]);
    
    double distance;
    double normDistance;
    double colorAverage;
    Scalar colorSTDev;
    
    int b1,b2,g1,g2,r1,r2;
    
    // INFORMACIÓN DE COLOR EN ESPECTRO HSV (HUE (0-360), SATURATION (0-100), VALUE (0-100)
    
    hsvColor = RgbToHsv(color[0]) ;
    updatePlot();
    
    // ––––––––––––––––––––––––––––––––––
    
    //cv::plot::Plot2d::create(hsvColor[0], 100);
    
    cout << "\n COLOR EN ESPACIO HSV " << hsvColor << "\n";
    cout << "\n ****************************** \n HUE: " << hsvColor[0] << " ******************************  \n";
    
    
    int colorVectorSize = (int)colors.size();
    
    if (colorVectorSize>2){
        
        b1 = colors.at(colorVectorSize-2)[0];
        cout << "\n Valor <B> en i-1 : " << b1 << "\n";
        g1 = colors.at(colorVectorSize-2)[1];
        r1 = colors.at(colorVectorSize-2)[2];
        
        b2 = colors.at(colorVectorSize-1)[0];
        cout << "\n Valor <B> en i : " << b2 << "\n";
        g2 = colors.at(colorVectorSize-1)[1];
        r2 = colors.at(colorVectorSize-1)[2];
        
        
        double sqrDistance = abs((b2-b1)*(b2-b1))+
                             abs((g2-g1)*(g2-g1))+
                             abs((r2-r1)*(r2-r1));
        
         cout << "\n Valor <B^2> en i - (i-1) : " << abs((b2-b1)*(b2-b1)) << "\n";
        
         cout << "\n DISTANCIA DE COLOR CUADRADA EN i - (i-1) <B,G,R> : " << sqrDistance << "\n";
        
        distance = sqrt(sqrDistance);
        normDistance = norm(distance);
        
        cout << "\n DISTANCIA DE COLOR RESULTANTE EN i - (i-1) <B,G,R> : " << distance << "\n";
        
        distanceColor.push_back(distance);
        
        // Recorrer el arreglo de distancias para sacar el promedio aritmético
        
        double colorSum=0;
        double eightBitSqrt = sqrt((255*255)+(255*255)+(255*255));
        
        for (int i = 0; i < distanceColor.size(); i++){
            colorSum += distanceColor.at(i);
            colorAverage = colorSum / eightBitSqrt;
            //meanStdDev(colorSum, colorAverage, colorSTDev);
        }
        
        cout << "\n SUMATORIA DE DISTANCIAS ENTRE COLORES RESULTANTES ColorSum: " << colorSum << "\n";
        cout << "\n PROMEDIO ARITMÉTICO DE COLORES RESULTANTES ColorAverage: " << colorAverage << "\n";
        
        Mat colorMean(50, 50,CV_8UC3, colorSTDev);
        //imshow("COLOR MEAN", colorMean);
        
    }
}

void getClusteredColors(int i, int totalFrames){
    
    imgPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/02_ImageFrames/"+movieName+"/frame_"+to_string(i)+".jpg";
    sColorsPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/00_Data_Output/02_ColorPaletes/"+movieName+"/Color_"+to_string(i)+".jpg";
    
    
    Mat img = imread(imgPath);
    
    int width = img.cols;
    int height = img.rows;
    
    cout << "\n cW: " << width << " X  cH: " << height << "\n";
    
    if (img.empty()){
        cout << "No se pudo abrir la imagen. \n";
    }
    Mat original;
    img.convertTo(original, CV_8UC3);
    img.convertTo(img, CV_32F);
    Mat bluredImg;
    blur(img,bluredImg,Size(1,1));
    Mat samples = bluredImg.reshape(1, img.total());

    
    int attempts = 5 ;
    Mat labels, data, centers, clustered; 
    
    kmeans(samples, clusterCount, labels,TermCriteria(CV_TERMCRIT_EPS, 10 , 0.5), attempts, KMEANS_RANDOM_CENTERS, centers);
    
    centers = centers.reshape(3,0);
    labels = labels.reshape(1,img.rows);
    
    Mat result(img.size(), CV_8UC3);
    Scalar color;
    Scalar colorArray[clusterCount];
    Mat rectangulos;

    for (int i=0; i<centers.rows; i++){
        color = centers.at<Vec3f>(i);

        colorArray[i] = centers.at<Vec3f>(i);
        cout << "Valor de color en "<<(i)<<" : "<< color <<" \n";
        cout << "Color guardado: "<<colorArray[i]<<" en "<<i<<"\n";
        Mat mask(labels == i);
        result.setTo(color,mask);
        int colorG = colorArray[i][0];
        int colorB = colorArray[i][1];
        int colorR = colorArray[i][2];
        String colorRGB = to_string(colorR) + "," + to_string(colorG) + "," + to_string(colorB);
        cout << "\n" << colorRGB << "\n";
        
    xmlPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/00_Data_Output/XML/"+movieName+"_ColorData.xml";
        
        try{
            xmlFile.open(xmlPath, FileStorage::APPEND);
            if (xmlFile.isOpened()){
            xmlFile << "Color_RGB" <<  colorRGB;
            xmlFile.release();
            }
            else{
            cout << "no se pudo abrir el XML.  \n";
            }
        }
        catch (runtime_error& ex){
            cout << "Error al escribir la información de color en XML.  \n";
        }
    }
    
    csvPath = "/Users/mipapamedijo/PROYECTOS/PROGRAMAS/AnimationMetrics/00_Data_Output/CSV/"+movieName+"_ColorData.csv";
    
    ofstream tvsFile;
    try{
        tvsFile.open(csvPath, fstream::in | fstream::app);
        if (tvsFile.is_open()){
            //cout<<"\n CSV FILE OPEN! \n";
            //tvsFile<< "NUMERO_DE_CLUSTERS: "<< clusterCount <<"\n";
            //cout <<"SIZE of "<<sizeof(colorArray[0])<<"\n";
            tvsFile<< colorArray[0] <<"\n";
        }
        else  cout << "no se pudo abrir el CSV.  \n";
    }
    catch (runtime_error& ex){
         cout << "Error al escribir la información de color en CSV.  \n";
    }
    

    rectangulos = drawColors(colorArray, clusterCount);
    colorMap(colorArray, totalFrames);
    mapColorDistances(colorArray);
    
    int barSize = 25;
    
    clustered.convertTo(clustered, CV_8UC3);
    imshow("Frame original (reformateado)", original);
    moveWindow("Frame original (reformateado)", 0, 0);
    imshow("Clusters (blur)", result);
    moveWindow("Clusters (blur)",0, height+(barSize*2));
    imshow("Colores", rectangulos);
    moveWindow("Colores", 0, ((height*2)+barSize*3));
    
    try{
        imwrite(sColorsPath, rectangulos);
    }
    catch (runtime_error& ex){
        cout << "Error al escribir la información de color.  \n";
    }
    
    //waitKey(0);

}

//END OF PROCESS //

