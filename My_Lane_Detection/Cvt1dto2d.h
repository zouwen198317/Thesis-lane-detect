# include <vector>
# include<iostream>
# include <highgui.h>

using namespace std;

#ifndef CVT1DTO2D_H 
#define CVT1DTO2D_H 

void Cvt1dto2d( IplImage *src, int **r, int **g, int **b ) ;
void Cvt2dto1d( int **r, int **g, int **b, IplImage *dst ) ;

#endif