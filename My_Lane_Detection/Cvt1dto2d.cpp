# include "Cvt1dto2d.h"
# include <vector>
# include<iostream>
# include <highgui.h>
//# include <cxcore.h>
//# include <cv.h> 
//# include "opencv2/imgproc/imgproc.hpp"

using namespace std;

// vector<vector<int> > src_gray(height, vector<int>(width));

// vector<vector<vector<int> > > dst(height, vector<vector<int> >(width,vector<int>(3)));


void Cvt1dto2d( IplImage *src, int **r, int **g, int **b )
{
  int k = 0;
  if ( src -> nChannels == 1 )
  {
    for ( int i = 0 ; i < src -> height ; i++ )
    {
      if ( ( k + 3 ) % src -> widthStep == 0 ) k += 3;
      for ( int j = 0 ; j < src -> width ; j++, k += 3 )
      {
        b[i][j] = src->imageData[k];
        g[i][j] = src->imageData[k];
        r[i][j] = src->imageData[k];
      } // for
    } // for
  } // if
  else if ( src -> nChannels == 3 )
  {
    for ( int i = 0 ; i < src -> height ; i++ )
    {
      if ( ( k + 3 ) % src -> widthStep == 0 ) k += 3;
      for ( int j = 0 ; j < src -> width ; j++, k += 3 )
      {
        b[i][j] = src->imageData[k];
        g[i][j] = src->imageData[k+1];
        r[i][j] = src->imageData[k+2];
      } // for
    } // for
  } // else if
  else cout << "Number of channel error!!" << endl;

} // Cvt1dto2d()



void Cvt2dto1d( int **r, int **g, int **b, IplImage *dst )
{
  int k = 0;
  if ( dst -> nChannels == 3 )
  {
    for ( int i = 0 ; i < dst -> height ; i++ )
    {
      if ( ( k + 3 ) % dst -> widthStep == 0 ) k += 3;
      for ( int j = 0 ; j < dst -> width ; j++, k += 3 )
      {
        dst->imageData[k] = b[i][j];
        dst->imageData[k+1] = g[i][j];
        dst->imageData[k+2] = r[i][j];
      } // for
    } // for
  } // if
  else cout << "Number of channel error!!" << endl;

} // Cvt2dto1d()