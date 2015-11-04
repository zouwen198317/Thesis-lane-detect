// 切單張影像.cpp : 定義主控台應用程式的進入點。 (0618)  內圈減外圈  有加進判斷是戶外還是隧道內
  //                                                               但是追影子的問題還是沒解決

#include <stdio.h>
#include <math.h>
#include "stdafx.h"

#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h> 
#include <algorithm>    // std::max


using namespace std;
using namespace cv;

typedef unsigned char uchar ;

# define PI 3.1415926
# define NUM 30  // vanishing point queue 的大小
# define VIDEO_WIDTH 640  // 定義影片大小
# define VIDEO_HEIGHT 480
# define TRACKING 0  // 看是否要追蹤  1為true 
# define WRITEVIDEO 1
# define WRITEFRAME 0
int frame_num ;

int vanishing_point_queue_x[NUM]; // 存放x 座標
int vanishing_point_queue_y[NUM]; // 存放y 座標
int sizeof_vanishing_point_queue = 0, index_in_queue = 0;
Point VP ; // 該畫面的vanish point 位置
Point left_lane_top_point,left_lane_bot_point,right_lane_top_point,right_lane_bot_point;
// 最後決定的左右車道 用global方式存

//CvCapture *CAPTURE;
VideoCapture CAPTURE ;
Rect FRAME_ROI_RECT;

void Find_HoughLines_and_VanishingPoint( cv::Mat image, cv::Mat frame,
	           Point &pre_RT, Point &pre_RB, Point &pre_LT, Point &pre_LB) ;// 找直線且找出vanishimg point
double Slope( int x1, int y1, int x2 , int y2 );  // 計算斜率
void laneMarkingDetector( cv::Mat &srcGRAY , cv::Mat &dstGRAY , int tau ) ; // 大師的filter
Point Find_Intersection_Point( Point LT, Point LB, Point RT, Point RB ) ; // 利用兩直線 找vanishing point
void AddInQueue(int data, int data1) ; // 加入東西
void quick_sort(int queue[],int low,int high) ; // QUICKSORT
Point Find_Vanishing_Point( Point ans ) ;  //存成queue 找vanishing point的中位數
double Distance_of_two_points( Point pt1, Point pt2 ) ; // 求兩點之間的距離
void Extand_line( Point pt1,Point pt2,Point &ans_pt1, Point &ans_pt2 ) ; //延伸車道線

bool LoadTrainingDetect( std::vector<float>&desc, char* filename, int DIM ) ;
void RunDetectionLoop(cv::Mat frame , Rect FRAME_ROI_RECT, std::vector<cv::Rect> &found, int type ) ;
void SelectingAndDrowingObject( cv::Mat frame, std::vector<cv::Rect> found ) ;
void Label_Tracking( cv::Mat temp_frame, cv::Mat frame,std::vector<cv::Rect> found ) ;   // 為追蹤的車子做編號
double CalculateDistance( CvPoint a , CvPoint b ) ;
bool If_tracking_rect_already_exist( CvPoint detect_pt, CvSize detect_size,cv::Mat frame ) ;
bool IsOverlap( cv::Rect rect1, cv::Rect rect2 ) ;
double Template_matching_rate( CvPoint a, CvSize a_size, CvPoint b, CvSize b_size, cv::Mat frame ) ; 
                   // 計算兩個template 的相似程度
bool If_Point_Outofframe( CvPoint pt, CvSize size ) ;
void Calculate_car_color( CvPoint position, CvSize size, cv::Mat frame, int which_Car ) ;
// 計算車身顏色  以及非車顏色 的中位數
double Point_Compare_Vehicle_Color( CvPoint pt, CvPoint midfound, int which_car ) ;
double detect_compare_tracking(CvPoint midfound, int which_car) ;
void Cvt1dto2d( IplImage *src, int **r, int **g, int **b ) ;// 這個function用來存影像的所有pixel
void Cvt2dto1d( int **r, int **g, int **b, IplImage *dst ) ;//這個function用來把三個通到的值assign回一張影像
double Calculating_overlap_rate(cv::Rect A, cv::Rect B) ; // 計算兩個矩形的覆蓋率
void Delete_tracking_overlap_tracking() ; // 刪掉有追蹤框重複的
bool Environment_is_Outdoor(); // 判斷現在是在戶外還是在隧道內

cv::Scalar BLUE = cv::Scalar(255,0,0) ;
cv::Scalar RED = cv::Scalar(0,0,255) ;
cv::Scalar GREEN = cv::Scalar(0,255,0) ;
cv::Scalar YELLOW = cv::Scalar(0,255,255) ;
cv::Scalar PURPLE = cv::Scalar(255,0,255) ;
cv::Scalar ORANGE = cv::Scalar(0,128,255) ;
cv::Scalar WHITE = cv::Scalar(255,255,255) ;
cv::Scalar DARK_RED = cv::Scalar(50,50,180) ;
cv::Scalar DARK_GREEN = cv::Scalar(50,180,50) ;
cv::Scalar DARK_BLUE = cv::Scalar(180,50,50) ;


int BIN = 9;
int DIM = 324;   
cv::Size SAMPLE_SIZE(32, 32);
cv::gpu::HOGDescriptor *GPU_MODE;
cv::gpu::HOGDescriptor *GPU_MODE_TUNNEL;
cv::HOGDescriptor *HOG_PTR;
bool Write_Retraining_Image = false ; // 是否要輸出retraining image
bool Write_Tracking_Result_image = false ;
int outputSample_Index = 0;  // 輸出retraining image 時的編號
int outputTracking_Index =0 ;
bool Is_TUNNEL = false ;
//char FILE_ADDR[300] = "C:\\Users\\Joy\\Desktop\\TME影片\\tme08.avi" ;
char FILE_ADDR[300] = "C:\\Users\\Joy\\Desktop\\dataset\\test data\\一般\\FILE0008.mov" ;
char Output[300]="C:\\Users\\Joy\\Desktop\\RESULT\\一般\\FILE2417\\DETECT\\";
char Output_frame[300]="C:\\Users\\Joy\\Desktop\\RESULT\\一般\\FILE2417\\";
char TRAIN_DATA_ADDR[300] = "C:\\Users\\Joy\\Desktop\\好的樣本\\一般ALL\\SVMD.txt"; // 戶外的SV
char TRAIN_DATA_ADDR_TUNNEL[300] = "C:\\Users\\Joy\\Desktop\\好的樣本\\隧道ALL\\SVMD.txt"; // 隧道的SV
char Tracking_ADDR[300] = "C:\\Users\\Joy\\Desktop\\RESULT\\一般\\FILE2417\\TRACKING\\";
char Tracking_ADDR_TUNNEL[300] = "C:\\Users\\Joy\\Desktop\\RESULT\\一般\\FILE2417\\TRACKING_TUNNEL\\";
char Output_TUNNEL[300] = "C:\\Users\\Joy\\Desktop\\RESULT\\一般\\FILE2417\\DETECT_TUNNEL\\";


class LineFinder{  
private:  
        // 直線對應的點參數向量
        std::vector<cv::Vec4i> lines;  
        //步長 
        double deltaRho;  
        double deltaTheta;  
        // 判斷是直線的最小投票數  
        int minVote;  
        // 判斷是直線的最小長度  
        double minLength;  
        // 同一條直線上點距離之間的最小容忍度 
        double maxGap;  
public:  
        //初始化  
        LineFinder() : deltaRho(1), deltaTheta(PI/180),  
        minVote(10), minLength(0.), maxGap(0.) {}  
        // 設置步長 
        void setAccResolution(double dRho, double dTheta) {  
                deltaRho= dRho;  
                deltaTheta= dTheta;  
        }  
        // 設置最小投票數
        void setMinVote(int minv) {  
                minVote= minv;  
        }  
        // 設置最小長度和最小投票數 
        void setLineLengthAndGap(double length, double gap) {  
                minLength= length;  
                maxGap= gap;  
        }  
  
        // 尋找線段  
        std::vector<cv::Vec4i> findLines(cv::Mat& binary) {  
                lines.clear();  
                cv::HoughLinesP(binary,lines, deltaRho, deltaTheta, minVote,minLength, maxGap);  
                return lines;  
        }  
  
        // 畫車道線段 
        void drawDetectedLines(cv::Mat &image) {           
          std::vector<cv::Vec4i>::const_iterator it2=lines.begin();  
          Point closest_left_top,closest_left_bot,closest_right_top,closest_right_bot; // 找到離VP最近的線
          double min_left = 1000, min_right = 1000;
          bool find_right = false, find_left = false ;

          if ( lines.size() != 0 ) {
            while (it2!=lines.end()) {  
              cv::Point pt1((*it2)[0],(*it2)[1]+150);  
              cv::Point pt2((*it2)[2],(*it2)[3]+150); 

				    if( abs(Slope( pt1.x, pt1.y, pt2.x ,pt2.y ) ) < 0.5) // 取絕對值後的斜率 太水平 不會是車道
				    	;
				    else if (Slope( pt1.x, pt1.y,pt2.x ,pt2.y ) > 0 ) {   // 右車道
				    	if ( pt1.y >240 && pt2.y > 240 && pt1.x > VP.x && pt2.x > VP.x ) {
                if ( Distance_of_two_points(  pt1, VP ) < min_right ) {
                  closest_right_top.x = pt1.x;
                  closest_right_top.y = pt1.y;
                  closest_right_bot.x = pt2.x;
                  closest_right_bot.y = pt2.y;
                  min_right = Distance_of_two_points(  pt1, VP ) ;
                  find_right = true ;
                }//if
              } // if
				    } // else if
				    else if (Slope( pt1.x, pt1.y, pt2.x ,pt2.y ) < 0 ) {  // 左車道
				    	if (pt1.y >240 && pt2.y > 240 && pt1.x < VP.x && pt2.x < VP.x ) {
                if ( Distance_of_two_points(  pt1, VP ) < min_left ) {
                  closest_left_top.x = pt1.x;
                  closest_left_top.y = pt1.y;
                  closest_left_bot.x = pt2.x;
                  closest_left_bot.y = pt2.y;
                  min_left = Distance_of_two_points( pt1, VP ) ;
                  find_left = true ;
                 }//if
               } // if
			    	 } // else if

                ++it2;
            }  // while

            if ( find_left == true && find_right == true ) {  // 若是左右兩車道都有找到
              Extand_line( closest_left_top,closest_left_bot,left_lane_top_point,left_lane_bot_point ) ;
              Extand_line( closest_right_top,closest_right_bot,right_lane_top_point,right_lane_bot_point ) ;
              cv::line( image, left_lane_top_point, left_lane_bot_point, BLUE,3,8,0);
              cv::line( image, right_lane_top_point, right_lane_bot_point, RED,3,8,0);
            } // if
            else {
              cv::line( image, left_lane_top_point, left_lane_bot_point, BLUE,3,8,0);
              cv::line( image, right_lane_top_point, right_lane_bot_point, RED,3,8,0);
            } // end else
          } // if
          else {
            cv::line( image, left_lane_top_point, left_lane_bot_point, BLUE,3,8,0);
            cv::line( image, right_lane_top_point, right_lane_bot_point, RED,3,8,0);
          } // else
        } // drawDetectedLines
}; // LineFinder()


//1.condensation setup  
const int stateNum=4;  
const int measureNum=2;  
const int sampleNum=600;

const double template_matching_threshold = 0.5;  // 追蹤框與偵測框比較色彩的閥值
const double spatial_distance_threshold = 50 ;
CvMat* lowerBound;  
CvMat* upperBound;  

struct Particle{
 bool Create_Condens ;   // 是否有創建這個particle filter
 bool Condens_work ;  // 這個particle 有沒有在使用
 CvSize Condens_size ;  // 這個particle 目前追蹤的框  size 應該是多少
 bool detecting_tracking_overlapping ;  // 是不是有跟偵測框重疊  也就是判斷追蹤根偵測是否是同一台車
 CvPoint temp_predict_pt ;  // 暫存前一個時刻的預測結果
 CvPoint detect_result_pt ;  // 記錄 偵測框的大小
 CvPoint pre_tracking_position ; // 前一刻追蹤的結果位置
 int counter_start ; // 計算多久沒有備偵測框框覆蓋的兩個變數
 int counter_end ;  // 計算多久沒有備偵測框框覆蓋的兩個變數
 cv::Scalar car_color ;
 cv::Scalar not_car_color;
 double R_car_color[256] ;    // 存這個particle 是車子的Histogram 
 double G_car_color[256] ;
 double B_car_color[256] ;
 double R_not_car_color[256] ;  // 存這個particle 不是車子的Histogram
 double G_not_car_color[256] ;
 double B_not_car_color[256] ;
 double B_confidence[256] ;  // 用來存非車扣掉車子的色彩histogram 要決定confidence的histogram
 double G_confidence[256] ;
 double R_confidence[256] ;
 bool If_Draw_Result ;
} ;
//宣告六個追蹤器
Particle particle_1, particle_2, particle_3, particle_4, particle_5, particle_6 ;
// 創建六個追蹤器

CvConDensation* condens_1 = cvCreateConDensation(stateNum,measureNum,sampleNum);
CvConDensation* condens_2 = cvCreateConDensation(stateNum,measureNum,sampleNum);
CvConDensation* condens_3 = cvCreateConDensation(stateNum,measureNum,sampleNum);
CvConDensation* condens_4 = cvCreateConDensation(stateNum,measureNum,sampleNum);
CvConDensation* condens_5 = cvCreateConDensation(stateNum,measureNum,sampleNum);
CvConDensation* condens_6 = cvCreateConDensation(stateNum,measureNum,sampleNum);


int **BLUE_ARRAY,**RED_ARRAY,**GREEN_ARRAY ;  // 存影像的pixel值

int control = 0 ;

int _tmain(int argc, _TCHAR* argv[])
{
	frame_num = 0 ;
  /* particle filter 設定*/	
    lowerBound = cvCreateMat(stateNum, 1, CV_32F);  
    upperBound = cvCreateMat(stateNum, 1, CV_32F);  
    cvmSet(lowerBound,0,0,0.0 );   
    cvmSet(upperBound,0,0,VIDEO_WIDTH );  
    cvmSet(lowerBound,1,0,0.0 );   
    cvmSet(upperBound,1,0,VIDEO_HEIGHT );  
    cvmSet(lowerBound,2,0,0.0);   
    cvmSet(upperBound,2,0,0.0);  
    cvmSet(lowerBound,3,0,0.0 );   
    cvmSet(upperBound,3,0,0.0 );  
    float A[stateNum][stateNum] ={  
        1,0,1,0,  
        0,1,0,1,  
        0,0,1,0,  
        0,0,0,1  
    };  
    memcpy(condens_1->DynamMatr,A,sizeof(A));  
    memcpy(condens_2->DynamMatr,A,sizeof(A));  
    memcpy(condens_3->DynamMatr,A,sizeof(A));  
    memcpy(condens_4->DynamMatr,A,sizeof(A));  
    memcpy(condens_5->DynamMatr,A,sizeof(A));  
    memcpy(condens_6->DynamMatr,A,sizeof(A));  
    cvConDensInitSampleSet(condens_1, lowerBound, upperBound);   
    cvConDensInitSampleSet(condens_2, lowerBound, upperBound);   
    cvConDensInitSampleSet(condens_3, lowerBound, upperBound);   
    cvConDensInitSampleSet(condens_4, lowerBound, upperBound);   
    cvConDensInitSampleSet(condens_5, lowerBound, upperBound);   
    cvConDensInitSampleSet(condens_6, lowerBound, upperBound);  
  
    CvRNG rng_state = cvRNG(0xffffffff);  
    for(int i=0; i < sampleNum; i++){  
        condens_1->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_1->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
        condens_2->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_2->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
        condens_3->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_3->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
        condens_4->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_4->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
        condens_5->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_5->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
        condens_6->flSamples[i][0] = float(cvRandInt( &rng_state ) % VIDEO_WIDTH); //width  
        condens_6->flSamples[i][1] = float(cvRandInt( &rng_state ) % VIDEO_HEIGHT);//height  
    }  

	particle_1.Condens_work = false ;
	particle_2.Condens_work = false ;
	particle_3.Condens_work = false ;
	particle_4.Condens_work = false ;
	particle_5.Condens_work = false ;
	particle_6.Condens_work = false ;

	particle_1.detecting_tracking_overlapping = false ;
	particle_2.detecting_tracking_overlapping = false ;
	particle_3.detecting_tracking_overlapping = false ;
	particle_4.detecting_tracking_overlapping = false ;
	particle_5.detecting_tracking_overlapping = false ;
	particle_6.detecting_tracking_overlapping = false ;

	particle_1.Create_Condens = false ;
	particle_2.Create_Condens = false ;
	particle_3.Create_Condens = false ;
	particle_4.Create_Condens = false ;
	particle_5.Create_Condens = false ;
	particle_6.Create_Condens = false ;

	particle_1.If_Draw_Result = false ;
	particle_2.If_Draw_Result = false ;
	particle_3.If_Draw_Result = false ;
	particle_4.If_Draw_Result = false ;
	particle_5.If_Draw_Result = false ;
	particle_6.If_Draw_Result = false ;

  /* particle filter 設定*/	



  cv::Mat frame;
  cv::Mat ROI_img;
  cv::Mat black_white_img ;
  cv::Mat temp_frame ;

  // 設定ROI座標
  FRAME_ROI_RECT = Rect(0,150,640,330) ; // 左上角x  左上角 y  ROI 寬  ROI 高
  // 設定ROI座標

  CAPTURE=VideoCapture(FILE_ADDR);  // 從指定檔案抓影像 (*) avi格式
  
  /* 是否要寫檔*/

# if WRITEVIDEO
  int AviForamt = CV_FOURCC('D', 'I', 'V', 'X')  ;   
  double FPS = CAPTURE.get( CV_CAP_PROP_FPS ) ;
  int AviColor = 1;

  VideoWriter writer("C:\\Users\\Joy\\Desktop\\result.avi", CV_FOURCC('M', 'J', 'P', 'G'), FPS, cvSize(VIDEO_WIDTH,VIDEO_HEIGHT)); 

#endif

  /*SVM HOG descriptor*/
  GPU_MODE = new cv::gpu::HOGDescriptor( SAMPLE_SIZE, cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), BIN,
	                                          cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
	                                          cv::gpu::HOGDescriptor::DEFAULT_NLEVELS );
  GPU_MODE_TUNNEL = new cv::gpu::HOGDescriptor( SAMPLE_SIZE, cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), BIN,
	                                          cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
	                                          cv::gpu::HOGDescriptor::DEFAULT_NLEVELS );											  
  /*HOG_PTR = new cv::HOGDescriptor( SAMPLE_SIZE, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 
                                     BIN, 1, -1, cv::HOGDescriptor::L2Hys, 0.2, false, cv::HOGDescriptor::DEFAULT_NLEVELS );
  */
  std::vector<float> training_descriptors ;
  std::vector<float> training_descriptors_TUNNEL ;
  if ( !LoadTrainingDetect( training_descriptors, TRAIN_DATA_ADDR, DIM ) 
	  || !LoadTrainingDetect( training_descriptors_TUNNEL,TRAIN_DATA_ADDR_TUNNEL, DIM) ) {
    std::cout << "****Error:SVMDescriptors載入失敗" << std::endl;
    system( "pause" );
    return 0;
  } // if
  else {
    GPU_MODE->setSVMDetector( training_descriptors );
	GPU_MODE_TUNNEL->setSVMDetector( training_descriptors_TUNNEL );
    //HOG_PTR.setSVMDetector( training_descriptors );
	std::cout << "SVMDescriptors載入成功" << std::endl;
    std::cout << "維度(含rho)：" << training_descriptors.size() << std::endl;
	system("pause");
  } // else

  /*SVM HOG descriptor*/

  std::vector<cv::Rect> found ; // 找到的車子位置  一堆框框的集合


  if ( !CAPTURE.isOpened() ) {
    std::cout << "****Error:影片讀取失敗" << std::endl;
    system( "pause" );
    return 0;
  } // if

  std::cout << "影片的fps是:" << CAPTURE.get( CV_CAP_PROP_FPS) <<endl;
  std::cout <<"這部影片總共有" << CAPTURE.get( CV_CAP_PROP_FRAME_COUNT) << "張frame" << endl;
  system("pause");

  Point pre_RT, pre_RB, pre_LT,pre_LB ; // 紀錄前一筆有資料的vanishing line 防止沒有找到要的vanishing line
  // 線段初始化為畫面又上到左下 以及左下到右上 的兩條線	
  pre_RT.x = 0;
  pre_RT.y = 0;
  pre_RB.x = frame.cols;
  pre_RB.y = frame.rows;
  pre_LT.x = frame.cols;
  pre_LT.y = 0;
  pre_LB.x = 0;
  pre_LB.y = frame.rows;
  // 紀錄要決定vanishing point的兩條線段


  // *************************************************
  CAPTURE.grab() ;
  CAPTURE.retrieve( frame );
  cv::resize(frame,frame,cv::Size(640,480));
  IplImage * pixel_img = cvCreateImage(cvSize(640,480),8,3);
  pixel_img = &IplImage(frame) ;
 
  /*存影像的pixel值到BGR中*/
  
  int i ;  
  int *pData ;  // 存image的pixel值 用二維取比較快

  BLUE_ARRAY = new int*[pixel_img->height];
  RED_ARRAY = new int*[pixel_img->height];
  GREEN_ARRAY = new int*[pixel_img->height];
  for ( i = 0 ; i < pixel_img->height ; i++ ) {
	  BLUE_ARRAY[i] = new int[pixel_img->width];
	  RED_ARRAY[i] = new int[pixel_img->width];
	  GREEN_ARRAY[i] = new int[pixel_img->width];
  } // for

   // ************************************************* 


  // QueryFrame = GrabFrame + RetrieveFrame !!!!!!!!!!!!!!!
  int detect_per_how_many_frame = 0;
  while ( CAPTURE.grab() ) {
    std::cout <<"這是第" << CAPTURE.get( CV_CAP_PROP_POS_FRAMES) << "張frame" << endl;
	//if ( CAPTURE.get( CV_CAP_PROP_POS_FRAMES) == 9 ) system("pause");

    CAPTURE.retrieve( frame );

	temp_frame = frame.clone() ;  // 複製一份影像
    cv::resize(frame,frame,cv::Size(640,480));
    cv::resize(temp_frame,temp_frame,cv::Size(640,480));

	pixel_img = &IplImage(frame) ;


	Cvt1dto2d(pixel_img,RED_ARRAY,GREEN_ARRAY,BLUE_ARRAY); // 把影像的值分別放到BGR的 Array 中

    double tt1 = (double)cvGetTickCount();	
	/*
	IplImage * R_img = cvCreateImage(cvSize(640,480),8,1);
	
	int k = 0 ;
	for ( int i =0 ;i<pixel_img->height;i++ ) {
		for( int j=0;j<pixel_img->width ;j++) {	
			double RR = (double)(uchar)RED_ARRAY[i][j] ;	
			double BB = (double)(uchar)BLUE_ARRAY[i][j] ;	
			double GG = (double)(uchar)GREEN_ARRAY[i][j] ;
			RR = max(RR-BB-GG-fabs(BB-GG),0.0);
		    double t = pow(RR/255.0,0.2)*1020.0 ;
			R_img->imageData[k] = (uchar)min(t,255.0) ;
			k++ ;
		} // for
	} // for
	
	cvNamedWindow("pixel_img pic",1);
	cvShowImage( "pixel_img pic", pixel_img);	
	cvNamedWindow("Red pic",1);
	cvShowImage( "Red pic", R_img);
	cvWaitKey(27) ;

	cvReleaseImage( &R_img );	
	*/
    tt1 = (double)cvGetTickCount() - tt1;	
    //printf( "Extraction time1 = %gms\n", tt1/(cvGetTickFrequency()*1000.));

	
    ROI_img = cv::Mat( frame, FRAME_ROI_RECT );  // 取得ROI
    cv::cvtColor(ROI_img, ROI_img, CV_BGR2GRAY); // 轉灰階圖
	black_white_img = ROI_img.clone();  // 複製一份影像

	//cv::Canny (ROI_img,ROI_img,125,350);  
    laneMarkingDetector( ROI_img , black_white_img , 45 ) ;   // 大師的轉灰階方法 類似濾波
    cv::threshold(black_white_img, black_white_img, 100, 255, CV_THRESH_OTSU);  


	

	// 轉成二值化影像 將亮度值在100以上的設為255  
	if (0) {
		/*偵測汽車*/
		if ( detect_per_how_many_frame == 1 ) { //  每幾張frame偵測一次???
		  //if ( control < 650 ) {
			  if ( Environment_is_Outdoor() ) { // 先判斷是戶外的還是隧道內的環境 
				cv::putText( frame, "OUTDOOR", cv::Point(10,30), CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				RunDetectionLoop( temp_frame ,FRAME_ROI_RECT, found,1 ) ; // 偵測完車子可以得到車子坐標的集合(found)
				Is_TUNNEL = false ;
			  } // if
			  else {
				cv::putText( frame, "TUNNEL", cv::Point(10,30), CV_FONT_HERSHEY_COMPLEX , 0.7, YELLOW, 2, 8, false );
				RunDetectionLoop( temp_frame ,FRAME_ROI_RECT, found,2 ) ; // 偵測完車子可以得到車子坐標的集合(found)
				Is_TUNNEL = true ;
			  } // else
		//  } // if

	      //SelectingAndDrowingObject( cv::Mat(frame,FRAME_ROI_RECT), found );  // 畫出框框


#if TRACKING
		  /* 拿偵測結果一個一個去跟追蹤的畫面比*/

		//if ( control < 650 ) {
		Label_Tracking( temp_frame,frame,found ) ; // 先替要追蹤的車子給他追蹤器
	    //control++ ;
	   // } // if
		//else if ( control == 650 ) {
		//   control++ ;
		//	  system("pause");
		//} // else if
	   // else{ ;}


  double slope_left = Slope( left_lane_top_point.x, left_lane_top_point.y, left_lane_bot_point.x, left_lane_bot_point.y );
  double offset_left = left_lane_bot_point.y - (slope_left*left_lane_bot_point.x) ;
  double slope_right = Slope(right_lane_top_point.x,right_lane_top_point.y,right_lane_bot_point.x,right_lane_bot_point.y );
  double offset_right = right_lane_bot_point.y - (slope_right*right_lane_bot_point.x) ;




		if ( particle_1.Condens_work == true ) {
			CvPoint predict_pt1=cvPoint((int)condens_1->State[0],(int)condens_1->State[1]);

			particle_1.temp_predict_pt.x = predict_pt1.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_1.temp_predict_pt.y = predict_pt1.y;
			particle_1.pre_tracking_position.x = predict_pt1.x;
			particle_1.pre_tracking_position.y = predict_pt1.y;

			int radius_draw = particle_1.Condens_size.width/2;  // 畫追蹤圓的半徑

			for (int i=0;i<condens_1->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_1->flSamples[i][0],condens_1->flSamples[i][1]);
				//cv::circle(frame,a,1,YELLOW,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_1.Condens_size.width, h = particle_1.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_1->flSamples[i][0]-(w/2) > 0 && condens_1->flSamples[i][1]-(h/2) > 0 && 
					 condens_1->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_1->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_1->flSamples[i][0],condens_1->flSamples[i][1]);
					if ( particle_1.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt1.x = particle_1.detect_result_pt.x ;
						predict_pt1.y = particle_1.detect_result_pt.y ;
						condens_1->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt1,1) ;				
						particle_1.If_Draw_Result = false ;
						} // if
					else if (particle_1.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_1.Create_Condens = false ;		
						particle_1.If_Draw_Result = false ;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_1->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt1,1) ;	
						particle_1.If_Draw_Result = true ;								
						} // else
				} // if
			}  // for
			particle_1.detecting_tracking_overlapping = false ;
			//4.update condensation  

			cvConDensUpdateByTime(condens_1); 
			//cv::circle(frame,predict_pt1,5,particle_1.not_car_color,3);
			if ( particle_1.If_Draw_Result == true ) {

				if ( predict_pt1.y-(slope_left*predict_pt1.x)-offset_left < 0 ) {
				  cv::putText( frame, "1", predict_pt1, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt1,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt1.y-(slope_right*predict_pt1.x)-offset_right < 0) {
				  cv::putText( frame, "1", predict_pt1, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt1,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "1", predict_pt1, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt1,radius_draw,ORANGE,3);//predicted point with green  
				} // else
				/*
			  cv::putText( frame, "1", predict_pt1, CV_FONT_HERSHEY_COMPLEX , 0.7, DARK_RED, 2, 8, false );
			  cv::circle(frame,predict_pt1,radius_draw,DARK_RED,3);//predicted point with green  
			  */

				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt1.x-radius_draw > 0 && predict_pt1.y-radius_draw > 0
					&& predict_pt1.x+radius_draw < 640 && predict_pt1.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt1.x-radius_draw,predict_pt1.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

							cv::imwrite( str, retraining);
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if
			} // if
		} // if
		if ( particle_2.Condens_work == true ) {
			CvPoint predict_pt2=cvPoint((int)condens_2->State[0],(int)condens_2->State[1]);

			particle_2.temp_predict_pt.x = predict_pt2.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_2.temp_predict_pt.y = predict_pt2.y;
			particle_2.pre_tracking_position.x = predict_pt2.x;
			particle_2.pre_tracking_position.y = predict_pt2.y;

			int radius_draw = particle_2.Condens_size.width/2;  // 畫追蹤圓的半徑


			for (int i=0;i<condens_2->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_2->flSamples[i][0],condens_2->flSamples[i][1]);
				//cv::circle(frame,a,1,RED,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_2.Condens_size.width, h = particle_2.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_2->flSamples[i][0]-(w/2) > 0 && condens_2->flSamples[i][1]-(h/2) > 0 && 
					 condens_2->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_2->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_2->flSamples[i][0],condens_2->flSamples[i][1]);
					if ( particle_2.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt2.x = particle_2.detect_result_pt.x ;
						predict_pt2.y = particle_2.detect_result_pt.y ;
						condens_2->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt2,2) ;
						particle_2.If_Draw_Result = false ;
					} // if
					else if (particle_2.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_2.Create_Condens = false ;
						particle_2.If_Draw_Result = false ;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_2->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt2,2) ;
						particle_2.If_Draw_Result = true ;
						} // else
				} // if
			}  // for

			particle_2.detecting_tracking_overlapping = false ;
			//4.update condensation  

			cvConDensUpdateByTime(condens_2); 
			//cv::circle(frame,predict_pt6,5,particle_6.not_car_color,3);
		if ( particle_2.If_Draw_Result == true ) {

				if ( predict_pt2.y-(slope_left*predict_pt2.x)-offset_left < 0 ) {
				  cv::putText( frame, "2", predict_pt2, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt2,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt2.y-(slope_right*predict_pt2.x)-offset_right < 0) {
				  cv::putText( frame, "2", predict_pt2, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt2,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "2", predict_pt2, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt2,radius_draw,ORANGE,3);//predicted point with green  
				} // else


		    /*cv::putText( frame, "2", predict_pt2, CV_FONT_HERSHEY_COMPLEX , 0.7, DARK_GREEN, 2, 8, false );
			cv::circle(frame,predict_pt2,radius_draw,DARK_GREEN,3);//predicted point with green  
			*/
				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt2.x-radius_draw > 0 && predict_pt2.y-radius_draw > 0
					&& predict_pt2.x+radius_draw < 640 && predict_pt2.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt2.x-radius_draw,predict_pt2.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

	
							cv::imwrite( str, retraining );
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if

		} // if


		} // if
		if ( particle_3.Condens_work == true ) {
			CvPoint predict_pt3=cvPoint((int)condens_3->State[0],(int)condens_3->State[1]);

			particle_3.temp_predict_pt.x = predict_pt3.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_3.temp_predict_pt.y = predict_pt3.y;
			particle_3.pre_tracking_position.x = predict_pt3.x;
			particle_3.pre_tracking_position.y = predict_pt3.y;

			int radius_draw = particle_3.Condens_size.width/2;  // 畫追蹤圓的半徑


			for (int i=0;i<condens_3->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_3->flSamples[i][0],condens_3->flSamples[i][1]);
				//cv::circle(frame,a,1,RED,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_3.Condens_size.width, h = particle_3.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_3->flSamples[i][0]-(w/2) > 0 && condens_3->flSamples[i][1]-(h/2) > 0 && 
					 condens_3->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_3->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_3->flSamples[i][0],condens_3->flSamples[i][1]);
					if ( particle_3.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt3.x = particle_3.detect_result_pt.x ;
						predict_pt3.y = particle_3.detect_result_pt.y ;
						condens_3->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt3,3) ;
						particle_3.If_Draw_Result = false ;
						} // if
					else if (particle_3.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_3.Create_Condens = false ;
						particle_3.If_Draw_Result = false ;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_3->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt3,3) ;
						particle_3.If_Draw_Result = true ;
						} // else
				} // if
			}  // for
			particle_3.detecting_tracking_overlapping = false ;

			//4.update condensation  

			cvConDensUpdateByTime(condens_3); 
			//cv::circle(frame,predict_pt3,5,particle_3.not_car_color,3);
			if (particle_3.If_Draw_Result == true) {

				if ( predict_pt3.y-(slope_left*predict_pt3.x)-offset_left < 0 ) {
				  cv::putText( frame, "3", predict_pt3, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt3,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt3.y-(slope_right*predict_pt3.x)-offset_right < 0) {
				  cv::putText( frame, "3", predict_pt3, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt3,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "3", predict_pt3, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt3,radius_draw,ORANGE,3);//predicted point with green  
				} // else

				/*cv::putText( frame, "3", predict_pt3, CV_FONT_HERSHEY_COMPLEX , 0.7, DARK_BLUE, 2, 8, false );
				cv::circle(frame,predict_pt3,radius_draw,DARK_BLUE,3);//predicted point with green*/
			
				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt3.x-radius_draw > 0 && predict_pt3.y-radius_draw > 0
					&& predict_pt3.x+radius_draw < 640 && predict_pt3.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt3.x-radius_draw,predict_pt3.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

	
							cv::imwrite( str, retraining );
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if			
			
			
			} // if



		} // if
		if ( particle_4.Condens_work == true ) {
			CvPoint predict_pt4=cvPoint((int)condens_4->State[0],(int)condens_4->State[1]);

			particle_4.temp_predict_pt.x = predict_pt4.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_4.temp_predict_pt.y = predict_pt4.y;
			particle_4.pre_tracking_position.x = predict_pt4.x;
			particle_4.pre_tracking_position.y = predict_pt4.y;


			int radius_draw = particle_4.Condens_size.width/2;  // 畫追蹤圓的半徑


			for (int i=0;i<condens_4->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_4->flSamples[i][0],condens_4->flSamples[i][1]);
				//cv::circle(frame,a,1,GREEN,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_4.Condens_size.width, h = particle_4.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_4->flSamples[i][0]-(w/2) > 0 && condens_4->flSamples[i][1]-(h/2) > 0 && 
					 condens_4->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_4->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_4->flSamples[i][0],condens_4->flSamples[i][1]);
					if ( particle_4.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt4.x = particle_4.detect_result_pt.x ;
						predict_pt4.y = particle_4.detect_result_pt.y ;
						condens_4->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt4,4) ;
						particle_4.If_Draw_Result = false ;
					} // if
					else if (particle_4.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_4.Create_Condens = false ;
						particle_4.If_Draw_Result = false ;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_4->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt4,4) ;
						particle_4.If_Draw_Result = true ;
						} // else
				} // if
			}  // for
			
			particle_4.detecting_tracking_overlapping = false ;
			//4.update condensation  

			cvConDensUpdateByTime(condens_4); 
			//cv::circle(frame,predict_pt4,5,particle_4.not_car_color,3);
			if ( particle_4.If_Draw_Result == true  ) {
				if ( predict_pt4.y-(slope_left*predict_pt4.x)-offset_left < 0 ) {
				  cv::putText( frame, "4", predict_pt4, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt4,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt4.y-(slope_right*predict_pt4.x)-offset_right < 0) {
				  cv::putText( frame, "4", predict_pt4, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt4,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "4", predict_pt4, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt4,radius_draw,ORANGE,3);//predicted point with green  
				} // else


				/*cv::putText( frame, "4", predict_pt4, CV_FONT_HERSHEY_COMPLEX , 0.7, YELLOW, 2, 8, false );
				cv::circle(frame,predict_pt4,radius_draw,YELLOW,3);//predicted point with green
			*/
				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt4.x-radius_draw > 0 && predict_pt4.y-radius_draw > 0
					&& predict_pt4.x+radius_draw < 640 && predict_pt4.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt4.x-radius_draw,predict_pt4.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

	
							cv::imwrite( str, retraining );
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if			
						
			
			
			} // if


		} // if
		if ( particle_5.Condens_work == true ) {
			CvPoint predict_pt5=cvPoint((int)condens_5->State[0],(int)condens_5->State[1]);

			particle_5.temp_predict_pt.x = predict_pt5.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_5.temp_predict_pt.y = predict_pt5.y;
			particle_5.pre_tracking_position.x = predict_pt5.x;
			particle_5.pre_tracking_position.y = predict_pt5.y;

			int radius_draw = particle_5.Condens_size.width/2;  // 畫追蹤圓的半徑


			for (int i=0;i<condens_5->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_5->flSamples[i][0],condens_5->flSamples[i][1]);
				//cv::circle(frame,a,1,WHITE,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_5.Condens_size.width, h = particle_5.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_5->flSamples[i][0]-(w/2) > 0 && condens_5->flSamples[i][1]-(h/2) > 0 && 
					 condens_5->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_5->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_5->flSamples[i][0],condens_5->flSamples[i][1]);
					if ( particle_5.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt5.x = particle_5.detect_result_pt.x ;
						predict_pt5.y = particle_5.detect_result_pt.y ;
						condens_5->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt5,5) ;
						particle_5.If_Draw_Result = false;
						} // if
					else if (particle_5.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_5.Create_Condens = false ;
						particle_5.If_Draw_Result = false;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_5->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt5,5) ;
						particle_5.If_Draw_Result = true;
						} // else
				} // if
			}  // for

		    particle_5.detecting_tracking_overlapping = false ;
     
			//4.update condensation  

			cvConDensUpdateByTime(condens_5); 
			//cv::circle(frame,predict_pt5,5,particle_5.not_car_color,3);
			if ( particle_5.If_Draw_Result == true ) { 

				if ( predict_pt5.y-(slope_left*predict_pt5.x)-offset_left < 0 ) {
				  cv::putText( frame, "5", predict_pt5, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt5,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt5.y-(slope_right*predict_pt5.x)-offset_right < 0) {
				  cv::putText( frame, "5", predict_pt5, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt5,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "5", predict_pt5, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt5,radius_draw,ORANGE,3);//predicted point with green  
				} // else



				/*
				cv::putText( frame, "5", predict_pt5, CV_FONT_HERSHEY_COMPLEX , 0.7, PURPLE, 2, 8, false );
				cv::circle(frame,predict_pt5,radius_draw,PURPLE,3);//predicted point with green 
		*/
				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt5.x-radius_draw > 0 && predict_pt5.y-radius_draw > 0
					&& predict_pt5.x+radius_draw < 640 && predict_pt5.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt5.x-radius_draw,predict_pt5.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

	
							cv::imwrite( str, retraining );
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if			
			
			
			} // if


		} // if
		if ( particle_6.Condens_work == true ) {
			CvPoint predict_pt6=cvPoint((int)condens_6->State[0],(int)condens_6->State[1]);

			particle_6.temp_predict_pt.x = predict_pt6.x; // 暫存  給後續比對是否有偵測的框框跟他接近
			particle_6.temp_predict_pt.y = predict_pt6.y;
			particle_6.pre_tracking_position.x = predict_pt6.x;
			particle_6.pre_tracking_position.y = predict_pt6.y;

			int radius_draw = particle_6.Condens_size.width/2;  // 畫追蹤圓的半徑


			for (int i=0;i<condens_6->SamplesNum;i++) {  
				// 畫particle
				
				CvPoint a = cvPoint(condens_6->flSamples[i][0],condens_6->flSamples[i][1]);
				//cv::circle(frame,a,1,DARK_BLUE,1);
				
				// 追蹤框的比對大小是根據偵測框框而來的

				int w = particle_6.Condens_size.width, h = particle_6.Condens_size.height ;
				// 框出每個particle 要去跟偵測框框比較hist的框
				
				if ( condens_6->flSamples[i][0]-(w/2) > 0 && condens_6->flSamples[i][1]-(h/2) > 0 && 
					 condens_6->flSamples[i][0]+(h/2) < VIDEO_WIDTH && condens_6->flSamples[i][1]+(h/2)<VIDEO_HEIGHT) {
				
					CvPoint temp_sample=cvPoint(condens_6->flSamples[i][0],condens_6->flSamples[i][1]);
					if ( particle_6.detecting_tracking_overlapping == true ) {  
					//   如果有根偵測的框框距離很接近   就用偵測的框框的中心點	
						predict_pt6.x = particle_6.detect_result_pt.x ;
						predict_pt6.y = particle_6.detect_result_pt.y ;
						condens_6->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt6,6) ;
						particle_6.If_Draw_Result = false ;
						} // if
					else if (particle_6.Create_Condens == true) {  // 如果是剛創建好的新的追蹤框 就用偵測結果			
						particle_6.Create_Condens = false ;
						particle_6.If_Draw_Result = false ;
					} // else if
					else {
						// 如果是沒有上述兩項  就用預測的結果當作比較的點
						condens_6->flConfidence[i] = Point_Compare_Vehicle_Color(temp_sample,predict_pt6,6) ;
						particle_6.If_Draw_Result = true ;
					} // else
				} // if
			}  // for

			particle_6.detecting_tracking_overlapping = false ;
			//4.update condensation  

			cvConDensUpdateByTime(condens_6); 
			//cv::circle(frame,predict_pt6,5,particle_6.not_car_color,3);
			if ( particle_6.If_Draw_Result == true ) {

				if ( predict_pt6.y-(slope_left*predict_pt6.x)-offset_left < 0 ) {
				  cv::putText( frame, "6", predict_pt6, CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
				  cv::circle(frame,predict_pt6,radius_draw,BLUE,3);//predicted point with green  
				} // if
				else if(predict_pt6.y-(slope_right*predict_pt6.x)-offset_right < 0) {
				  cv::putText( frame, "6", predict_pt6, CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
				  cv::circle(frame,predict_pt6,radius_draw,RED,3);//predicted point with green  
				} // else if
				else {
				  cv::putText( frame, "6", predict_pt6, CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
				  cv::circle(frame,predict_pt6,radius_draw,ORANGE,3);//predicted point with green  
				} // else



				/*
				cv::putText( frame, "6", predict_pt6, CV_FONT_HERSHEY_COMPLEX , 0.7, WHITE, 2, 8, false );
				cv::circle(frame,predict_pt6,radius_draw,WHITE,3);//predicted point with green  
			*/
				cv::Mat sample ;
				frame.copyTo(sample) ;
	      
				if ( predict_pt6.x-radius_draw > 0 && predict_pt6.y-radius_draw > 0
					&& predict_pt6.x+radius_draw < 640 && predict_pt6.y+radius_draw < 480) {
				  cv::Rect r = Rect( predict_pt6.x-radius_draw,predict_pt6.y-radius_draw,2*radius_draw,2*radius_draw);	  
				  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

				  bool OutOfRange = true ;
				  if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
				    OutOfRange = false ;

				/*寫出tracking 結果的圖片*/
				  
				  if ( Write_Tracking_Result_image == true ) {
						if( OutOfRange == false ) {
							cv::Mat retraining = cv::Mat( sample, r ) ;
							cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
							char str[200];
							if ( Is_TUNNEL == false ) 
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR,outputTracking_Index );
							else if ( Is_TUNNEL == true )
							  sprintf( str, "%s_%d.JPEG", Tracking_ADDR_TUNNEL,outputTracking_Index );

	
							cv::imwrite( str, retraining );
							outputTracking_Index++;
							retraining.release();
						} // if
				  } // if
				  
				sample.release();
			  } // if				
			
			
			
			} // if


		} // if

#endif

          SelectingAndDrowingObject( cv::Mat(frame,FRAME_ROI_RECT), found );  // 畫出偵測框框

		  detect_per_how_many_frame = 0 ;
		} // if
		detect_per_how_many_frame++;

	} // if

	if (1) {  // 要不要開啟車道偵測??????
		/* 車道 */   
		LineFinder finder ;
		finder.setMinVote (50);  
		finder.setLineLengthAndGap (50,30);  
	
		Find_HoughLines_and_VanishingPoint( black_white_img, frame, pre_RT, pre_RB, pre_LT, pre_LB);
	
		finder.findLines (black_white_img);    
		finder.drawDetectedLines (frame);  
		/* 車道 */  
	} // if


# if WRITEVIDEO

	writer << frame ; // 寫出影片
	/*IplImage * result = &IplImage( frame );
	cvWriteFrame(writer,result);*/
#endif
	/*
	cv::line( frame, Point(0,340), Point(640,340), WHITE ,3,8,0);

	cv::line( frame, Point(304,154),Point(336,154), WHITE,3,8,0);
	cv::line( frame, Point(336,154),Point(336,186), WHITE,3,8,0);
	cv::line( frame, Point(304,186),Point(336,186), WHITE,3,8,0);
	cv::line( frame, Point(304,154),Point(304,186), WHITE,3,8,0);
	*/
# if WRITEFRAME
	char str[500];
	sprintf( str, "%s_%d.jpeg", Output_frame,frame_num );
	cv::imwrite( str, frame );
	frame_num++ ;
# endif

	imshow("濾波後結果圖", black_white_img);
	imshow("結果圖", frame);
    imshow("二值化影像",ROI_img) ;

    if( waitKey (30) >= 0) break;

	black_white_img.release();
	frame.release();
	ROI_img.release();
	temp_frame.release();
  } // while

  std::cout << "DONE!!!";
  system("pause");

  

  frame.release();


  free(BLUE_ARRAY);
  free(GREEN_ARRAY);
  free(RED_ARRAY);

  return 0;
} // main()
//*******************************************************************
bool LoadTrainingDetect( std::vector<float>&desc, char* filename, int DIM ) {
    FILE*file = fopen( filename, "r" );
    if ( !file )
        return false;
    float input = 0.0;
    for( int i = 0 ; i < DIM+1; i++ ) {
        desc.push_back( 0.0 ) ;
        fscanf( file, "%f", &desc[i] );
    } // for
    return true;
} // LoadTrainingDetec()
//*******************************************************************
void AddInQueue(int data, int data1) { 

	// 如果放滿了 從第一個開始依次交換
	if ( index_in_queue >= NUM ) index_in_queue = 0; // 換回頭

	vanishing_point_queue_x[index_in_queue] = data ;
	vanishing_point_queue_y[index_in_queue] = data1 ;
	index_in_queue++;

	if ( sizeof_vanishing_point_queue < NUM ) sizeof_vanishing_point_queue++;

} // AddInQueue()
//*******************************************************************
void Find_HoughLines_and_VanishingPoint( cv::Mat image, cv::Mat frame,
	                                     Point &pre_RT, Point &pre_RB, Point &pre_LT, Point &pre_LB) {

    Point Vanish_Right_Line_Top, Vanish_Right_Line_Bot , Vanish_Left_Line_Top , Vanish_Left_Line_Bot ;
    std::vector<cv::Vec2f> lines; 
    //霍夫變換，獲得一組及座標參數(RHO,theta) 每一對對應一條直線，保存到lines 
    //地3.4個參數表示在(rho,theta) 座標系李恆縱座標的最小單位，即步長
    HoughLines(image, lines,1,CV_PI/180,60);
	  bool find_left = false, find_right = false ;
     for( size_t i =0 ; i < lines.size(); i++ ) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta) ;
        double x0 = a*rho, y0 = b*rho;

        Point pt1(cvRound(x0+1000*(-b)), cvRound(y0+1000*(a))+150);
        Point pt2(cvRound(x0-1000*(-b)), cvRound(y0-1000*(a))+150);

        cv::clipLine(frame.size(),pt1,pt2) ;

		//line(frame,pt1,pt2,YELLOW,1,8);
		    if ( find_right == true && find_left == true ) 
              break ;
		    else if ( abs(Slope( pt1.x, pt1.y , pt2.x , pt2.y )) < 0.5 )
			    ;
		    else if ( Slope( pt1.x, pt1.y, pt2.x , pt2.y ) > 0.5 && Slope( pt1.x, pt1.y, pt2.x , pt2.y )< 2) {
			    find_right = true ;
			    Vanish_Right_Line_Top.x = pt1.x;   // 決定vanishing point的右線
			    Vanish_Right_Line_Top.y = pt1.y;
			    Vanish_Right_Line_Bot.x = pt2.x;
			    Vanish_Right_Line_Bot.y = pt2.y;
				//line(frame,pt1,pt2,YELLOW,1,8);
		    } // else if
		    else if ( Slope( pt1.x, pt1.y , pt2.x , pt2.y ) < -0.5 && Slope( pt1.x, pt1.y , pt2.x , pt2.y )>-2) {
			    find_left = true ;
			    Vanish_Left_Line_Top.x = pt1.x;  // 決定vanishing point 的左線
			    Vanish_Left_Line_Top.y = pt1.y;
			    Vanish_Left_Line_Bot.x = pt2.x;
			    Vanish_Left_Line_Bot.y = pt2.y;
				//line(frame,pt1,pt2,YELLOW,1,8);
		    } // else if

        
    } // for

	if ( find_right == true && find_left == true ) { // 若是有找到vanishing line 記錄下來這一筆
		pre_RT.x = Vanish_Right_Line_Top.x;
		pre_RT.y = Vanish_Right_Line_Top.y;
		pre_RB.x = Vanish_Right_Line_Bot.x;
		pre_RB.y = Vanish_Right_Line_Bot.y;
		pre_LT.x = Vanish_Left_Line_Top.x;
		pre_LT.y = Vanish_Left_Line_Top.y;
		pre_LB.x = Vanish_Left_Line_Bot.x;
		pre_LB.y = Vanish_Left_Line_Bot.y;		
	} // if
	else {                  // 若是沒有找到條件內的vanishing line 用之前紀錄的
		Vanish_Right_Line_Top.x = pre_RT.x;
		Vanish_Right_Line_Top.y = pre_RT.y;
		Vanish_Right_Line_Bot.x = pre_RB.x;
		Vanish_Right_Line_Bot.y = pre_RB.y;
		Vanish_Left_Line_Top.x = pre_LT.x;
		Vanish_Left_Line_Top.y = pre_LT.y;
		Vanish_Left_Line_Bot.x = pre_LB.x;
		Vanish_Left_Line_Bot.y = pre_LB.y;
	} // else 

	// 這邊開始找 vanishinf point
	Point Intersection = Find_Intersection_Point( Vanish_Left_Line_Top,Vanish_Left_Line_Bot,Vanish_Right_Line_Top,Vanish_Right_Line_Bot );
     // 先找到交點

    VP = Find_Vanishing_Point(Intersection) ; // 找要求的vanishing point 
  
	//cv::line( frame, Vanish_Left_Line_Top, Vanish_Left_Line_Bot, PURPLE,2,8,0);
	//cv::line( frame, Vanish_Right_Line_Top, Vanish_Right_Line_Bot, PURPLE,2,8,0);
  
	Point a,b,c,d,e,f,g,h ;
	a.x = VP.x-10 ;
	a.y = VP.y;
	b.x = VP.x+10 ;
	b.y = VP.y;
	c.x = VP.x;
	c.y = VP.y-10;
	d.x = VP.x;
	d.y = VP.y+10;

	e.x = VP.x-10 ;
	e.y = VP.y;
	f.x = VP.x+10 ;
	f.y = VP.y;
	g.x = VP.x;
	g.y = VP.y-10;
	h.x = VP.x;
	h.y = VP.y+10;
  
  
	cv::line( frame, a, b, GREEN ,3,8,0);
	cv::line( frame, c, d, GREEN ,3,8,0);
	cv::line( frame, e, f, GREEN ,3,8,0);
	cv::line( frame, g, h, GREEN ,3,8,0);
  

}// Find_HoughLines_and_VanishingPoint()
//*******************************************************************
Point Find_Intersection_Point( Point LT, Point LB, Point RT, Point RB ) {  // y = mx+b

	Point ans ; // 要求的交點  vanishing point
	double Slope_Left = Slope( LT.x, LT.y, LB.x, LB.y );  // 求斜率
	double Slope_Right = Slope( RT.x, RT.y, RB.x, RB.y );

	double offset_Left = LB.y - (Slope_Left*LB.x);  // b = y-mx
	double offset_Right = RB.y - (Slope_Right*RB.x);

	ans.x =(offset_Right-offset_Left)/(Slope_Left-Slope_Right);
	                        // m1*x + b1 = m2*x + b2    =>  x = (b2 - b1)/(m1 - m2)
	ans.y = (Slope_Right*ans.x) + offset_Right ;

	return ans ;
} // Find_Intersection_Point()
//*******************************************************************
void Extand_line( Point pt1,Point pt2,Point &ans_pt1, Point &ans_pt2 ) {
  int x ;
  double slope = Slope( pt1.x, pt1.y, pt2.x, pt2.y );
  double offset = pt2.y - (slope*pt2.x) ;
  /*向上延伸到VP*/
  ans_pt1.x = (VP.y-offset)/slope ;
  ans_pt1.y = VP.y;
  /*向上延伸到VP*/
  /*向下延伸到畫面大小*/
  if ( slope < 0 ) {  // left
    x = (480-offset)/slope;
    if ( x >=0 && x <= 640 ) {
      ans_pt2.x = x;
      ans_pt2.y = 480 ;
    } // if
    else {
      ans_pt2.x = 0 ;
      ans_pt2.y = (slope*0)+offset;
    } // else
  } // if
  else if ( slope > 0 ) {  // right
    x = (480-offset)/slope;
    if ( x >=0 && x <= 640 ) {
      ans_pt2.x = x;
      ans_pt2.y = 480 ;
    } // if
    else {
      ans_pt2.x = 640 ;
      ans_pt2.y = (slope*640)+offset;
    } // else
  } // else if
  /*向下延伸到畫面大小*/

} // Extand_line()
//*******************************************************************
double Distance_of_two_points( Point pt1, Point pt2 ) {
  double a = pt1.x-pt2.x;
  double b = pt1.y-pt2.y;
  double c = pow(a,2) + pow(b,2);
  c = pow(c,0.5) ;
  return c ;
} // double Distance_of_two_points()
//*******************************************************************
double Slope( int x1, int y1, int x2 , int y2 ) {
  return (double)(y2-y1)/(double)(x2-x1) ; 
} // Slope
//*******************************************************************
void laneMarkingDetector( cv::Mat &srcGRAY , cv::Mat &dstGRAY , int tau ) {
  dstGRAY.setTo(0);

  int aux = 0;
  for ( int j=0; j < srcGRAY.rows ; ++j ) {
    unsigned char *ptRowSrc = srcGRAY.ptr<uchar>(j);
	  unsigned char *ptRowDst = dstGRAY.ptr<uchar>(j);

	  for ( int i=tau ; i < srcGRAY.cols-tau ; ++i ) {
        if( ptRowSrc[i] != 0 ) {
          aux = 2*ptRowSrc[i];
          aux += -ptRowSrc[i-tau];
          aux += -ptRowSrc[i+tau];
          aux += -abs((int)(ptRowSrc[i-tau] - ptRowSrc[i+tau])) ;

          aux = (aux<0) ? (0):(aux);
          aux = (aux>255) ? (255):aux;

          ptRowDst[i] = (unsigned char)aux ;
        } // if
	  } // for
  } // for

} // laneMarkingDetector()
//*******************************************************************
void quick_sort(int *queue,int low,int high) {  
   int pivot_point,pivot_item,i,j,temp;  
   // 指標交界結束排序  
  // cout << queue[high] ;
   if(high<=low){return ;}  
  
   // 紀錄樞紐值  
    //  cout << queue[low] ;
   pivot_item = queue[low];  
   j=low;  
     
   // 尋找比樞紐小的數  
   for(i=low+1; i<=high; i++) {  
       // 跳過等於或大於的數  
       if(queue[i]>=pivot_item){continue;}  
  
       j++;  
       // 交換 array[i] , array[j]  
       temp = queue[i];  
       queue[i] = queue[j];  
       queue[j] = temp;  
   }  
  
   // 將樞紐位址移到中間  
   pivot_point=j;  
   // 交換 array[low] , array[pivot_point]  
   temp = queue[low];  
   queue[low] = queue[pivot_point];  
   queue[pivot_point] = temp;  
   // 遞迴處理左側區段  
   quick_sort(queue,low,pivot_point-1);  
   // 遞迴處理右側區段  
   quick_sort(queue,pivot_point+1,high);  
  
} // quick_sort()

//*******************************************************************
Point Find_Vanishing_Point( Point ans ) {  
	 // 要求的交點  vanishing point
  if (ans.y >= 200 && ans.y <= 280 ) // 若是交點的位置 不在此範圍內就不要家進queue裡
	  AddInQueue( ans.x, ans.y ); // 將這次得到的vanishing point 存進去 

  int *array_for_x = new int[NUM]; // 動態宣告 暫存用的queue
  int *array_for_y = new int[NUM];

  for( int i =0; i < NUM ; i++ ) {   // 把資料暫存到動態宣告的array中做quick sort  原本的queue資料不動
    array_for_x[i] = vanishing_point_queue_x[i];
    array_for_y[i] = vanishing_point_queue_y[i];
  } // end for

  quick_sort( array_for_x,0,sizeof_vanishing_point_queue-1); // 對x位置做排序後取中位數
  quick_sort( array_for_y,0,sizeof_vanishing_point_queue-1); // 對y位置做排序後取中位數

  ans.x = array_for_x[sizeof_vanishing_point_queue/2];
  ans.y = array_for_y[sizeof_vanishing_point_queue/2];

  delete []array_for_x;
  delete []array_for_y;
  
  return ans ;
} // Find_Vanishing_Point()
//*******************************************************************
bool Environment_is_Outdoor(){  // 若當前的環境是戶外回傳true   隧道的話就是false
	int *B = new int[3200];  // 畫面的上半部 中間垂直一條的位置
	int *G = new int[3200];
	int *R = new int[3200];

	int index = 0 ;
	
	for ( int i = 0 ; i < 100 ; i++ ) {
		for ( int j = 304 ; j < 336 ; j++ ) {
			B[index] = (int)(uchar)BLUE_ARRAY[i][j];
			G[index] = (int)(uchar)GREEN_ARRAY[i][j];
			R[index] = (int)(uchar)RED_ARRAY[i][j];
			index++;
		} // for
	} // for

	quick_sort( B,0,index-1);
	quick_sort( G,0,index-1);
	quick_sort( R,0,index-1);


	if ( (B[index/2]+G[index/2]+R[index/2])/3 > 100 ) {  //  若是戶外的情況  中位數的亮度大於一定的值
		delete []B;
		delete []G;
		delete []R;
		return true ;
	} // if
	else {
		delete []B;
		delete []G;
		delete []R;
		return false ;
	} // else

} // Environment_is_Outdoor()
//*******************************************************************
void RunDetectionLoop(cv::Mat frame , Rect roi, std::vector<cv::Rect> &found, int type) {
	Mat test = cv::Mat(frame,roi);  //使用彩圖的ROI去做偵測
	cvtColor(test,test,CV_BGR2BGRA); // 轉成4通道的影像  因為GPU運算只支援四通道的

	if ( type == 1 )  // 若是戶外的情況
	  GPU_MODE->detectMultiScale( cv::gpu::GpuMat::GpuMat( test ), found, 0, cv::Size( 8, 8 ), cv::Size( 0, 0 ), 1.05, 2 );
	  //HOG_PTR.detectMultiScale( image, found, 0, cv::Size( 8, 8 ), cv::Size( 0, 0 ), 1.05, 2 );
	else if ( type == 2 )  // 若是隧道的情況
	   //GPU_MODE->detectMultiScale( cv::gpu::GpuMat::GpuMat( test ), found, 0, cv::Size( 8, 8 ), cv::Size( 0, 0 ), 1.05, 2 );
	   GPU_MODE_TUNNEL->detectMultiScale( cv::gpu::GpuMat::GpuMat( test ), found, 0, cv::Size( 8, 8 ), cv::Size( 0, 0 ), 1.05, 2 );
	else
		system("pause");

	for ( int i = 0 ; i < found.size() ; i++ ) {  // 移除框車子時大包小的情況
		for ( int j = 0 ; j < found.size() ; j++ ) {
			if (j==i) ;   // 不跟自己比較  
			else {  // 跟其他的框框比較  如果有大包小的情況  移除小框框
				cv::Rect r = found[j];	  
	            int h1_j = r.tl().y, w1_j = r.tl().x, h2_j = r.br().y, w2_j = r.br().x;
				cv::Rect r1 = found[i];	 
	            int h1_i = r1.tl().y, w1_i = r1.tl().x, h2_i = r1.br().y, w2_i = r1.br().x;
				if ( h1_j > h1_i && h2_j < h2_i && w1_j > w1_i && w2_j < w2_i ) { // 若j在i裡面  移除i框框
					found.erase (found.begin()+i) ;   // 把外面的框框拿掉
					i = 0 ;
					break ;
				} // if
			} // else
		} // for
	} // for  

	
} // RunDetectionLoop()
//*******************************************************************
void SelectingAndDrowingObject( cv::Mat frame, std::vector<cv::Rect> found ) {
	
  double slope_left = Slope( left_lane_top_point.x, left_lane_top_point.y, left_lane_bot_point.x, left_lane_bot_point.y );
  double offset_left = left_lane_bot_point.y - (slope_left*left_lane_bot_point.x) ;
  double slope_right = Slope(right_lane_top_point.x,right_lane_top_point.y,right_lane_bot_point.x,right_lane_bot_point.y );
  double offset_right = right_lane_bot_point.y - (slope_right*right_lane_bot_point.x) ;

	cv::Mat sample ;
	frame.copyTo(sample) ;
	

	for( int i = 0; i < found.size(); i++ ) {
	  cv::Rect r = found[i];	  
	  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

      bool OutOfRange = true ;
      if ( h1 <= VIDEO_HEIGHT && w1 <= VIDEO_WIDTH &&  h2 <= VIDEO_HEIGHT && w2 <= VIDEO_WIDTH ) 
        OutOfRange = false ;

	  h1 = ( h1 < VIDEO_HEIGHT ) ? h1:VIDEO_HEIGHT-1;
      h2 = ( h2 < VIDEO_HEIGHT ) ? h2:VIDEO_HEIGHT-1;
      w1 = ( w1 < VIDEO_WIDTH ) ? w1:VIDEO_WIDTH-1;
      w2 = ( w2 < VIDEO_WIDTH ) ? w2:VIDEO_WIDTH-1;

	  int mid_height = h2 + 150;  // 找到框框底線的中心點
      int mid_width = (w1+w2)/2 ;
	  
	  /*寫出retraining用的圖片*/
	  
	  if ( Write_Retraining_Image == true ) {
		  if( OutOfRange == false ) {
			cv::Mat retraining = cv::Mat( sample, r ) ;
			cv::resize( retraining, retraining, SAMPLE_SIZE, cv::INTER_LINEAR);
			char str[200];
			if ( Is_TUNNEL == false ) 
				sprintf( str, "%s_%d.JPEG", Output,outputSample_Index );
			else if ( Is_TUNNEL == true )
				sprintf( str, "%s_%d.JPEG", Output_TUNNEL,outputSample_Index );

	
			cv::imwrite( str, retraining );
			outputSample_Index++;
		  } // if
	  } // if
	  
	  /*寫出retraining用的圖片*/	
	// if (control <650) {
    if ( mid_height-(slope_left*mid_width)-offset_left < 0 ) {
      cv::rectangle( frame, r.tl(), r.br(), BLUE, 3 );  // 框左邊汽車
      //cv::putText( frame, "L", cv::Point( r.tl().x, r.tl().y-10 ), CV_FONT_HERSHEY_COMPLEX , 0.7, BLUE, 2, 8, false );
    } // if
    else if(mid_height-(slope_right*mid_width)-offset_right < 0) {
      cv::rectangle( frame, r.tl(), r.br(), RED, 3 );  // 框右邊汽車
      //cv::putText( frame, "R", cv::Point( r.br().x, r.tl().y-10 ), CV_FONT_HERSHEY_COMPLEX , 0.7, RED, 2, 8, false );
    } // else if
    else {
      cv::rectangle( frame, r.tl(), r.br(), ORANGE, 3 );  // 框汽車
      //cv::putText( frame, "M", cv::Point( mid_width, r.tl().y-10 ), CV_FONT_HERSHEY_COMPLEX , 0.7, ORANGE, 2, 8, false );
    } // else
	} // for
	//} // if
    sample.release();
} // SelectingAndDrowingObject()
//*******************************************************************
void Label_Tracking( cv::Mat temp_frame, cv::Mat frame,std::vector<cv::Rect> found ) {  // 給追蹤的車輛做編號

	/*先做刪除 看有沒有追蹤器已經追到將近邊界了*/

	if ( If_Point_Outofframe(particle_1.temp_predict_pt, particle_1.Condens_size ) == true 
		|| (particle_1.counter_end-particle_1.counter_start) > 120 ) // 若是有段時間沒有被偵測框覆蓋到  就關掉
		particle_1.Condens_work = false ;
	if ( If_Point_Outofframe(particle_2.temp_predict_pt, particle_2.Condens_size ) == true 
		|| (particle_2.counter_end-particle_2.counter_start) > 120 ) // 若是有段時間沒有被偵測框覆蓋到  就關掉) 
		particle_2.Condens_work = false ;
	if ( If_Point_Outofframe(particle_3.temp_predict_pt, particle_3.Condens_size ) == true 
		|| (particle_3.counter_end-particle_3.counter_start) > 120) 
		particle_3.Condens_work = false ;
	if ( If_Point_Outofframe(particle_4.temp_predict_pt, particle_4.Condens_size ) == true 
		|| (particle_4.counter_end-particle_4.counter_start) > 120) 
		particle_4.Condens_work = false ;
	if ( If_Point_Outofframe(particle_5.temp_predict_pt, particle_5.Condens_size ) == true 
		|| (particle_5.counter_end-particle_5.counter_start) > 120) 
		particle_5.Condens_work = false ;
    if ( If_Point_Outofframe(particle_6.temp_predict_pt, particle_6.Condens_size ) == true 
		|| (particle_6.counter_end-particle_6.counter_start) > 120) 
		particle_6.Condens_work = false ;
	
	/*先做刪除 看有沒有追蹤器已經追到將近邊界了*/

	/*這邊開始是要把有互相重疊到的追蹤框給濾除掉一個   要濾除掉追蹤框的size比較小的那個
	  因為通常是比較大的追蹤框離我們自己車比較近  追蹤框比較小的表示是被遮蔽到的*/



	
	Delete_tracking_overlap_tracking(); // 刪除多餘的追蹤框

	bool Condens_1_has_been_matched = false ;  // 在一張frame裡面  一個追蹤器  只會match到一個偵測框
	bool Condens_2_has_been_matched = false ;
	bool Condens_3_has_been_matched = false ;
	bool Condens_4_has_been_matched = false ;
	bool Condens_5_has_been_matched = false ;
	bool Condens_6_has_been_matched = false ;


	for( int i = 0 ; i < found.size() ; i++ ) {  // *************************
	  bool create_new_tracking = true ; // 是否要創建新的追蹤框(particle filter)
	  bool has_been_matched = false ;  // 確認這個偵測框有沒有跟追蹤框match過了
      cv::Rect r = found[i];	  
	  int h1 = r.tl().y, w1 = r.tl().x, h2 = r.br().y, w2 = r.br().x;

	  CvPoint mid_found = cvPoint((w1+w2)/2,(h1+h2)/2+150);  //車子偵測框框的中心點那個點

	  if ( particle_1.Condens_work == true && has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_1.Condens_size ) == false 
			  && If_Point_Outofframe(particle_1.temp_predict_pt, particle_1.Condens_size ) == false
			  && CalculateDistance( mid_found, particle_1.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found,1 ) > template_matching_threshold
			  && Condens_1_has_been_matched == false ) {

			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_1, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_1.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_1.detect_result_pt.x = mid_found.x ;
			  particle_1.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_1.detect_result_pt,particle_1.Condens_size,frame,1);
			  particle_1.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_1_has_been_matched = true ;
		  } // if
	  } // if
	  if ( particle_2.Condens_work == true &&  has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_2.Condens_size ) == false 
			  && If_Point_Outofframe(particle_2.temp_predict_pt, particle_2.Condens_size ) == false 
			  && CalculateDistance( mid_found, particle_2.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found,2) > template_matching_threshold
			  && Condens_2_has_been_matched == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_2, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_2.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_2.detect_result_pt.x = mid_found.x ;
			  particle_2.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_2.detect_result_pt,particle_2.Condens_size,frame,2);
			  particle_2.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_2_has_been_matched = true ;
		  } // if
	  } //if 
	  if ( particle_3.Condens_work == true &&  has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_3.Condens_size ) == false 
			  && If_Point_Outofframe(particle_3.temp_predict_pt, particle_3.Condens_size ) == false 
			  && CalculateDistance( mid_found, particle_3.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found,3) > template_matching_threshold
			  && Condens_3_has_been_matched == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_3, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_3.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_3.detect_result_pt.x = mid_found.x ;
			  particle_3.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_3.detect_result_pt,particle_3.Condens_size,frame,3);
			  particle_3.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_3_has_been_matched = true ;
		  } // if
	  } //if 
	  if ( particle_4.Condens_work == true && has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_4.Condens_size ) == false 
			  && If_Point_Outofframe(particle_4.temp_predict_pt, particle_4.Condens_size ) == false 
			  && CalculateDistance( mid_found, particle_4.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found, 4 ) > template_matching_threshold
			  && Condens_4_has_been_matched == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_4, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_4.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_4.detect_result_pt.x = mid_found.x ;
			  particle_4.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_4.detect_result_pt,particle_4.Condens_size,frame,4);
			  particle_4.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_4_has_been_matched = true ;
		  } // if
	  } //if 
	  if ( particle_5.Condens_work == true &&  has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_5.Condens_size ) == false 
			  && If_Point_Outofframe(particle_5.temp_predict_pt, particle_5.Condens_size ) == false
			  && CalculateDistance( mid_found, particle_5.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found, 5 ) > template_matching_threshold
			  && Condens_5_has_been_matched == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_5, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_5.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_5.detect_result_pt.x = mid_found.x ;
		      particle_5.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_5.detect_result_pt,particle_5.Condens_size,frame,5);
			  particle_5.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_5_has_been_matched = true ;
		  } // if
	  } //if
	  if ( particle_6.Condens_work == true &&  has_been_matched == false ) {
		  if ( If_Point_Outofframe(mid_found, particle_6.Condens_size ) == false 
			  && If_Point_Outofframe(particle_6.temp_predict_pt, particle_6.Condens_size ) == false
			  && CalculateDistance( mid_found, particle_6.pre_tracking_position ) < spatial_distance_threshold
			  && detect_compare_tracking( mid_found, 6 ) > template_matching_threshold
			  && Condens_6_has_been_matched == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_6, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_6.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_6.detect_result_pt.x = mid_found.x ;
			  particle_6.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_6.detect_result_pt,particle_6.Condens_size,frame,6);
			  particle_6.detecting_tracking_overlapping = true ;
			  create_new_tracking = false ;
			  has_been_matched = true ;
			  Condens_6_has_been_matched = true ;
		  } // if
	  } //if

	  // ******************************************************************************
	  // ******************************************************************************
	  // ******************************************************************************
	  // 這邊開始是創建新的tracking

	  if ( create_new_tracking == true 
		  && If_tracking_rect_already_exist(mid_found,cvSize(w2-w1,h2-h1),frame) == false ) { 
		                     // 如果需要建立新的追蹤器  看哪一個追蹤器沒再用
		  if ( particle_1.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_1, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_1.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_1.detect_result_pt.x = mid_found.x ;
			  particle_1.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_1.detect_result_pt,particle_1.Condens_size,frame,1);
			  particle_1.Condens_work = true ;
			  particle_1.Create_Condens = true ;
			  create_new_tracking = false ;

		  } // if
		  else if( particle_2.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_2, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_2.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_2.detect_result_pt.x = mid_found.x ;
			  particle_2.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_2.detect_result_pt,particle_2.Condens_size,frame,2);
			  particle_2.Condens_work = true ;
			  particle_2.Create_Condens = true ;
			  create_new_tracking = false ;


		  } // else if
		  else if( particle_3.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_3, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_3.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_3.detect_result_pt.x = mid_found.x ;
			  particle_3.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_3.detect_result_pt,particle_3.Condens_size,frame,3);
			  particle_3.Condens_work = true ;
			  particle_3.Create_Condens = true ;
			  create_new_tracking = false ;


		  } // else if
		  else if( particle_4.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_4, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_4.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_4.detect_result_pt.x = mid_found.x ;
			  particle_4.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_4.detect_result_pt,particle_4.Condens_size,frame,4);
			  particle_4.Condens_work = true ;
			  particle_4.Create_Condens = true ;
			  create_new_tracking = false ;
		  } // else if
		  else if( particle_5.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_5, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_5.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_5.detect_result_pt.x = mid_found.x ;
			  particle_5.detect_result_pt.y = mid_found.y ;
              Calculate_car_color( particle_5.detect_result_pt,particle_5.Condens_size,frame,5);
			  particle_5.Condens_work = true ;
			  particle_5.Create_Condens = true ;
			  create_new_tracking = false ;

		  } // else if
		  else if( particle_6.Condens_work == false ) {
			  /*更新灑點位置*/
			  cvmSet(lowerBound,0,0,w1 );   
		      cvmSet(upperBound,0,0,w2 );  
			  cvmSet(lowerBound,1,0,h1+150 );   
			  cvmSet(upperBound,1,0,h2+150 );  
			  cvmSet(lowerBound,2,0,0.0 );   
			  cvmSet(upperBound,2,0,0.0 );  
			  cvmSet(lowerBound,3,0,0.0 );   
			  cvmSet(upperBound,3,0,0.0 ); 
			  cvConDensInitSampleSet(condens_6, lowerBound, upperBound);  
			  /*更新灑點位置*/

			  particle_6.Condens_size = cvSize( (w2-w1),(h2-h1) ); // 得到偵測框框的大小
			  particle_6.detect_result_pt.x = mid_found.x ;
			  particle_6.detect_result_pt.y = mid_found.y ;
			  Calculate_car_color( particle_6.detect_result_pt,particle_6.Condens_size,frame,6);
			  particle_6.Condens_work = true ;
			  particle_6.Create_Condens = true ;
			  create_new_tracking = false ;

		  } // else if
	  } // if
	} // for

	/* 這邊要來判斷說是不是有追蹤框太久沒有被偵測框給覆蓋了  這邊要來計數*/

	if ( particle_1.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_1.detecting_tracking_overlapping == true || particle_1.Create_Condens == true )
			particle_1.counter_start = particle_1.counter_end = 0 ;
		else particle_1.counter_end++ ;
	} // if
	if ( particle_2.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_2.detecting_tracking_overlapping == true || particle_2.Create_Condens == true )
			particle_2.counter_start = particle_2.counter_end = 0 ;
		else particle_2.counter_end++ ;
	} // if
	if ( particle_3.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_3.detecting_tracking_overlapping == true || particle_3.Create_Condens == true )
			particle_3.counter_start = particle_3.counter_end = 0 ;
		else particle_3.counter_end++ ;
	} // if
	if ( particle_4.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_4.detecting_tracking_overlapping == true || particle_4.Create_Condens == true )
			particle_4.counter_start = particle_4.counter_end = 0 ;
		else particle_4.counter_end++ ;
	} // if
	if ( particle_5.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_5.detecting_tracking_overlapping == true || particle_5.Create_Condens == true )
			particle_5.counter_start = particle_5.counter_end = 0 ;
		else particle_5.counter_end++ ;
	} // if
	if ( particle_6.Condens_work == true ) {  //  若是有被覆蓋到 或是剛創見的 那start跟end都是0   否則end +1
		if ( particle_6.detecting_tracking_overlapping == true || particle_6.Create_Condens == true )
			particle_6.counter_start = particle_6.counter_end = 0 ;
		else particle_6.counter_end++ ;
	} // if

} // Label_Tracking()
//*******************************************************************
double CalculateDistance( CvPoint a , CvPoint b ) {
	return pow(pow((double)(a.x-b.x), 2.0) + pow((double)(a.y-b.y), 2.0),0.5);
} // CalculateDistance()
//*******************************************************************
void Delete_tracking_overlap_tracking() {  
	// 判斷是否有追蹤框彼此互相重疊到   要刪掉小的那一個
	// 判斷是否有追蹤框彼此互相重疊到   要刪掉小的那一個
	// 用很笨的方法寫  總共有六個追蹤器  兩兩去比較  = =
	double overlapRate = 0.6 ;  // 覆蓋率

	if ( particle_1.Condens_work == true && particle_2.Condens_work == true ) {
		int tl_x_A = particle_1.temp_predict_pt.x-(particle_1.Condens_size.width/2);
		int tl_y_A = particle_1.temp_predict_pt.y-(particle_1.Condens_size.height/2);
		int tl_x_B = particle_2.temp_predict_pt.x-(particle_2.Condens_size.width/2);
		int tl_y_B = particle_2.temp_predict_pt.y-(particle_2.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_1.Condens_size.width,particle_1.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_2.Condens_size.width,particle_2.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			if ( A.width < B.width ) particle_1.Condens_work = false ;  // 刪掉size比較小的
			else particle_2.Condens_work = false ;
		} // if
	} // if
	if ( particle_1.Condens_work == true && particle_3.Condens_work == true ) {
		int tl_x_A = particle_1.temp_predict_pt.x-(particle_1.Condens_size.width/2);
		int tl_y_A = particle_1.temp_predict_pt.y-(particle_1.Condens_size.height/2);
		int tl_x_B = particle_3.temp_predict_pt.x-(particle_3.Condens_size.width/2);
		int tl_y_B = particle_3.temp_predict_pt.y-(particle_3.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_1.Condens_size.width,particle_1.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_3.Condens_size.width,particle_3.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "2:  in" << endl ;
			if ( A.width < B.width ) particle_1.Condens_work = false ;  // 刪掉size比較小的
			else particle_3.Condens_work = false ;
		} // if
	} // if
	if ( particle_1.Condens_work == true && particle_4.Condens_work == true ) {
		int tl_x_A = particle_1.temp_predict_pt.x-(particle_1.Condens_size.width/2);
		int tl_y_A = particle_1.temp_predict_pt.y-(particle_1.Condens_size.height/2);
		int tl_x_B = particle_4.temp_predict_pt.x-(particle_4.Condens_size.width/2);
		int tl_y_B = particle_4.temp_predict_pt.y-(particle_4.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_1.Condens_size.width,particle_1.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_4.Condens_size.width,particle_4.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "3:  in" << endl ;
			if ( A.width < B.width ) particle_1.Condens_work = false ;  // 刪掉size比較小的
			else particle_4.Condens_work = false ;
		} // if
	} // if
	if ( particle_1.Condens_work == true && particle_5.Condens_work == true ) {
		int tl_x_A = particle_1.temp_predict_pt.x-(particle_1.Condens_size.width/2);
		int tl_y_A = particle_1.temp_predict_pt.y-(particle_1.Condens_size.height/2);
		int tl_x_B = particle_5.temp_predict_pt.x-(particle_5.Condens_size.width/2);
		int tl_y_B = particle_5.temp_predict_pt.y-(particle_5.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_1.Condens_size.width,particle_1.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_5.Condens_size.width,particle_5.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "4:  in" << endl ;
			if ( A.width < B.width ) particle_1.Condens_work = false ;  // 刪掉size比較小的
			else particle_5.Condens_work = false ;
		} // if
	} // if
	if ( particle_1.Condens_work == true && particle_6.Condens_work == true ) {
		int tl_x_A = particle_1.temp_predict_pt.x-(particle_1.Condens_size.width/2);
		int tl_y_A = particle_1.temp_predict_pt.y-(particle_1.Condens_size.height/2);
		int tl_x_B = particle_6.temp_predict_pt.x-(particle_6.Condens_size.width/2);
		int tl_y_B = particle_6.temp_predict_pt.y-(particle_6.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_1.Condens_size.width,particle_1.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_6.Condens_size.width,particle_6.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "5:  in" << endl ;
			if ( A.width < B.width ) particle_1.Condens_work = false ;  // 刪掉size比較小的
			else particle_6.Condens_work = false ;
		} // if
	} // if
	if ( particle_2.Condens_work == true && particle_3.Condens_work == true ) {
		int tl_x_A = particle_2.temp_predict_pt.x-(particle_2.Condens_size.width/2);
		int tl_y_A = particle_2.temp_predict_pt.y-(particle_2.Condens_size.height/2);
		int tl_x_B = particle_3.temp_predict_pt.x-(particle_3.Condens_size.width/2);
		int tl_y_B = particle_3.temp_predict_pt.y-(particle_3.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_2.Condens_size.width,particle_2.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_3.Condens_size.width,particle_3.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "6:  in" << endl ;
			if ( A.width < B.width ) particle_2.Condens_work = false ;  // 刪掉size比較小的
			else particle_3.Condens_work = false ;
		} // if
	} // if
	if ( particle_2.Condens_work == true && particle_4.Condens_work == true ) {
		int tl_x_A = particle_2.temp_predict_pt.x-(particle_2.Condens_size.width/2);
		int tl_y_A = particle_2.temp_predict_pt.y-(particle_2.Condens_size.height/2);
		int tl_x_B = particle_4.temp_predict_pt.x-(particle_4.Condens_size.width/2);
		int tl_y_B = particle_4.temp_predict_pt.y-(particle_4.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_2.Condens_size.width,particle_2.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_4.Condens_size.width,particle_4.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "7:  in" << endl ;
			if ( A.width < B.width ) particle_2.Condens_work = false ;  // 刪掉size比較小的
			else particle_4.Condens_work = false ;
		} // if
	} // if
	if ( particle_2.Condens_work == true && particle_5.Condens_work == true ) {
		int tl_x_A = particle_2.temp_predict_pt.x-(particle_2.Condens_size.width/2);
		int tl_y_A = particle_2.temp_predict_pt.y-(particle_2.Condens_size.height/2);
		int tl_x_B = particle_5.temp_predict_pt.x-(particle_5.Condens_size.width/2);
		int tl_y_B = particle_5.temp_predict_pt.y-(particle_5.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_2.Condens_size.width,particle_2.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_5.Condens_size.width,particle_5.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "8:  in" << Calculating_overlap_rate( A,B ) <<  endl ;
			//cout << particle_2.Condens_size.width << "  " << particle_5.Condens_size.width << endl ;
			if ( A.width < B.width ) particle_2.Condens_work = false ;  // 刪掉size比較小的
			else particle_5.Condens_work = false ;
		} // if
	} // if
	if ( particle_2.Condens_work == true && particle_6.Condens_work == true ) {
		int tl_x_A = particle_2.temp_predict_pt.x-(particle_2.Condens_size.width/2);
		int tl_y_A = particle_2.temp_predict_pt.y-(particle_2.Condens_size.height/2);
		int tl_x_B = particle_6.temp_predict_pt.x-(particle_6.Condens_size.width/2);
		int tl_y_B = particle_6.temp_predict_pt.y-(particle_6.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_2.Condens_size.width,particle_2.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_6.Condens_size.width,particle_6.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "9:  in" << endl ;
			if ( A.width < B.width ) particle_2.Condens_work = false ;  // 刪掉size比較小的
			else particle_6.Condens_work = false ;
		} // if
	} // if
	if ( particle_3.Condens_work == true && particle_4.Condens_work == true ) {
		int tl_x_A = particle_3.temp_predict_pt.x-(particle_3.Condens_size.width/2);
		int tl_y_A = particle_3.temp_predict_pt.y-(particle_3.Condens_size.height/2);
		int tl_x_B = particle_4.temp_predict_pt.x-(particle_4.Condens_size.width/2);
		int tl_y_B = particle_4.temp_predict_pt.y-(particle_4.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_3.Condens_size.width,particle_3.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_4.Condens_size.width,particle_4.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "10:  in" << endl ;
			if ( A.width < B.width ) particle_3.Condens_work = false ;  // 刪掉size比較小的
			else particle_4.Condens_work = false ;
		} // if
	} // if
	if ( particle_3.Condens_work == true && particle_5.Condens_work == true ) {
		int tl_x_A = particle_3.temp_predict_pt.x-(particle_3.Condens_size.width/2);
		int tl_y_A = particle_3.temp_predict_pt.y-(particle_3.Condens_size.height/2);
		int tl_x_B = particle_5.temp_predict_pt.x-(particle_5.Condens_size.width/2);
		int tl_y_B = particle_5.temp_predict_pt.y-(particle_5.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_3.Condens_size.width,particle_3.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_5.Condens_size.width,particle_5.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "11:  in" << endl ;
			if ( A.width < B.width ) particle_3.Condens_work = false ;  // 刪掉size比較小的
			else particle_5.Condens_work = false ;
		} // if
	} // if
	if ( particle_3.Condens_work == true && particle_6.Condens_work == true ) {
		int tl_x_A = particle_3.temp_predict_pt.x-(particle_3.Condens_size.width/2);
		int tl_y_A = particle_3.temp_predict_pt.y-(particle_3.Condens_size.height/2);
		int tl_x_B = particle_6.temp_predict_pt.x-(particle_6.Condens_size.width/2);
		int tl_y_B = particle_6.temp_predict_pt.y-(particle_6.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_3.Condens_size.width,particle_3.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_6.Condens_size.width,particle_6.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "12:  in" << endl ;
			if ( A.width < B.width ) particle_3.Condens_work = false ;  // 刪掉size比較小的
			else particle_6.Condens_work = false ;
		} // if
	} // if
	if ( particle_4.Condens_work == true && particle_5.Condens_work == true ) {
		int tl_x_A = particle_4.temp_predict_pt.x-(particle_4.Condens_size.width/2);
		int tl_y_A = particle_4.temp_predict_pt.y-(particle_4.Condens_size.height/2);
		int tl_x_B = particle_5.temp_predict_pt.x-(particle_5.Condens_size.width/2);
		int tl_y_B = particle_5.temp_predict_pt.y-(particle_5.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_4.Condens_size.width,particle_4.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_5.Condens_size.width,particle_5.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "13:  in" << endl ;
			if ( A.width < B.width ) particle_4.Condens_work = false ;  // 刪掉size比較小的
			else particle_5.Condens_work = false ;
		} // if
	} // if
	if ( particle_4.Condens_work == true && particle_6.Condens_work == true ) {
		int tl_x_A = particle_4.temp_predict_pt.x-(particle_4.Condens_size.width/2);
		int tl_y_A = particle_4.temp_predict_pt.y-(particle_4.Condens_size.height/2);
		int tl_x_B = particle_6.temp_predict_pt.x-(particle_6.Condens_size.width/2);
		int tl_y_B = particle_6.temp_predict_pt.y-(particle_6.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_4.Condens_size.width,particle_4.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_6.Condens_size.width,particle_6.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "14:  in" << endl ;
			if ( A.width < B.width ) particle_4.Condens_work = false ;  // 刪掉size比較小的
			else particle_6.Condens_work = false ;
		} // if
	} // if
	if ( particle_5.Condens_work == true && particle_6.Condens_work == true ) {
		int tl_x_A = particle_5.temp_predict_pt.x-(particle_5.Condens_size.width/2);
		int tl_y_A = particle_5.temp_predict_pt.y-(particle_5.Condens_size.height/2);
		int tl_x_B = particle_6.temp_predict_pt.x-(particle_6.Condens_size.width/2);
		int tl_y_B = particle_6.temp_predict_pt.y-(particle_6.Condens_size.height/2);
		cv::Rect A = Rect( tl_x_A, tl_y_A,particle_5.Condens_size.width,particle_5.Condens_size.height);
		cv::Rect B = Rect( tl_x_B, tl_y_B,particle_6.Condens_size.width,particle_6.Condens_size.height);
		if ( Calculating_overlap_rate( A,B ) > overlapRate ) { // 覆蓋率大於五成 就要砍掉其中一個
			//cout << "15:  in" << endl ;
			if ( A.width < B.width ) particle_5.Condens_work = false ;  // 刪掉size比較小的
			else particle_6.Condens_work = false ;
		} // if
	} // if

} // If_tracking_rect_exist()
//*******************************************************************
double Calculating_overlap_rate(cv::Rect A, cv::Rect B) {
	// 找出兩個矩形交集的區域  並計算出覆蓋率是多少回傳
    int tl_x = ( A.tl().x > B.tl().x ) ? A.tl().x:B.tl().x;
    int tl_y = ( A.tl().y > B.tl().y ) ? A.tl().y:B.tl().y;
    int br_x = ( A.br().x < B.br().x ) ? A.br().x:B.br().x;
    int br_y = ( A.br().y < B.br().y ) ? A.br().y:B.br().y;
	//cout << tl_x << "  " << tl_y << "  " << br_x << "  " << br_y << endl;
    int area_join = ( br_x - tl_x ) * ( br_y - tl_y );
	//cout << area_join << endl ;
    int area_rect1 = ( A.br().x - A.tl().x ) * ( A.br().y - A.tl().y );
    int area_rect2 = ( B.br().x - B.tl().x ) * ( B.br().y - B.tl().y );
	//cout << ( (double)area_join / ( (double)area_rect1 + (double)area_rect2 - (double)area_join ) ) << endl ;
    return ( (double)area_join / ( (double)area_rect1 + (double)area_rect2 - (double)area_join ) );
} // Calculating_overlap_rate()
//*******************************************************************
bool If_tracking_rect_already_exist( CvPoint detect_pt, CvSize detect_size, cv::Mat frame) {  
	// 判斷準備要創建追蹤框的時候  是不是已經有追蹤框在這了
	if ( particle_1.Condens_work == true ) {
		int tl_x = particle_1.temp_predict_pt.x - particle_1.Condens_size.width/2 ;
		int tl_y = particle_1.temp_predict_pt.y - particle_1.Condens_size.height/2 ;
		
		// 追蹤框的矩形/2 的原因是因為怕有車子太近  但還是要去追蹤  所以判斷式要小一點
		cv::Rect tracking = Rect( tl_x,tl_y,particle_1.Condens_size.width/2,particle_1.Condens_size.height/2 );
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;

		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );

		if ( IsOverlap( tracking, detecting ) == true  ) return true ;
	} // if
	if (  particle_2.Condens_work == true  ) {
		int tl_x = particle_2.temp_predict_pt.x - particle_2.Condens_size.width/2 ;
		int tl_y = particle_2.temp_predict_pt.y - particle_2.Condens_size.height/2 ;
		cv::Rect tracking = Rect( tl_x,tl_y,particle_2.Condens_size.width/2,particle_2.Condens_size.height/2 );

		//cv::rectangle( frame, tracking, cv::Scalar(128,128,128), 3 );	
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;
		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );
		//system( "pause" );
		//cv::rectangle( frame, detecting, cv::Scalar(0,0,0), 3 );
		if ( IsOverlap( tracking, detecting ) == true  ) return true ;
	} // if
	if (  particle_3.Condens_work == true  ) {
		int tl_x = particle_3.temp_predict_pt.x - particle_3.Condens_size.width/2 ;
		int tl_y = particle_3.temp_predict_pt.y - particle_3.Condens_size.height/2 ;
		cv::Rect tracking = Rect( tl_x,tl_y,particle_3.Condens_size.width/2,particle_3.Condens_size.height/2 );
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;
		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );

		if ( IsOverlap( tracking, detecting ) == true  ) return true ;
	} // if
	if (  particle_4.Condens_work == true  ) {
		int tl_x = particle_4.temp_predict_pt.x - particle_4.Condens_size.width/2 ;
		int tl_y = particle_4.temp_predict_pt.y - particle_4.Condens_size.height/2 ;
		cv::Rect tracking = Rect( tl_x,tl_y,particle_4.Condens_size.width/2,particle_4.Condens_size.height/2 );
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;
		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );

		if ( IsOverlap( tracking, detecting ) == true  ) return true ;
	} // if
	if (  particle_5.Condens_work == true  ) {
		int tl_x = particle_5.temp_predict_pt.x - particle_5.Condens_size.width/2 ;
		int tl_y = particle_5.temp_predict_pt.y - particle_5.Condens_size.height/2 ;
		cv::Rect tracking = Rect( tl_x,tl_y,particle_5.Condens_size.width/2,particle_5.Condens_size.height/2 );
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;
		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );

		if ( IsOverlap( tracking, detecting ) == true ) return true ;
	} // if
	if (  particle_6.Condens_work == true  ) {
		int tl_x = particle_6.temp_predict_pt.x - particle_6.Condens_size.width/2 ;
		int tl_y = particle_6.temp_predict_pt.y - particle_6.Condens_size.height/2 ;
		cv::Rect tracking = Rect( tl_x,tl_y,particle_6.Condens_size.width/2,particle_6.Condens_size.height/2 );
		tl_x = detect_pt.x - detect_size.width/2 ;
		tl_y = detect_pt.y - detect_size.height/2 ;
		cv::Rect detecting = Rect( tl_x,tl_y,detect_size.width,detect_size.height );

		if ( IsOverlap( tracking, detecting ) == true ) return true ;
	} // if

	return false ;
} // If_tracking_rect_exist()
//*******************************************************************
bool IsOverlap( cv::Rect rect1, cv::Rect rect2 ) {
	// 判斷兩個矩形是否有重疊
	if ( ( rect1.br().x <= rect2.tl().x || rect2.br().x <= rect1.tl().x ) ||
         ( rect1.br().y <= rect2.tl().y || rect2.br().y <= rect1.tl().y ) )
        return false;
    else
        return true;
} // IsOverlap()
//*******************************************************************
bool If_Point_Outofframe( CvPoint pt, CvSize size ) {
  //if ( pt.x == 491 )  system("pause");
	double boundary_left = 40.0, boundary_right = 600, boundary_top = 150.0, boundary_bot = 450.0 ;
    // 設定particle tracking 的邊界
	if ( (double)pt.x - (double)( size.width/2 ) > boundary_left 
		&& (double)pt.x + (double)( size.width/2 ) < boundary_right 
		&& (double)pt.y - (double)(size.height/2) > boundary_top 
		&& (double)pt.y + (double)( size.height/2 ) < boundary_bot ) {
		return false ;
	} // if
	else{
		return true ;
	} // else
} // If_Point_Outofframe()
//*******************************************************************
void Calculate_car_color( CvPoint position, CvSize size, cv::Mat frame, int which_Car) {
	// 這個function要算出車子的顏色  以及車框外圍一圈"非車"的顏色
	// 只有在label tracking裡面有用到

	for ( int a =0 ; a < 256 ; a++ ) {
		particle_1.B_car_color[a] = particle_1.G_car_color[a] = particle_1.R_car_color[a] = 0 ;   
		particle_2.B_car_color[a] = particle_2.G_car_color[a] = particle_2.R_car_color[a] = 0 ;   
		particle_3.B_car_color[a] = particle_3.G_car_color[a] = particle_3.R_car_color[a] = 0 ;   
		particle_4.B_car_color[a] = particle_4.G_car_color[a] = particle_4.R_car_color[a] = 0 ;   
		particle_5.B_car_color[a] = particle_5.G_car_color[a] = particle_5.R_car_color[a] = 0 ;   
		particle_6.B_car_color[a] = particle_6.G_car_color[a] = particle_6.R_car_color[a] = 0 ;      
	} // for


	int extend_x = cvRound(pow(2,0.5)*size.width) ; // 向外延伸非車的距離  根號2 * w 
	int extend_y = cvRound(pow(2,0.5)*size.height) ; // 向外延伸非車的部分  根號2 * h 
	
	// 邊界不能超過向外延伸的距離

	if ( position.x-(extend_x/2)>0 && position.y-(extend_y/2)>150
		&& position.x+(extend_x/2)<640 && position.y+(extend_y/2)<440) {
		// 在框框外圍額外延伸  算出非車身顏色的周圍顏色
			CvRect ROI_RECT = cvRect(position.x-(extend_x/2),position.y-(extend_y/2),extend_x,extend_y);
			IplImage * image = &IplImage( frame ) ;
			cvSetImageROI(image,ROI_RECT);
			IplImage *img = cvCreateImage( cvSize(ROI_RECT.width,ROI_RECT.height), 8, 3 );
			cvCopy(image,img); 
			cvResetImageROI(image);

			double area = size.width*size.height ; //車子框框內的pixel總數
			double outside_area = (extend_x*extend_y)-area ;  // 外圍的pixel 總數

			CvPoint innerPt_tl = cvPoint((extend_x/2)-(size.width/2),(extend_y/2)-(size.height/2));  // 車子框框內80%的那個矩形左上點 跟右下點
			CvPoint innerPt_br = cvPoint((extend_x/2)+(size.width/2),(extend_y/2)+(size.height/2));

			if ( which_Car == 1 ) {
				// 這邊用來存好車子和非車部分的pixel
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_1.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_1.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_1.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) { 
							// 若是非車部分的顏色
						particle_1.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_1.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_1.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for

			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0

			      particle_1.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_1.G_car_color[a]/=area ;
				  particle_1.R_car_color[a]/=area ;
				  
				  particle_1.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_1.G_not_car_color[a]/=outside_area ;
				  particle_1.R_not_car_color[a]/=outside_area ;

				  if ( particle_1.B_car_color[a] - particle_1.B_not_car_color[a] > 0 ) {
					particle_1.B_confidence[a] = particle_1.B_car_color[a] - particle_1.B_not_car_color[a] ;
				    if ( particle_1.B_confidence[a] > max_B ) 
						max_B = particle_1.B_confidence[a] ;
				  } // if
				  else
					particle_1.B_confidence[a] = 0 ;
				  
				  if ( particle_1.G_car_color[a] - particle_1.G_not_car_color[a] > 0 ) {
					particle_1.G_confidence[a] = particle_1.G_car_color[a] - particle_1.G_not_car_color[a] ;
				    if ( particle_1.G_confidence[a] > max_G )
						max_G = particle_1.G_confidence[a] ;
				  } // if
				  else
					particle_1.G_confidence[a] = 0 ;

				  if ( particle_1.R_car_color[a] - particle_1.R_not_car_color[a] > 0 ) {
					particle_1.R_confidence[a] = particle_1.R_car_color[a] - particle_1.R_not_car_color[a] ;
				    if ( particle_1.R_confidence[a] > max_R )
						max_R = particle_1.R_confidence[a] ;
				  } // if
				  else
					particle_1.R_confidence[a] = 0 ;

			  } // for
			  
 			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bin也按倍率成長  
				  particle_1.B_confidence[a]*=scale_B;
				  particle_1.G_confidence[a]*=scale_G;
				  particle_1.R_confidence[a]*=scale_R;
			  } // for

			} // if
			else if (which_Car == 2) {
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_2.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_2.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_2.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
						/*img->imageData[i*img->widthStep+j] = 0 ;
						img->imageData[i*img->widthStep+j+1] = 255 ;
						img->imageData[i*img->widthStep+j+2] = 255 ;*/
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) {  // 若是非車部分的顏色
						particle_2.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_2.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_2.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
						/*img->imageData[i*img->widthStep+j] = 0 ;
						img->imageData[i*img->widthStep+j+1] = 255 ;
						img->imageData[i*img->widthStep+j+2] = 255 ;*/
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for	

			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0
				  particle_2.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_2.G_car_color[a]/=area ;
				  particle_2.R_car_color[a]/=area ;
				  particle_2.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_2.G_not_car_color[a]/=outside_area ;
				  particle_2.R_not_car_color[a]/=outside_area ;

				  if ( particle_2.B_car_color[a] - particle_2.B_not_car_color[a] >= 0 ) {
					particle_2.B_confidence[a] = particle_2.B_car_color[a] - particle_2.B_not_car_color[a] ;
				    if ( particle_2.B_confidence[a] > max_B )
						max_B = particle_2.B_confidence[a] ;
				  } // if
				  else
					particle_2.B_confidence[a] = 0 ;

				  if ( particle_2.G_car_color[a] - particle_2.G_not_car_color[a] >= 0 ) {
					particle_2.G_confidence[a] = particle_2.G_car_color[a] - particle_2.G_not_car_color[a] ;
				    if ( particle_2.G_car_color[a] > max_G )
						max_G = particle_2.G_car_color[a] ;
				  } // if
				  else
					particle_2.G_confidence[a] = 0 ;

				  if ( particle_2.R_car_color[a] - particle_2.R_not_car_color[a] >= 0 ) {
					particle_2.R_confidence[a] = particle_2.R_car_color[a] - particle_2.R_not_car_color[a] ;
				    if ( particle_2.R_confidence[a] > max_R )
						max_R = particle_2.R_confidence[a] ;
				  } // if
				  else
					particle_2.R_confidence[a] = 0 ;

			  } // for

			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;
			  
			  double scale_BT = 300.0/max_B;
			  double scale_GT = 300.0/max_G;
			  double scale_RT = 300.0/max_R;
			  
			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bin也按倍率成長  
				  //particle_6.B_confidence[a] = particle_2.B_confidence[a] ;
				 // particle_6.G_confidence[a] = particle_2.G_confidence[a] ;
				 // particle_6.R_confidence[a] = particle_2.R_confidence[a] ;
				  particle_2.B_confidence[a]*=scale_B;
				  particle_2.G_confidence[a]*=scale_G;
				  particle_2.R_confidence[a]*=scale_R;
				 // particle_6.B_confidence[a]*=scale_BT;
				 // particle_6.G_confidence[a]*=scale_GT;
				 // particle_6.R_confidence[a]*=scale_RT;
			  } // for

			  
			  /*
			  
				IplImage *HistogramImageB;
				HistogramImageB = cvCreateImage(cvSize(256,300),8,3);
				HistogramImageB->origin=1;
				IplImage *HistogramImageG;
				HistogramImageG = cvCreateImage(cvSize(256,300),8,3);
				HistogramImageG->origin=1;	
				IplImage *HistogramImageR;
				HistogramImageR = cvCreateImage(cvSize(256,300),8,3);
				HistogramImageR->origin=1;


			  for(int i=0;i<256;i++) {
				cvLine(HistogramImageB,cvPoint(i,0),cvPoint(i,particle_6.B_confidence[i]),CV_RGB(0,0,255));
				cvLine(HistogramImageG,cvPoint(i,0),cvPoint(i,particle_6.G_confidence[i]),CV_RGB(0,255,0));
				cvLine(HistogramImageR,cvPoint(i,0),cvPoint(i,particle_6.R_confidence[i]),CV_RGB(255,0,0));
			  }

			  cvNamedWindow("B",1);
              cvShowImage("B",HistogramImageB);
			  cvNamedWindow("G",1);
              cvShowImage("G",HistogramImageG);
			  cvNamedWindow("R",1);
              cvShowImage("R",HistogramImageR);

              //if( waitKey (30) >= 0) ;
			 
			
			 cvReleaseImage(&HistogramImageB);			
			 cvReleaseImage(&HistogramImageG);			
			 cvReleaseImage(&HistogramImageR);


			  
			  cvNamedWindow("test",1);
              cvShowImage("test",img);
			  cvReleaseImage(&img);




			  cout << "IN!!!" << endl;
			  
			  */
			} // else if
			else if (which_Car == 3) {
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_3.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_3.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_3.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) {  // 若是非車部分的顏色
						particle_3.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_3.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_3.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for	
	
			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0
				  particle_3.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_3.G_car_color[a]/=area ;
				  particle_3.R_car_color[a]/=area ;
				  particle_3.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_3.G_not_car_color[a]/=outside_area ;
				  particle_3.R_not_car_color[a]/=outside_area ;

				  if ( particle_3.B_car_color[a] - particle_3.B_not_car_color[a] >= 0 ) {
					particle_3.B_confidence[a] = particle_3.B_car_color[a] - particle_3.B_not_car_color[a] ;
				    if ( particle_3.B_confidence[a] > max_B )
						max_B = particle_3.B_confidence[a] ;
				  } // if
				  else
					particle_3.B_confidence[a] = 0 ;

				  if ( particle_3.G_car_color[a] - particle_3.G_not_car_color[a] >= 0 ) {
					particle_3.G_confidence[a] = particle_3.G_car_color[a] - particle_3.G_not_car_color[a] ;
				    if ( particle_3.G_confidence[a] > max_G )
						max_G = particle_3.G_confidence[a] ;
				  } // if
				  else
					particle_3.G_confidence[a] = 0 ;

				  if ( particle_3.R_car_color[a] - particle_3.R_not_car_color[a] >= 0 ) {
					particle_3.R_confidence[a] = particle_3.R_car_color[a] - particle_3.R_not_car_color[a] ;
				    if ( particle_3.R_confidence[a] > max_R )
						max_R = particle_3.R_confidence[a] ;
				  } // if
				  else
					particle_3.R_confidence[a] = 0 ;

			  } // for

			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bun也按倍率成長  
				  particle_3.B_confidence[a]*=scale_B;
				  particle_3.G_confidence[a]*=scale_G;
				  particle_3.R_confidence[a]*=scale_R;
			  } // for

			} // else if
			else if (which_Car ==4) {
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_4.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_4.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_4.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) {  // 若是非車部分的顏色
						particle_4.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_4.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_4.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for	

			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0
				  particle_4.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_4.G_car_color[a]/=area ;
				  particle_4.R_car_color[a]/=area ;
				  particle_4.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_4.G_not_car_color[a]/=outside_area ;
				  particle_4.R_not_car_color[a]/=outside_area ;

				  if ( particle_4.B_car_color[a] - particle_4.B_not_car_color[a] >= 0 ) {
					particle_4.B_confidence[a] = particle_4.B_car_color[a] - particle_4.B_not_car_color[a] ;
				    if ( particle_4.B_confidence[a] > max_B )
						max_B = particle_4.B_confidence[a] ;
				  } // if
				  else
					particle_4.B_confidence[a] = 0 ;

				  if ( particle_4.G_car_color[a] - particle_4.G_not_car_color[a] >= 0 ) {
					particle_4.G_confidence[a] = particle_4.G_car_color[a] - particle_4.G_not_car_color[a] ;
				    if ( particle_4.G_confidence[a] > max_G )
						max_G = particle_4.G_confidence[a] ;
				  } // if
				  else
					particle_4.G_confidence[a] = 0 ;

				  if ( particle_4.R_car_color[a] - particle_4.R_not_car_color[a] >= 0 ) {
					particle_4.R_confidence[a] = particle_4.R_car_color[a] - particle_4.R_not_car_color[a] ;
				    if ( particle_4.R_confidence[a] > max_R )
						max_R = particle_4.R_confidence[a] ;
				  } // if
				  else
					particle_4.R_confidence[a] = 0 ;

			  } // for


			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bun也按倍率成長  
				  particle_4.B_confidence[a]*=scale_B;
				  particle_4.G_confidence[a]*=scale_G;
				  particle_4.R_confidence[a]*=scale_R;
			  } // for


			} // else if
			else if (which_Car == 5) {
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_5.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_5.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_5.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) {  // 若是非車部分的顏色
						particle_5.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_5.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_5.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for	
	
			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0
				  particle_5.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_5.G_car_color[a]/=area ;
				  particle_5.R_car_color[a]/=area ;
				  particle_5.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_5.G_not_car_color[a]/=outside_area ;
				  particle_5.R_not_car_color[a]/=outside_area ;

				  if ( particle_5.B_car_color[a] - particle_5.B_not_car_color[a] >= 0 ) {
					particle_5.B_confidence[a] = particle_5.B_car_color[a] - particle_5.B_not_car_color[a] ;
				    if ( particle_5.B_confidence[a] > max_B )
						max_B = particle_5.B_confidence[a] ;
				  } // if
				  else
					particle_5.B_confidence[a] = 0 ;

				  if ( particle_5.G_car_color[a] - particle_5.G_not_car_color[a] >= 0 ) {
					particle_5.G_confidence[a] = particle_5.G_car_color[a] - particle_5.G_not_car_color[a] ;
				    if ( particle_5.G_confidence[a] > max_G )
						max_G = particle_5.G_confidence[a] ;
				  } // if
				  else
					particle_5.G_confidence[a] = 0 ;

				  if ( particle_5.R_car_color[a] - particle_5.R_not_car_color[a] >= 0 ) {
					particle_5.R_confidence[a] = particle_5.R_car_color[a] - particle_5.R_not_car_color[a] ;
				    if ( particle_5.R_confidence[a] > max_R )
						max_R = particle_5.R_confidence[a] ;
				  } // if
				  else
					particle_5.R_confidence[a] = 0 ;

			  } // for

			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bin也按倍率成長  
				  particle_5.B_confidence[a]*=scale_B;
				  particle_5.G_confidence[a]*=scale_G;
				  particle_5.R_confidence[a]*=scale_R;
			  } // for

			} // else if
			else if (which_Car == 6) {
			  for ( int i =0 ;i<img->height;i++ ) {
				for( int j=0;j<img->widthStep;j=j+3) {		
					if ( (j/3) > innerPt_tl.x && (j/3) < innerPt_br.x
						&& i > innerPt_tl.y && i < innerPt_br.y ) {  // 若是車身顏色
						particle_6.B_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_6.G_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_6.R_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // if
					else if ( (j/3) > (extend_y/2)+(size.height/2) || i >(extend_x/2)+(size.width/2)
						|| i < (extend_x/2)-(size.width/2) || (j/3) < (extend_y/2)-(size.height/2) ) {  // 若是非車部分的顏色
						particle_6.B_not_car_color[(uchar)img->imageData[i*img->widthStep+j]]++ ;
						particle_6.G_not_car_color[(uchar)img->imageData[i*img->widthStep+j+1]]++ ;
						particle_6.R_not_car_color[(uchar)img->imageData[i*img->widthStep+j+2]]++ ;
					} // else if
					else {;}  // 夾在中間的不理他
				} // for
			  } // for	
		
			  double max_B=0.0,max_G=0.0,max_R=0.0 ; // 要把大的值調整成1  所以要記錄下最大的bin值是多少

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 是車子的扣掉不是車子的  決定出confidence
				  // 相減後如果小於0  設為0
				  particle_6.B_car_color[a]/=area ; // 作正規化  除掉是車子區域的總數 變成0~1
				  particle_6.G_car_color[a]/=area ;
				  particle_6.R_car_color[a]/=area ;
				  particle_6.B_not_car_color[a]/=outside_area ; // 作正規化  除掉飛車部分的總數 變成0~1
				  particle_6.G_not_car_color[a]/=outside_area ;
				  particle_6.R_not_car_color[a]/=outside_area ;

				  if ( particle_6.B_car_color[a] - particle_6.B_not_car_color[a] >= 0 ) {
					particle_6.B_confidence[a] = particle_6.B_car_color[a] - particle_6.B_not_car_color[a] ;
				    if ( particle_6.B_confidence[a] > max_B ) 
						max_B = particle_6.B_confidence[a] ; // 記錄下最大的bin值 要把它變成1
				  } // if
				  else
					particle_6.B_confidence[a] = 0 ;

				  if ( particle_6.G_car_color[a] - particle_6.G_not_car_color[a] >= 0 ) {
					particle_6.G_confidence[a] = particle_6.G_car_color[a] - particle_6.G_not_car_color[a] ;
					if ( particle_6.G_confidence[a] > max_G ) 
						max_G = particle_6.G_confidence[a] ;
				  } // if
				  else
					particle_6.G_confidence[a] = 0 ;

				  if ( particle_6.R_car_color[a] - particle_6.R_not_car_color[a] >= 0 ) {
					particle_6.R_confidence[a] = particle_6.R_car_color[a] - particle_6.R_not_car_color[a] ;
				    if ( particle_6.R_confidence[a] > max_R )
						max_R = particle_6.R_confidence[a] ;
				  } // if
				  else
					particle_6.R_confidence[a] = 0 ;

			  } // for

			  double scale_B = 1.0/max_B;  // 算出把最大的bin 乘上去變成1的倍率
			  double scale_G = 1.0/max_G;
			  double scale_R = 1.0/max_R;

			  for ( int a = 0 ; a < 256 ; a++ ) {  // 把confidence最大bin值設為1 其他bun也按倍率成長  
				  particle_6.B_confidence[a]*=scale_B;
				  particle_6.G_confidence[a]*=scale_G;
				  particle_6.R_confidence[a]*=scale_R;
			  } // for

			} // else if

		} // if

} // Calculate_car_color() 
//*******************************************************************
double Point_Compare_Vehicle_Color( CvPoint pt, CvPoint midfound, int which_car ) {
	// 這個function用來餵給particles confidence  
    // Pt 是 particle 的位置   midfound 是汽車的位置 

	int i = pt.y, j=pt.x ;

	double temp = (double)(pt.x-midfound.x)*(pt.x-midfound.x) + (double)(pt.y-midfound.y)*(pt.y-midfound.y) ;

	if ( which_car == 1 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_1.Condens_size.width*particle_1.Condens_size.width/2)) ;
		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_1.B_confidence[B]+particle_1.G_confidence[G]+particle_1.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // if
	else if ( which_car == 2 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_2.Condens_size.width*particle_2.Condens_size.width/2)) ;

		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_2.B_confidence[B]+particle_2.G_confidence[G]+particle_2.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // else if
	else if ( which_car == 3 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_3.Condens_size.width*particle_3.Condens_size.width/2)) ;

		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_3.B_confidence[B]+particle_3.G_confidence[G]+particle_3.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // else if
	else if ( which_car == 4 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_4.Condens_size.width*particle_4.Condens_size.width/2)) ;

		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_4.B_confidence[B]+particle_4.G_confidence[G]+particle_4.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // else if
	else if ( which_car == 5 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_5.Condens_size.width*particle_5.Condens_size.width/2)) ;

		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_5.B_confidence[B]+particle_5.G_confidence[G]+particle_5.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // else if
	else if ( which_car == 6 ) {
	    // spatial Gaussian 做weight	    
		double weight = (double)exp(-temp/(particle_6.Condens_size.width*particle_6.Condens_size.width/2)) ;

		int B = BLUE_ARRAY[i][j] ;
		int G = GREEN_ARRAY[i][j] ;
		int R = RED_ARRAY[i][j] ;
		// 回傳 RGB 三個相加/3的confidence  
		double color = (particle_6.B_confidence[B]+particle_6.G_confidence[G]+particle_6.R_confidence[R])/3 ;
	    return pow( 100, color*weight );
	} // else if
	  
} // Point_Compare_Vehicle_Color()
//*******************************************************************
double detect_compare_tracking(CvPoint midfound, int which_car){
	// 這個function用來比較偵測框與追蹤框兩個框框的色彩相似度  以達到matching的效果	
	// 這function只有在 Label tracking 被呼叫到
	// 回傳一個兩者之間的色彩相似度  判斷是不是同一台車(偵測框要去match tracking)

	// 準備要在這邊加入GAUSSIAN 進去比較!!!!
 
	// 使用高斯距離當作matching的依據
	int i = midfound.y, j=midfound.x ;
	
	if ( which_car == 1 ) {
		double temp = (double)(midfound.x-particle_1.pre_tracking_position.x)*(midfound.x-particle_1.pre_tracking_position.x) + (double)(midfound.y-particle_1.pre_tracking_position.y)*(midfound.y-particle_1.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_1.Condens_size.width*particle_1.Condens_size.width/2)) ;
		return weight ;
	} // if
	else if ( which_car == 2 ) {
		double temp = (double)(midfound.x-particle_2.pre_tracking_position.x)*(midfound.x-particle_2.pre_tracking_position.x) + (double)(midfound.y-particle_2.pre_tracking_position.y)*(midfound.y-particle_2.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_2.Condens_size.width*particle_2.Condens_size.width/2)) ;
		return weight ;
	} // else if
	else if ( which_car == 3 ) {
		double temp = (double)(midfound.x-particle_3.pre_tracking_position.x)*(midfound.x-particle_3.pre_tracking_position.x) + (double)(midfound.y-particle_3.pre_tracking_position.y)*(midfound.y-particle_3.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_3.Condens_size.width*particle_3.Condens_size.width/2)) ;
		return weight ;
	} // else if
	else if ( which_car == 4 ) {
		double temp = (double)(midfound.x-particle_4.pre_tracking_position.x)*(midfound.x-particle_4.pre_tracking_position.x) + (double)(midfound.y-particle_4.pre_tracking_position.y)*(midfound.y-particle_4.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_4.Condens_size.width*particle_4.Condens_size.width/2)) ;
		return weight ;
	} // else if
	else if ( which_car == 5 ) {
		double temp = (double)(midfound.x-particle_5.pre_tracking_position.x)*(midfound.x-particle_5.pre_tracking_position.x) + (double)(midfound.y-particle_5.pre_tracking_position.y)*(midfound.y-particle_5.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_5.Condens_size.width*particle_5.Condens_size.width/2)) ;
		return weight ;
	} // else if
	else if ( which_car == 6 ) {
		double temp = (double)(midfound.x-particle_6.pre_tracking_position.x)*(midfound.x-particle_6.pre_tracking_position.x) + (double)(midfound.y-particle_6.pre_tracking_position.y)*(midfound.y-particle_6.pre_tracking_position.y) ;
		double weight = (double)exp(-temp/(particle_6.Condens_size.width*particle_6.Condens_size.width/2)) ;
		return weight ;
	} // else if

} // detect_compare_tracking()
//*******************************************************************
void Cvt1dto2d( IplImage *src, int **r, int **g, int **b ) {
	// 這個矩陣用來把一張影像的所有pixel 存成RGB 三個個別的二維陣列
	
  int k = 0;
  if ( src -> nChannels == 1 )
  {
    for ( int i = 0 ; i < src -> height ; i++ )
    {
      if ( ( k + 3 ) % src -> widthStep == 0 ) k += 3;
      for ( int j = 0 ; j < src -> width ; j++, k += 3 )
      {
        b[i][j] = (uchar)src->imageData[k];
        g[i][j] = (uchar)src->imageData[k];
        r[i][j] = (uchar)src->imageData[k];
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

        b[i][j] = (uchar)src->imageData[k];
        g[i][j] = (uchar)src->imageData[k+1];
        r[i][j] = (uchar)src->imageData[k+2];
      } // for
    } // for
  } // else if
  else cout << "Number of channel error!!" << endl;

} // Cvt1dto2d()
//*******************************************************************
void Cvt2dto1d( int **r, int **g, int **b, IplImage *dst ){
	// 這個function用來把三個通到的值assign回一張影像

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
//*******************************************************************