

#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"

#include "omp.h"
#include "iostream"
#include <opencv2/highgui/highgui_c.h>


using namespace cv;
using namespace std;

#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5

void brightAdjust(Mat src, Mat dst, double dContrast, double dBright); //亮度调节函数
vector<RotatedRect> armorDetect(vector<RotatedRect> vEllipse); //检测装甲
void drawBox(RotatedRect box, Mat img); //标记装甲

int main()
{
	
	VideoCapture cap0("armor.mp4");
	Mat frame0;

	Size imgSize;
	RotatedRect s;   //定义旋转矩形
	vector<RotatedRect> vEllipse; //定以旋转矩形的向量，用于存储发现的目标区域
	vector<RotatedRect> vRlt;
	vector<RotatedRect> vArmor;
	bool bFlag = false;

	vector<vector<Point> > contour;

	cap0 >> frame0;
	imgSize = frame0.size();

	Mat rawImg = Mat(imgSize, CV_8UC3);
	Mat grayImage = Mat(imgSize, CV_8UC1);
	Mat binary = Mat(imgSize, CV_8UC1);
	Mat rlt = Mat(imgSize, CV_8UC1);



	int iLowH = 20;
	//int iLowH = 0;
	int iHighH = 93;

	int iLowS = 132;
	int iHighS = 255;

	int iLowV = 248;
	int iHighV = 255;


	Mat imgThresholded;

	namedWindow("xhhimage");
	while (1)
	{
	
		if (cap0.read(frame0))
		{
			//brightAdjust(frame0, rawImg, 1, -120);  //网上查了每个像素每个通道的值都减去120
			Mat imgHSV;
			cvtColor(frame0, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
			inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
			binary = imgThresholded;
			dilate(binary, grayImage, Mat(), Point(-1, -1), 3);   //图像膨胀
			erode(grayImage, rlt, Mat(), Point(-1, -1), 1);  //图像腐蚀闭运算
			findContours(rlt, contour, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //在二值图像中寻找轮廓
			for (int i = 0; i < contour.size(); i++)
			{
				if (contour[i].size()> 5)  //判断当前轮廓是否大于10个像素点
				{
					bFlag = true;   //如果大于10个，则检测到目标区域
					//拟合目标区域成为椭圆，返回一个旋转矩形（中心、角度、尺寸）
					s = fitEllipse(contour[i]);
					vEllipse.push_back(s); //将发现的目标保存
				}

			}
			//调用子程序，在输入的LED所在旋转矩形的vector中找出装甲的位置，并包装成旋转矩形，存入vector并返回
			vRlt = armorDetect(vEllipse);

			for (unsigned int nI = 0; nI < vRlt.size(); nI++) //在当前图像中标出装甲的位置
				drawBox(vRlt[nI], frame0);
			imshow("Raw", frame0);
			if (waitKey(10) == 27)
			{
				break;
			}
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
		}
		else
		{
			break;
		}
	}
	cap0.release();


	return 0;

}

void brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
	int nVal;
	omp_set_num_threads(8);
#pragma omp parallel for

	for (int nI = 0; nI < src.rows; nI++)
	{
		Vec3b* p1 = src.ptr<Vec3b>(nI);
		Vec3b* p2 = dst.ptr<Vec3b>(nI);
		for (int nJ = 0; nJ < src.cols; nJ++)
		{
			for (int nK = 0; nK < 3; nK++)
			{
				//每个像素的每个通道的值都进行线性变换
				nVal = (int)(dContrast * p1[nJ][nK] + dBright);
				if (nVal < 0)
					nVal = 0;
				if (nVal > 255)
					nVal = 255;
				p2[nJ][nK] = nVal;
			}
		}
	}
}


vector<RotatedRect> armorDetect(vector<RotatedRect> vEllipse)
{
	vector<RotatedRect> vRlt;
	RotatedRect armor; //定义装甲区域的旋转矩形
	int nL, nW;
	double dAngle;
	vRlt.clear();
	if (vEllipse.size() < 2) //如果检测到的旋转矩形个数小于2，则直接返回
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++) //求任意两个旋转矩形的夹角
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			
			dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
			while (dAngle > 180)
				dAngle -= 180;
			//判断这两个旋转矩形是否是一个装甲的两个LED等条..是这样吧。
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE) && abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / T_SIZE_THRE && abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / T_SIZE_THRE)
			{
				armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2; //装甲中心的x坐标 
				armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2; //装甲中心的y坐标
				armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;   //装甲所在旋转矩形的旋转角度
				if (180 - dAngle < T_ANGLE_THRE)
					armor.angle += 90;
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2; //装甲的高度
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y)); //装甲的宽度等于两侧LED所在旋转矩形中心坐标的距离
				if (nL < nW )
				{
					armor.size.height = nL;
					armor.size.width = nW;
				}
  				else
  				{
  					armor.size.height = nW;
  					armor.size.width = nL;
  				}
				if (vEllipse[nI].boundingRect().width<vEllipse[nI].boundingRect().height && vEllipse[nJ].boundingRect().width<vEllipse[nJ].boundingRect().height && armor.size.width < 100 && armor.size.height < 300)
  					vRlt.push_back(armor); //将找出的装甲的旋转矩形保存到vector
			}
		}
	}
	return vRlt;
}

void drawBox(RotatedRect box, Mat img)
{
	Point2f pt[4];
	int i;
	for (i = 0; i < 4; i++)
	{
		pt[i].x = 0;
		pt[i].y = 0;
	}

	auto cen = box.center;

	circle(img, cen, 3, Scalar(0, 0, 255));//img为源图像指针center为画圆的圆心坐标radius为圆的半径
	printf("(%f,%f)\n", cen.x, cen.y);
	box.points(pt); //计算二维盒子顶点 
	line(img, pt[0], pt[1], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[1], pt[2], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[2], pt[3], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[3], pt[0], Scalar(0, 0, 255), 2, 8, 0);
}
