

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

void brightAdjust(Mat src, Mat dst, double dContrast, double dBright); //���ȵ��ں���
vector<RotatedRect> armorDetect(vector<RotatedRect> vEllipse); //���װ��
void drawBox(RotatedRect box, Mat img); //���װ��

int main()
{
	
	VideoCapture cap0("armor.mp4");
	Mat frame0;

	Size imgSize;
	RotatedRect s;   //������ת����
	vector<RotatedRect> vEllipse; //������ת���ε����������ڴ洢���ֵ�Ŀ������
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
			//brightAdjust(frame0, rawImg, 1, -120);  //���ϲ���ÿ������ÿ��ͨ����ֵ����ȥ120
			Mat imgHSV;
			cvtColor(frame0, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
			inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
			binary = imgThresholded;
			dilate(binary, grayImage, Mat(), Point(-1, -1), 3);   //ͼ������
			erode(grayImage, rlt, Mat(), Point(-1, -1), 1);  //ͼ��ʴ������
			findContours(rlt, contour, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //�ڶ�ֵͼ����Ѱ������
			for (int i = 0; i < contour.size(); i++)
			{
				if (contour[i].size()> 5)  //�жϵ�ǰ�����Ƿ����10�����ص�
				{
					bFlag = true;   //�������10�������⵽Ŀ������
					//���Ŀ�������Ϊ��Բ������һ����ת���Σ����ġ��Ƕȡ��ߴ磩
					s = fitEllipse(contour[i]);
					vEllipse.push_back(s); //�����ֵ�Ŀ�걣��
				}

			}
			//�����ӳ����������LED������ת���ε�vector���ҳ�װ�׵�λ�ã�����װ����ת���Σ�����vector������
			vRlt = armorDetect(vEllipse);

			for (unsigned int nI = 0; nI < vRlt.size(); nI++) //�ڵ�ǰͼ���б��װ�׵�λ��
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
				//ÿ�����ص�ÿ��ͨ����ֵ���������Ա任
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
	RotatedRect armor; //����װ���������ת����
	int nL, nW;
	double dAngle;
	vRlt.clear();
	if (vEllipse.size() < 2) //�����⵽����ת���θ���С��2����ֱ�ӷ���
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++) //������������ת���εļн�
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			
			dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
			while (dAngle > 180)
				dAngle -= 180;
			//�ж���������ת�����Ƿ���һ��װ�׵�����LED����..�������ɡ�
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE) && abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / T_SIZE_THRE && abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / T_SIZE_THRE)
			{
				armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2; //װ�����ĵ�x���� 
				armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2; //װ�����ĵ�y����
				armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;   //װ��������ת���ε���ת�Ƕ�
				if (180 - dAngle < T_ANGLE_THRE)
					armor.angle += 90;
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2; //װ�׵ĸ߶�
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y)); //װ�׵Ŀ�ȵ�������LED������ת������������ľ���
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
  					vRlt.push_back(armor); //���ҳ���װ�׵���ת���α��浽vector
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

	circle(img, cen, 3, Scalar(0, 0, 255));//imgΪԴͼ��ָ��centerΪ��Բ��Բ������radiusΪԲ�İ뾶
	printf("(%f,%f)\n", cen.x, cen.y);
	box.points(pt); //�����ά���Ӷ��� 
	line(img, pt[0], pt[1], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[1], pt[2], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[2], pt[3], Scalar(0, 0, 255), 2, 8, 0);
	line(img, pt[3], pt[0], Scalar(0, 0, 255), 2, 8, 0);
}
