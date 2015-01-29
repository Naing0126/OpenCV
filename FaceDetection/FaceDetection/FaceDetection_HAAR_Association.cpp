/*

Face Detection and Labeling with Association

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define HAARCASCADE_FACE_DB "c:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"

#define VIDEO_NAME "C:\\Users\\SSM\\Documents\\OpenCV\\Apink_NoNoNo.avi"
#define FRAMES_NAME "./NoNoNo_haar_alt2/frame%03d.bmp"
#define DETECTED_IMAGE_DESTINATION_DIRECTORY "./NoNoNo_haar_alt2/associated%03d.bmp"

Scalar Color[] = {
	Scalar(255, 0, 0),
	Scalar(0, 255, 0),
	Scalar(0, 0, 255),
	Scalar(255, 255, 0),
	Scalar(255, 0, 255),
	Scalar(0,255, 255),

};

// struct of each detected area
typedef struct area{
	Point leftTop;
	Point rightBottom;

	int right;
	int top;
	int width;
	
	int index;
}AREA;

std::vector<AREA> befArea;
int befAreaSize;

int MaxRatio[10]; // Max Ratio about each befArea[index]
int index[10] = { 0 }; // to check using of index

int main(void)
{
	/*
	
		각 프레임들을 Image로 저장

	*/

	IplImage *img;

	// Load Video
	CvCapture *capture = cvCaptureFromAVI(VIDEO_NAME);
	

	int count = 0;
	if (capture)
	{
		while (1)
		{
			img = cvQueryFrame(capture);
			if (!img) break;

			// Save Image
			string filename;
			filename = cv::format(FRAMES_NAME, count);
			cvSaveImage(filename.c_str(), img);
			cvWaitKey(1);
			count++;
		}
	}

	cvReleaseCapture(&capture);

	/*

		Face Detection
	
	*/

	// Load the cascades
	CascadeClassifier face_cascade;
	if (!face_cascade.load(HAARCASCADE_FACE_DB))
	{
		printf("Error loading\n");
	}

	for (count = 0; count < 600;count++)
	{
		// Read the image
		Mat frame;
		string filename;
		filename = cv::format(FRAMES_NAME, count);

		frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		IplImage * transFrame = &IplImage(frame);

		// detect and display
		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		// face detection
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// Max Ratio array initialize
		int MaxRatioIdx;
		for (MaxRatioIdx = 0; MaxRatioIdx < 10; MaxRatioIdx++){
			MaxRatio[MaxRatioIdx] = -10000;
		}

		// curArea array initialize
		std::vector<AREA> curArea;
		int curAreaSize;

		// for all faces in a picture
		for (int i = 0; i < faces.size(); i++)
		{
			int idx;

			// rectangles' left-top and right-bottom position
			Point leftTop(faces[i].x, faces[i].y);
			Point rightBottom(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

			AREA temp;
			temp.leftTop = leftTop;
			temp.rightBottom = rightBottom;
			temp.right = faces[i].x;
			temp.top = faces[i].y;
			temp.width = faces[i].width;
			temp.index = -1; // initialize

			/*
			
				Association (find idx)
			
			*/
			// 모든 befArea와 비율 비교하여 0 이상이고 최대 값인 befArea을 찾으면 
			// curArea에 Area 정보와 함께 index를 저장
			int Max = -10000; // 최대 비율
			int j; // befArea index

			for (j = 0; j < befArea.size(); j++){
				int width, height, ratio;

				// 중복되는 가로길이 구하기
				if (befArea[j].right>temp.right)
					width = befArea[j].width - (befArea[j].right - temp.right);
				else
					width = temp.width - (temp.right - befArea[j].right);
				// 중복되는 새로길이 구하기
				if (befArea[j].top>temp.top)
					height = befArea[j].width - (befArea[j].top - temp.top);
				else
					height = temp.width - (temp.top - befArea[j].top);

				// 겹치지 않을 경우
				if (width < 0 || height < 0){
					ratio = -1 * abs(befArea[j].top - temp.top)*abs(befArea[j].right - temp.right);
					if (ratio < -10000) ratio = -10001;
					// 너무 멀리 떨어져 있으면 유사 하지 않다고 봄
				}
				// 겹칠 경우
				else{
					ratio = (width*height) * 100 / (temp.width*temp.width);
				}

				// 가장 비슷한 area 및 그 ratio 저장
				if (Max < ratio){
					temp.index = befArea[j].index;
					Max = ratio;
				}

			}

			// 기존 index의 Max Ratio 비율보다 작다면 index를 -1로 저장. 
			// 더 크다면 index의 Max Ratio를 갱신			
			if (temp.index > -1 && MaxRatio[temp.index]<Max){
				MaxRatio[temp.index] = Max;
			}
			else{
				temp.index = -1;
			}

			curArea.push_back(temp);

		}

		// index initialize
		for (int iidx = 0; iidx < 10; iidx++){
			index[iidx] = 0;
		}

		int cidx;
		// curArea를 뒤에서부터 검사하여, index가 중복되지 않고 설정되도록 함
		for (cidx = curArea.size() - 1; cidx >= 0; cidx--){
			if (curArea[cidx].index > -1){
				if (index[curArea[cidx].index] == 0)
					index[curArea[cidx].index] = 1;
				else
					curArea[cidx].index = -1;
			}
		}

		// curArea를 다 검사하여 index가 -1인 애들은 index[i]가 0인 애들의 index를
		// 순차적으로 넣어준다.
		int zeroIndex = 0;

		for (cidx = 0; cidx < curArea.size(); cidx++){
			if (curArea[cidx].index == -1){
				while (index[zeroIndex] != 0){
					zeroIndex++;
				}
				curArea[cidx].index = zeroIndex;
				zeroIndex++;
			}

			// draw rectangle
			rectangle(frame, curArea[cidx].leftTop, curArea[cidx].rightBottom, Color[curArea[cidx].index], 4, 8, 0);

			// Numbering

			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

			string str = to_string(curArea[cidx].index + 1);
			char const * number = str.c_str();

			cvPutText(transFrame, number, cvPoint(faces[cidx].x - 20, faces[cidx].y), &font, Color[curArea[cidx].index]);

		}

		// befArea = curArea
		befArea = curArea;

		string sfilename;
		sfilename = cv::format(DETECTED_IMAGE_DESTINATION_DIRECTORY, count);

		cvSaveImage(sfilename.c_str(), &IplImage(frame));
		
		cvWaitKey(1);
	}
	
	return 0;
}