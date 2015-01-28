/*

Face Detection and Labeling

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define HAARCASCADE_FACE_DB "c:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"

#define VIDEO_NAME "C:\\Users\\SSM\\Documents\\OpenCV\\Apink_NoNoNo.avi"
#define FRAMES_NAME "./NoNoNo_haar_alt2/frame%03d.bmp"
#define DETECTED_IMAGE_DESTINATION_DIRECTORY "./NoNoNo_haar_alt2/Associated%03d.bmp"

Scalar Color[] = {
	Scalar(255, 0, 0),
	Scalar(0, 255, 0),
	Scalar(0, 0, 255),
	Scalar(255, 255, 0),
	Scalar(255, 0, 255),
	Scalar(0, 255, 255),

};

typedef struct area{
	int right;
	int top;
	int width;
	int index;
	Point leftTop;
	Point rightBottom;
}AREA;

std::vector<AREA> befArea;
//AREA befArea[10];
int befAreaSize;

int MaxRatio[10];
int index[10] = { 0 };


int main(void)
{
	
	
	/*

	Face Detection

	*/

	// Load the cascades
	CascadeClassifier face_cascade;
	if (!face_cascade.load(HAARCASCADE_FACE_DB))
	{
		printf("Error loading\n");
	}

	int count = 0;

	for (count = 61; count < 63; count++)
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
		// MaxRatio는 befArea에 대해 제일 높은 ratio만을 저장해두는 배열
		int i;
		for (i = 0; i < 10;i++){
			MaxRatio[i] = -10000;
		}
		// curArea array initialize
		//AREA curArea[10];
		std::vector<AREA> curArea;
		int curAreaSize;

		//befAreaSize = (sizeof(befArea) / sizeof(befArea[0]));
		
		printf("%d frame's faces size = %d\n", count,faces.size());

		// for all faces in a picture
		for (int i = 0; i < faces.size(); i++)
		{
			int idx;

			// rectangles' left-top and right-bottom position
			Point leftTop(faces[i].x, faces[i].y);
			Point rightBottom(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

			AREA temp;
			temp.right = faces[i].x;
			temp.top = faces[i].y;
			temp.width = faces[i].width;
			temp.leftTop = leftTop;
			temp.rightBottom = rightBottom;
			temp.index = -1;

			// Association (find index)
				
				// 모든 befArea와 비율 비교하여 0 이상이고 최대 값인 befArea을 찾으면 
				// curArea에 Area 정보와 함께 index를 저장
			int Max = -10000; // 최대 비율
			int j;

			for (j = 0; j < befArea.size(); j++){
				int width,height, ratio;

				if (befArea[j].right>temp.right)
					width = befArea[j].width - (befArea[j].right - temp.right);
				else
					width = temp.width - (temp.right - befArea[j].right);
				if (befArea[j].top>temp.top)
					height = befArea[j].width - (befArea[j].top - temp.top);
				else
					height = temp.width - (temp.top - befArea[j].top);

				
				if (width < 0 || height < 0){
					ratio = -1*abs(befArea[j].top-temp.top)*abs(befArea[j].right-temp.right);
					if (ratio < -10000) ratio = -10001;
				} else{
					ratio = (width*height) * 100 / (temp.width*temp.width);
				}

				if (Max < ratio){
					// 가장 비슷한 area,ratio 찾기
					temp.index = befArea[j].index;
					Max = ratio;
				} 
				 
			}

			// 기존 index의 Max Ratio 비율보다 비율이 작다면 index를 -1로 저장. 더 크다면 갱신			
			if (temp.index > -1 && MaxRatio[temp.index]<Max){
				MaxRatio[temp.index] = Max;
			}
			else{
				temp.index = -1;
			}
			
			curArea.push_back(temp);
		
		}

		for (i = 0; i < 10; i++){
			index[i] = 0;
		}

		// curArea를 뒤에서부터 검사하여, index가 중복되지 않고 설정되도록 함
		for (i = curArea.size()-1; i >= 0; i--){
			if (curArea[i].index > -1){
				if (index[curArea[i].index] == 0)
					index[curArea[i].index] = 1;
				else
					curArea[i].index = -1;
			}
		}

		// curArea를 다 검사하여 index가 -1인 애들은 index[i]가 0인 애들의 index를
		// 순차적으로 넣어준다.
		int zeroIndex = 0;

		for (i = 0; i < curArea.size(); i++){
			if (curArea[i].index == -1){
				while (index[zeroIndex] != 0){
					zeroIndex++;
				}
				curArea[i].index = zeroIndex;
				zeroIndex++;
			}
		
			// draw rectangle
			rectangle(frame, curArea[i].leftTop, curArea[i].rightBottom, Color[curArea[i].index], 4, 8, 0);

			// Numbering

			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

			string str = to_string(curArea[i].index + 1);
			char const * number = str.c_str();

			cvPutText(transFrame, number, cvPoint(faces[i].x - 20, faces[i].y), &font, Color[curArea[i].index]);

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