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
#define DETECTED_IMAGE_DESTINATION_DIRECTORY "./NoNoNo_haar_alt2/detected%03d.bmp"
Scalar Color[] = {
	Scalar(255, 0, 0),
	Scalar(0, 255, 0),
	Scalar(0, 0, 255),
	Scalar(255, 255, 0),
	Scalar(255, 0, 255),
	Scalar(0, 255, 255),
};
int main(void)
{
	/*
	°¢ ÇÁ·¹ÀÓµéÀ» Image·Î ÀúÀå
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
	for (count = 0; count < 600; count++)
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
		// for all faces in a picture
		for (int i = 0; i < faces.size(); i++)
		{
			// rectangles' left-top and right-bottom position
			Point leftTop(faces[i].x, faces[i].y);
			Point rightBottom(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			// draw rectangle
			rectangle(frame, leftTop, rightBottom, Color[i], 4, 8, 0);
			// Numbering
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
			string str = to_string(i + 1);
			char const * number = str.c_str();
			cvPutText(transFrame, number, cvPoint(faces[i].x - 20, faces[i].y), &font, Color[i]);
		}
		string sfilename;
		sfilename = cv::format(DETECTED_IMAGE_DESTINATION_DIRECTORY, count);
		cvSaveImage(sfilename.c_str(), &IplImage(frame));
		cvWaitKey(1);
	}
	return 0;
}