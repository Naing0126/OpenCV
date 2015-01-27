#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cstdio>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
namespace fs = boost::filesystem;
namespace po = boost::program_options;
static CascadeClassifier cascade;
bool parseOptions(int argc, const char** argv,
	string& videoFilename, string& cascadeFilename, string& outputDir,
	double& targetFps);
void detectFaces(Mat& frame, vector<Rect>& faces, const float scale = 1.0);
void drawRect(Mat& frame, int id, Rect& facePosition);
int main(int argc, const char** argv) {
	string videoFilename, cascadeFilename, outputDir;
	double targetFps;
	bool parsed =
		parseOptions(
		argc,
		argv,
		videoFilename,
		cascadeFilename,
		outputDir,
		targetFps);
	if (!parsed)
		return 1;
	VideoCapture cap(videoFilename);
	if (!cap.isOpened())
		return -1;
	if (!cascade.load(cascadeFilename))
		return -1;
	fs::path outputPath(outputDir);
	if (!fs::is_directory(outputPath) && !fs::create_directory(outputPath))
		return -1;
	double sourceFps = cap.get(CV_CAP_PROP_FPS);
	char filename[100];
	fs::path filepath;
	unsigned long pos = 0;
	unsigned long frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame;
	while (pos < frameCount) {
		if (fmod(pos, sourceFps / targetFps) - 1.0 >= 0.0001) {
			++pos;
			continue;
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)pos);
		cap >> frame;
		if (frame.empty())
			break;
		printf("Detecting faces in frame #%lu... ", pos);
		vector<Rect> faces;
		// detect position of faces here
		detectFaces(frame, faces);
		printf("Found %lu faces.\n", faces.size());
		printf("Drawing rectangles on detected faces... ");
		// draw rectangles here
		for (int i = 0, size = faces.size();
			i < size;
			++i)
			drawRect(frame, i, faces[i]);
		printf("done.\n");
		printf("Writing frame #%lu... ", pos);
		sprintf(filename, "%.3lu.jpg", pos);
		filepath = outputPath / fs::path(filename);
		imwrite(filepath.string(), frame);
		printf("done.\n");
		++pos;
	}
	return 0;
}
bool parseOptions(int argc, const char** argv,
	string& videoFilename, string& cascadeFilename, string& outputDir,
	double& targetFps) {
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message.")
			("video-file,v",
			po::value<string>(&videoFilename)->required(),
			"path to input video file.")
			("cascade-classifier,c",
			po::value<string>(&cascadeFilename)->required(),
			"path to cascade classifier file.")
			("output-dir,o",
			po::value<string>(&outputDir)->default_value("output"),
			"path to output directory.")
			("target-fps,f",
			po::value<double>(&targetFps)->default_value(10.0),
			"fps at which video will be scanned.")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			cout << desc << '\n';
			return false;
		}
		po::notify(vm);
	}
	catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return false;
	}
	catch (...) {
		std::cerr << "Unknown error!\n";
		return false;
	}
	return true;
}
void detectFaces(Mat& frame, vector<Rect>& faces, const float scale) {
	const static Scalar lowerBound(0, 133, 77);
	const static Scalar upperBound(255, 173, 127);
	Mat ycrcb;
	Mat mask;
	Mat gray;
	Mat smallImg(cvRound(frame.rows / scale),
		cvRound(frame.cols / scale),
		CV_8UC1);
	cvtColor(frame, ycrcb, CV_BGR2YCrCb);
	inRange(ycrcb, lowerBound, upperBound, mask);
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);
	gray &= mask;
	resize(gray, smallImg, smallImg.size());
	vector<Rect> facesInGray;
	cascade.detectMultiScale(
		gray,
		facesInGray,
		1.1,
		2,
		0 | CASCADE_SCALE_IMAGE,
		Size(30, 30));
	for (vector<Rect>::const_iterator r = facesInGray.begin();
		r != facesInGray.end();
		++r) {
		int sourceX = r->x * scale;
		int sourceY = r->y * scale;
		int sourceWidth = r->width * scale;
		int sourceHeight = r->height * scale;
		Mat croppedMask(mask,
			Range(sourceY, sourceY + sourceHeight),
			Range(sourceX, sourceX + sourceWidth));
		double m = norm(mean(croppedMask));
		if (m / 256 - 0.8 < 0.000001)
			continue;
		Rect new_r(sourceX, sourceY, sourceWidth, sourceHeight);
		faces.push_back(new_r);
	}
}
Scalar colorPreset[] = {
	CV_RGB(0, 255, 0),
	CV_RGB(255, 0, 0),
	CV_RGB(0, 0, 255),
	CV_RGB(255, 255, 0),
	CV_RGB(255, 0, 255),
	CV_RGB(0, 255, 255)
};
void drawRect(Mat& frame, int id, Rect& facePosition) {
	Scalar color = colorPreset[id % (sizeof(colorPreset) / sizeof(Scalar))];
	rectangle(frame,
		cvPoint(facePosition.x, facePosition.y),
		cvPoint(
		facePosition.x + facePosition.width - 1,
		facePosition.y + facePosition.height - 1),
		color);
	putText(frame,
		to_string(id),
		cvPoint(facePosition.x + 4, facePosition.y + 4),
		FONT_HERSHEY_PLAIN,
		1.0,
		color);
}