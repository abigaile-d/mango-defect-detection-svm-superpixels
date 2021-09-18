#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "tinydir.h"

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

static const char* window_name = "Mango Superpixels";
static string base_dir = "C:\\Users\\Abigaile Dionisio\\Pictures\\Mangoes\\";
static string train_input_dir = base_dir + "train\\";
static string train_output_dir = base_dir + "train_spxl\\";
static string train_labeled_dir = base_dir + "train_labeled_spxl\\";
static string test_labeled_dir = base_dir + "test_labeled_spxl\\";
static string predict_input_dir = base_dir + "predict\\";
static string predict_output_dir = base_dir + "predict_spxl\\";
static string predict_labeled_dir = base_dir + "predict_labeled_spxl\\";
static string svm_dir = base_dir + "svm_files\\";

char choice;
Ptr<SuperpixelSEEDS> seeds;

int preprocessImage(Mat &input_image);
void initSuperpixels(Mat &input_image);
void superpixelize(Mat &input_image, string &image_filename, Mat &output_image, Mat &output_labels);
void getHistogram(Mat &spxl_image, Mat &spxl_labels, vector<Mat> &histograms);
void labelSuperpixels(Mat &spxl_image, Mat &spxl_labels, string &image_filename, vector<int> &classification);
void onMouseClick(int event, int x, int y, int flags, void* userdata);
void markPredictions(Mat &spxl_image, Mat &spxl_labels, string &image_filename, vector<int> &predictions, Mat &labeled_image);
float getDefectRatio(Mat &labeled_image);



int main()
{
	/*** DISPLAY MENU ***/

	do {
		cout << "Select activity:" << endl;
		cout << "[1] Generate Training Data" << endl;
		cout << "[2] Generate Prediction Data" << endl;
		cout << "[3] Load Prediction" << endl;
		cout << "[0] Exit" << endl;
		cin >> choice;
		cout << endl << endl;

		if (choice != '0' && choice != '1' && choice != '2' && choice != '3')
			cout << choice << "is not in the menu!" << endl;

	} while (choice != '0' && choice != '1' && choice != '2' && choice != '3');

	if (choice == '0')
		return EXIT_SUCCESS;


	/*** GET FILENAMES OF IMAGES IN THE DIRECTORY ***/

	tinydir_dir dir;
	const char* input_dir;

	// if the user selects 'Generate Training Data', get images from 'train' folder
	if (choice == '1')
		input_dir = train_input_dir.c_str();
	// else if the user selects 'Generate Prediction Data' or 'Load Prediction', get images from 'predict' folder
	else
		input_dir = predict_input_dir.c_str();

	// open the selected folder to check the contents (creates a handler)
	tinydir_open(&dir, input_dir);
	vector<string> input_image_filelist;

	// for each elements inside the folder
	while (dir.has_next) {
		tinydir_file file;

		// get the next element
		tinydir_readfile(&dir, &file);

		// if it is a normal file, add the filename to the filenames list
		if (file.is_reg)
			input_image_filelist.push_back(file.name);

		// fetch the next element
		tinydir_next(&dir);
	}

	// close the handler
	tinydir_close(&dir);

	/*** GET SUPERPIXELS AND EXTRACT FEATURES (HISTOGRAM), SAVE IMAGES AND HISTOGRAM TO FILE ***/

	string svm_output_filename = "dummy.tmp";

	if (choice == '1' || choice == '2') {
		// if the user selected 'Generate Training Data', set output file to 'histogram_train.libsvm.txt'
		if (choice == '1')
			svm_output_filename = "histogram_train.libsvm.txt";
		// if the user selected 'Generate Prediction Data', set output file to 'histogram_predict.libsvm.txt'
		else if (choice == '2')
			svm_output_filename = "histogram_predict.libsvm.txt";
	}

	// set histogram output file name
	ofstream svm_output_file(svm_dir + svm_output_filename);
	
	// set input file name to 'mango_prediction.txt'
	ifstream svm_input_file(svm_dir + "mango_prediction.txt");
	//set prediction output file name
	ofstream prediction_output_file(predict_labeled_dir + "prediction.csv");

	// for all files in the directory
	for (int num = 0; num < input_image_filelist.size(); num++) {
		// load image
		Mat input_image;
		input_image = imread(input_dir + input_image_filelist[num]);

		bool init = false;
		if (!input_image.empty()) {
			
			preprocessImage(input_image);

			Mat spxl_image, spxl_labels;

			// call function [initSuperpixels] to initialize SEEDS superpixel processing
			if (!init){
				initSuperpixels(input_image);
				init = true;
			}

			cout << endl << endl << "Processing " << input_image_filelist[num] << " ..." << endl;

			// call function to segment using superpixels, output: superpixelized image, matrix of superpixel # per pixel, hsv format of the image
			// also save superpixelized image in jpg
			superpixelize(input_image, input_image_filelist[num], spxl_image, spxl_labels);
			int spxl_num = seeds->getNumberOfSuperpixels();

			if (choice == '1' || choice == '2'){
				// compute hue-saturation histogram per superpixel
				vector<Mat> histograms;
				vector<int> classification(spxl_num, 1);
				getHistogram(input_image, spxl_labels, histograms);

				// if for training, display user interface that allows user to label each superpixel, whether good and bad skin
				//if (choice == '1' || choice == '2')	//disabled choice 2 since only used for testing, not needed on actual prediction
				if (choice == '1')
					labelSuperpixels(spxl_image, spxl_labels, input_image_filelist[num], classification);

				// save features to a file in libsvm format
				for (int i = 0; i < histograms.size(); i++) {
					svm_output_file << classification[i] << " ";
					for (int j = 0; j < histograms[i].rows; j++) {
						svm_output_file << (j + 1) << ":" << histograms[i].at<float>(j, 0) << " ";
					}
					svm_output_file << endl;
				}
			}
			else {
				vector<int> predictions;
				int n;
				for (int i = 0; i < spxl_num; i++) {
					svm_input_file >> n;
					predictions.push_back(n);
				}

				// mark the superpixels according to their predicted class
				Mat labeled_spxl;
				markPredictions(spxl_image, spxl_labels, input_image_filelist[num], predictions, labeled_spxl);

				// get the percentage of good and bad class
				float percentage = getDefectRatio(labeled_spxl);
				int label = 1;
				if (percentage > 0.99)
					label = 0;
				prediction_output_file << percentage << "," << label << endl;
			}

		}
		else {
			cout << "Could not open image " << input_image_filelist[num] << " ...\n" << endl;
		}

	}
	svm_output_file.close();
	prediction_output_file.close();

	waitKey(0);

	return 0;
}


int preprocessImage(Mat &input_image) {

	vector<Mat> channels;

	/*** PREPROCESSING ***/

	// resize image
	//resize(input_image, input_image, Size(input_image.cols / 10, input_image.rows / 10), 0, 0, INTER_NEAREST);
	resize(input_image, input_image, Size(600, 450), 0, 0, INTER_NEAREST);

	// adjust brightness and contrast
	double alpha = 1.2; // control contrast
	int beta = 20;  // control brightness

	for (int y = 0; y < input_image.rows; y++) {
		for (int x = 0; x < input_image.cols; x++) {
			for (int c = 0; c < 3; c++) {
				input_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha * (input_image.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}

	// convert image to HSV and split channels
	Mat hsv_image;
	cvtColor(input_image, hsv_image, CV_BGR2HSV);
	split(hsv_image, channels);   

	//equalizeHist(channels[1], channels[1]);

	// detect edges of mango for to separate mango and background
	Mat edges, edges_morph;
	Canny(channels[1], edges, 30, 90);
	// apply morphological closing to close the loop on edge gaps
	morphologyEx(edges, edges_morph, MORPH_CLOSE, Mat::ones(5, 5, edges.type()));

	// find and draw contours
	vector<vector<Point>> contours;
	Mat hierarchy, contour_filled = Mat::zeros(edges.size(), CV_8U), binary = Mat::zeros(edges.size(), CV_8U);
	Mat final_mask;
	findContours(edges_morph, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++){
		drawContours(contour_filled, contours, i, Scalar(255), CV_FILLED);
	}
	
	// apply morphological opening to remove white artifacts i.e. from lines in background
	morphologyEx(contour_filled, final_mask, MORPH_OPEN, Mat::ones(5, 5, contour_filled.type()));

	//copy only 255 values from mask (mango part) to no background image, set image as blue = HSV:100,255,255
	Mat nobg_image = Mat(hsv_image.size(), hsv_image.type(), Scalar(100,255,255));
	hsv_image.copyTo(nobg_image, final_mask);

	//re-split channels to separate new saturation channel
	Mat temp_sat, sat_mask;
	split(nobg_image, channels);
	
	// increase saturation by 50
	compare(channels[1], Scalar(100, 255, 255), sat_mask, CMP_NE);
	channels[1].copyTo(temp_sat);
	//equalizeHist(temp_sat, temp_sat);
	temp_sat += 50;
	temp_sat.copyTo(channels[1], sat_mask);

	//merge and convert back to RGB
	merge(channels, nobg_image);
	cvtColor(nobg_image, input_image, CV_HSV2BGR);

	// check image
	//imshow("Pre-processed", input_image);
	//waitKey(0);

	// return 1 for success operation
	return 1;
}


void initSuperpixels(Mat &input_image){
	// parameters to initialize superpixels
	int prior = 2;
	bool double_step = true;
	int num_spxl = 200;
	int num_levels = 4;
	int num_histogram_bins = 6;

	// initialize superpixel seeds object (from cv::ximgproc::SuperpixelSEEDS)
	// createSuperpixelSEEDS parameters:
	//     input_image.cols : input image width
	//     input_image.rows : input image height
	//     input_image.channels() : number of channels of the input image
	//     num_spxl : number of desired super pixels
	//     num_levels : number of block levels (smaller value = more segments)
	//     prior : shape smoothing value of [0, 5] (larger value = smoother shape)
	//     num_histogram_bins : number of histogram bins
	//     double_step : double step (true = higher accuracy)
	// output:
	//     SuperpixelSEEDS object
	seeds = createSuperpixelSEEDS(input_image.cols, input_image.rows, input_image.channels(), num_spxl, num_levels, prior, num_histogram_bins, double_step);
}


void superpixelize(Mat &input_image, string &image_filename, Mat &spxl_image, Mat &spxl_labels){

	// convert to HSV format (Hue-Saturation-Value)
	Mat hsv_image;
	// cvtColor(input_image, hsv_image, COLOR_BGR2HSV);
	input_image.copyTo(hsv_image);

	// iterate per image
	double t = (double)getTickCount();
	int num_iterations = 5;
	seeds->iterate(hsv_image, num_iterations);

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("SEEDS segmentation took %i ms with %3i superpixels\n", (int)(t * 1000), seeds->getNumberOfSuperpixels());

	// retrieve the segmentation result (superpixel group # of each pixel)
	seeds->getLabels(spxl_labels);

	// get the contours (superpixel borders) for displaying
	Mat contour_mask;
	seeds->getLabelContourMask(contour_mask, false);

	// for display, convert image to HSV and add contours
	// cvtColor(input_image, spxl_image, CV_BGR2HSV);
	input_image.copyTo(spxl_image);
	spxl_image.setTo(Scalar(0, 255, 0), contour_mask);

	// save superpixelized image to a file
	string output_dir;
	if (choice == '1')
		output_dir = train_output_dir;
	else if (choice == '2')
		output_dir = predict_output_dir;

	if (choice == '1' || choice == '2')
		imwrite(output_dir + image_filename, spxl_image);
}


void getHistogram(Mat &input_image, Mat &spxl_labels, vector<Mat> &histograms){
	// quantize to 16 levels
	int histbins = 16;
	float range[] = { 0, 256 };
	const float* ranges[] = { range };

	//for each superpixel
	int num_actual_spxl = seeds->getNumberOfSuperpixels();
	for (int i = 0; i < num_actual_spxl; i++){
		//make a mask with value 255 at the pixels belonging to superpixel #i
		Mat mask;
		compare(spxl_labels, i, mask, CMP_EQ);

		//split channels to hue, saturation and value
		Mat hist_b, hist_g, hist_r;
		vector<Mat> bgr_channels;
		split(input_image, bgr_channels);

		//calculate histogram for RGB channels and normalize to 0-1 range
		calcHist(&bgr_channels[0], 1, 0, mask, hist_b, 1, &histbins, &ranges[0], true, false);
		normalize(hist_b, hist_b, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&bgr_channels[1], 1, 0, mask, hist_g, 1, &histbins, &ranges[0], true, false);
		normalize(hist_g, hist_g, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&bgr_channels[2], 1, 0, mask, hist_r, 1, &histbins, &ranges[0], true, false);
		normalize(hist_r, hist_r, 0, 1, NORM_MINMAX, -1, Mat());

		//save histogram to vector
		Mat hist_final;
		vconcat(hist_b, hist_g, hist_final);
		vconcat(hist_final, hist_r, hist_final);
		histograms.push_back(hist_final);
	}
}


void labelSuperpixels(Mat &spxl_image, Mat &spxl_labels, string &image_filename, vector<int> &classification){
	Mat labeled_image;
	spxl_image.copyTo(labeled_image);

	//display superpixelized image
	imshow(window_name, labeled_image);

	//set mouse callback to return location of mouse clicks in the image
	Point3i click, prevclick = Point3i(0, 0, 0);
	bool mousedrag = false;
	setMouseCallback(window_name, onMouseClick, &click);

	cout << endl << "RIGHT-CLICK on 'bad' mango parts." << endl;
	cout << "LEFT - CLICK to *undo* and tag as 'good' mango parts." << endl;
	cout << "DOUBLE LEFT CLICK to *undo* and tag as background." << endl;
	cout << "When done, press <esc> key." << endl << endl;

	//wait for mouse clicks until <esc> key is pressed
	while (true){

		//if there's a click
		if (prevclick != click){
			//get the superpixel # at the location of the click
			int spx_num = spxl_labels.at<int>(click.y, click.x);

			//save classification based on the click type
			//    left-click (or shift key + drag) = +1 (good skin)
			//    right-click = -1 (bad skin)
			//    double-click (only to undo) = 0 (background), default
			//*** UPDATE: only right-click required, updated to automatically label bg and mango good skin
			classification[spx_num] = click.z;
			cout << "Point:" << click.x << "," << click.y << " | Superpixel #: " << spx_num << "| Class:" << classification[spx_num] << endl;

			//create mask with value of 255 on all the pixels that is part of the superpixel clicked, for display
			Mat currspxl_mask;
			compare(spxl_labels, spx_num, currspxl_mask, CMP_EQ);
			//color it orange
			if (classification[spx_num] == 1)
				labeled_image.setTo(Scalar(0, 120, 255), currspxl_mask);
			//color it black
			else if (classification[spx_num] == -1)
				labeled_image.setTo(Scalar(0, 0, 0), currspxl_mask);
			//remove color
			else if (classification[spx_num] == 0)
				spxl_image.copyTo(labeled_image, currspxl_mask);
			//start of mouse drag
			else if (classification[spx_num] == 2){
				mousedrag = true;
				classification[spx_num] = 1;
			}
			//end of mouse drag
			else if (classification[spx_num] == 3 && mousedrag){
				//for each superpixel inside the area
				//color it with orange and label with +1
				vector<int> spx_nums;
				for (int i = min(prevclick.y, click.y); i < max(prevclick.y, click.y); i++){
					for (int j = min(prevclick.x, click.x); j < max(prevclick.x, click.x); j++){
						int value = spxl_labels.at<int>(i, j);
						if (find(spx_nums.begin(), spx_nums.end(), value) == spx_nums.end()){
							spx_nums.push_back(value);
							compare(spxl_labels, value, currspxl_mask, CMP_EQ);
							labeled_image.setTo(Scalar(0, 120, 255), currspxl_mask);
							//rectangle(labeled_image, Point2f(min(prevclick.x, click.x), min(prevclick.y, click.y)), Point2f(min(prevclick.x, click.x) + rectarea.rows, min(prevclick.y, click.y) + rectarea.cols), Scalar(0, 0, 255));

							classification[value] = 1;
						}
					}
				}

				classification[spx_num] = 1;
			}
			else if (classification[spx_num] == 3){
				classification[spx_num] = 1;
			}

			if (click.z != 2)
				mousedrag = false;

			prevclick = click;

			//display labeled image after the click
			imshow(window_name, labeled_image);
		}

		if (waitKey(33) >= 0){
			//check for blue values and label them as background
			//note:
			//default = +1 (good skin)
			//manual = -1 (bad skin)
			//auto = 0 (background)
			for (int i = 0; i < seeds->getNumberOfSuperpixels(); i++){
				Mat mask, spxl, bg_flag;
				compare(spxl_labels, i, mask, CMP_EQ);
				spxl_image.copyTo(spxl, mask);
				inRange(spxl, Scalar(255, 170, 0), Scalar(255, 170, 0), bg_flag);

				if (sum(bg_flag)[0] > 2550){
					classification[i] = 0;
				}
				else if (classification[i] == 1){
					labeled_image.setTo(Scalar(0, 120, 255), mask);
				}
			}

			//save to a file once labelling is done
			if (choice=='1')
				imwrite(train_labeled_dir + image_filename, labeled_image);
			else if (choice == '2')
				imwrite(test_labeled_dir + image_filename, labeled_image);
			break;
		}

		Sleep(50);
	}
}


void onMouseClick(int event, int x, int y, int flags, void* point){
	Point3i*p = (Point3i*)point;



	//left-click = +1 (good skin)
	if (flags == (EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON)){
		if (event == EVENT_LBUTTONDOWN) {
			p->x = x;
			p->y = y;
			p->z = 2;
		}

	}
	else{

		//left-click = +1 (good skin)
		if (event == EVENT_LBUTTONDOWN) {
			p->x = x;
			p->y = y;
			p->z = 1;
		}
		else if (event == EVENT_LBUTTONUP) {
			p->x = x;
			p->y = y;
			p->z = 3;
		}
		//right-click = -1 (bad skin)
		else if (event == EVENT_RBUTTONDOWN) {
			p->x = x;
			p->y = y;
			p->z = -1;
		}
		//double-click (only to undo) = 0 (background), default
		else if (event == EVENT_LBUTTONDBLCLK) {
			p->x = x;
			p->y = y;
			p->z = 0;
		}
	}

}


void markPredictions(Mat &spxl_image, Mat &spxl_labels, string &image_filename, vector<int> &predictions, Mat &labeled_image){
	spxl_image.copyTo(labeled_image);

	Mat mask;
	// for each superpixel
	for (int i = 0; i < predictions.size(); i++){

		// get the superpixel area in the image
		compare(spxl_labels, i, mask, CMP_EQ);

		// color the area according to the predicted values
		if (predictions[i] == 1)
			labeled_image.setTo(Scalar(0, 120, 255), mask);
		else if (predictions[i] == -1)
			labeled_image.setTo(Scalar(0, 0, 0), mask);
		else
			labeled_image.setTo(Scalar(255, 255, 255), mask);
	}

	imwrite(predict_labeled_dir + image_filename, labeled_image);

}


float getDefectRatio(Mat &labeled_image) {
	float count_bad = 0;
	float count_good = 0;
	float count_bg = 0;
	float percentage;
	//float percentage2;

	for (int y = 0; y < labeled_image.rows; y++) {
		for (int x = 0; x < labeled_image.cols; x++) {
			if (labeled_image.at<Vec3b>(y, x) == Vec3b(0, 120, 255)) {
				count_good++;
			}
			else if (labeled_image.at<Vec3b>(y, x) == Vec3b(0, 0, 0)) {
				count_bad++;
			}
			else if (labeled_image.at<Vec3b>(y, x) == Vec3b(255, 255, 255)) {
				count_bg++;
			}
		}
	}

	percentage = (count_bad / (count_good + count_bad)) * 100;
	//percentage2 = ((count_bad + count_good) / (count_good + count_bad + count_bg)) * 100;

	// display measurements
	printf("Good pixels: %.0f\n", count_good);
	printf("Bad pixels: %.0f\n", count_bad);
	printf("Total pixels occupied: %.0f\n", count_good + count_bad);
	printf("Defect: %.2f%%\n", percentage);

	//printf("Background: %.0f\n", count_bg);
	//printf("Total image: %.0f\n", count_bg + count_good + count_bad);
	//printf("Object: %.2f%%\n", percentage2);

	

	// return 1 for success
	return ceil(percentage*100)/100;
}