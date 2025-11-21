#pragma once

#include <string>
#include <opencv2/opencv.hpp>
using std::string;
using cv::Mat;

void conv_image(const string& img_path);
void otsu_binarization(const string& img_path);
double otsu_binarization(const Mat& src, Mat& dst);
void extract_object_multi(const string& img_path);