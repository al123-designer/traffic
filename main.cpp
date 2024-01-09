#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#define PI 3.1415926

using namespace cv;
using namespace std;
std::vector<cv::KeyPoint> keypoints1, keypoints2;
cv::Ptr<cv::ORB> orb = cv::ORB::create();
cv::Mat descriptors1, descriptors2;
string standardPath = "E:\\trafficsign_recognition_opencv-master\\real.jpg";
vector<Rect> boundRect;
struct BGR	// 定义BGR结构体
{
    uchar b;
    uchar g;
    uchar r;
};

struct HSV // 定义HSV结构体
{
    int h;
    double s;
    double v;
};
bool IsEquals(double val1, double val2)
{
    return fabs(val1 - val2) < 0.001;
}
//将RGB格式转换为HSV格式
void BGR2HSV(BGR& bgr, HSV& hsv)
{
    double b, g, r;
    double h, s, v;
    double min, max;
    double delta;

    b = bgr.b / 255.0;
    g = bgr.g / 255.0;
    r = bgr.r / 255.0;

    if (r > g)
    {
        max = MAX(r, b);
        min = MIN(g, b);
    }
    else
    {
        max = MAX(g, b);
        min = MIN(r, b);
    }

    v = max;
    delta = max - min;

    if (IsEquals(max, 0))
    {
        s = 0.0;
    }
    else
    {
        s = delta / max;
    }

    if (max == min)
    {
        h = 0.0;
    }
    else
    {
        if (IsEquals(r, max) && g >= b)
        {
            h = 60 * (g - b) / delta + 0;
        }
        else if (IsEquals(r, max) && g < b)
        {
            h = 60 * (g - b) / delta + 360;
        }
        else if (IsEquals(g, max))
        {
            h = 60 * (b - r) / delta + 120;
        }
        else if (IsEquals(b, max))
        {
            h = 60 * (r - g) / delta + 240;
        }
    }

    hsv.h = (int)(h + 0.5);
    hsv.h = (hsv.h > 359) ? (hsv.h - 360) : hsv.h;
    hsv.h = (hsv.h < 0) ? (hsv.h + 360) : hsv.h;
    hsv.s = s;
    hsv.v = v;
}

void fillHole(const Mat srcBw, Mat &dstBw) {
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

bool isInside(Rect rect1, Rect rect2) {
    Rect t = rect1 & rect2;
    if (rect1.area() > rect2.area()) {
        return false;
    } else {
        if (t.area() != 0)
            return true;
    }
}

vector<Mat> getCircle(Mat srcImg) {
    Mat srcImgCopy;
    srcImg.copyTo(srcImgCopy);

    int width = srcImg.cols;
    int height = srcImg.rows;
    double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
    // 第一步:分割红色颜色色块
    Mat matRgb = Mat::zeros(srcImg.size(), CV_8UC1);
    int x, y;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            BGR bgr;
            bgr.b = srcImg.at<Vec3b>(y, x)[0];
            bgr.g = srcImg.at<Vec3b>(y, x)[1];
            bgr.r = srcImg.at<Vec3b>(y, x)[2];
            HSV hsv;
            BGR2HSV(bgr, hsv);
            if ((hsv.h >= 135 * 2 && hsv.h <= 180 * 2 || hsv.h >= 0 && hsv.h <= 10 * 2) && hsv.s * 255 >= 16
                && hsv.s * 255 <= 255 && hsv.v * 255 >= 46 && hsv.v * 255 <= 255)
            {
                matRgb.at<uchar>(y, x) = 255;
            }


        }
    }
    imshow("step1: to hsv model", matRgb);

    medianBlur(matRgb, matRgb, 3);
    medianBlur(matRgb, matRgb, 5);
    imshow("step2: Median Blur", matRgb);

    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
    Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(matRgb, matRgb, element);
    imshow("step3: Erode", matRgb);
    dilate(matRgb, matRgb, element1);
    imshow("step4: Dilate", matRgb);
    fillHole(matRgb, matRgb);//填充
    imshow("step5: FillHole", matRgb);

    Mat matRgbCopy;
    matRgb.copyTo(matRgbCopy);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Mat> r;
    findContours(matRgb, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> contours_poly(contours.size());
    //vector<Rect> boundRect(contours.size());
    boundRect.resize(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
    }

    Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Rect rect = boundRect[i];
        bool inside = false;
        for (int j = 0; j < contours.size(); j++) {
            Rect t = boundRect[j];
            if (rect == t)
                continue;
            else if (isInside(rect, t)) {
                inside = true;
                break;
            }
        }
        if (inside)
            continue;

        float ratio = (float) rect.width / (float) rect.height;
        float Area = (float) rect.width * (float) rect.height;
        float dConArea = (float) contourArea(contours[i]);
        float dConLen = (float) arcLength(contours[i], 1);
        if (dConArea < 700)
            continue;
        if (ratio > 1.3 || ratio < 0.4)
            continue;

        Scalar color = (0, 0, 255);
        drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

        cv::Mat roi = srcImg(boundRect[i]);
        r.push_back(roi);

        Mat grayImg, dstImg, normImg, scaledImg;
        cvtColor(drawing, grayImg, COLOR_BGR2GRAY);
        cornerHarris(grayImg, dstImg, 2, 3, 0.04);

        normalize(dstImg, normImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(normImg, scaledImg);

        int harrisNum = 0;
        for (int j = 0; j < normImg.rows; j++) {
            for (int i = 0; i < normImg.cols; i++) {
                if ((int) normImg.at<float>(j, i) > 160) {
                    circle(scaledImg, Point(i, j), 4, Scalar(0, 10, 255), 2, 8, 0);
                    harrisNum++;
                }
            }
        }
        if (harrisNum > 33)
            continue;
    }
    return r;
}

float ORB_demo(Mat &img,Mat tar, Mat ori, string name,int i) {
    cvtColor(ori, ori, cv::COLOR_BGR2BGRA);
    //threshold(tar, tar, 0, 255, cv::THRESH_OTSU);

    orb->detect(ori, keypoints1);
    orb->compute(ori, keypoints1, descriptors1);

    orb->detect(tar, keypoints2);
    orb->compute(tar, keypoints2, descriptors2);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher bfmatcher(cv::NORM_HAMMING);
    bfmatcher.match(descriptors1, descriptors2, matches);

    double min_dist = 0, max_dist = 3000;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        //        std::cout << "dist = " << dist << std::endl;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= cv::max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }

    cv::Mat img_match;
    drawMatches(ori, keypoints1, tar, keypoints2, good_matches, img_match);
    float similarity;
    if (good_matches.size() >= 0 && good_matches.size() < 10)
        similarity = 100.0 * static_cast<double>(good_matches.size() + 30) / 80;
    else if (good_matches.size() >= 10 && good_matches.size() < 20)
        similarity = 100.0 * static_cast<double>(good_matches.size() * 2) / 80;
    else if (good_matches.size() >= 20 && good_matches.size() < 30)
        similarity = 100.0 * static_cast<double>(good_matches.size() + 20) / 80;
    else if (good_matches.size() >= 30 && good_matches.size() < 42)
        similarity = 100.0 * static_cast<double>(good_matches.size() + 20) / 80;
    else
        similarity = 100.0 * static_cast<double>(good_matches.size()) / 80;


    std::cout << std::fixed << "相似度" << similarity << std::endl;
    // 绘制矩形框
    Scalar color;
    if (similarity >= 80.0) {
        color = Scalar(128, 0, 128);  // 紫色
        rectangle(img, boundRect[i].tl(), boundRect[i].br(), color, 10, 8, 0);
    }
    else {
        color = Scalar(0, 255, 255);  // 黄色
        rectangle(img, boundRect[i].tl(), boundRect[i].br(), color, 8, 8, 0);
    }

    // 输出信息
    cout << std::fixed << "相似度" << similarity << std::endl;
    cv::imshow(name + ": " + " - Similarity: " + to_string(similarity) + "%", img);
    putText(img, "Similarity: " + to_string(similarity)+"%", Point(boundRect[i].tl().x, boundRect[i].tl().y - 5),
        FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

    cv::imshow(name + ": " + " - Similarity: " + to_string(similarity) + "%", img_match);
    return similarity;
}
int main(int argc, char** argv) {
    cout << "输入图片的绝对地址：" << endl;
    string originPath;
    cin >> originPath;
    Mat origin = imread(originPath, IMREAD_COLOR);
    Mat origins = imread(originPath, IMREAD_COLOR);
    Mat standard = imread(standardPath, IMREAD_COLOR);
    if (standard.empty()) {
        cout << "Could not load sample image due to the wrong path! Please fix the path of sample path in source code line 13!" << endl;
        exit(404);
    }
    if (origin.empty()) {
        cout << "Could not load image due to the wrong path! " << endl;
        exit(404);
    }
    vector<Mat> circled = getCircle(origin);
    imshow("Annotated Image", origins);
    int i = 0;
    int max = 0;
    vector<float> similarities;
    for (auto it : circled) {
       // i++;
        imshow("Detected Sign No." + to_string(i), it);
        cout << "Detected Sign No." + to_string(i) << endl;
        float num = ORB_demo(origin,standard, it, "Sign No." + to_string(i),i);
        similarities.push_back(static_cast<float>(num));
        cout << "Matched feature points number of No." + to_string(i) + " sign = " << num << endl;
        i++;
    }
    for (int i = 0; i < similarities.size(); i++)
        cout << similarities[i]<<"*****" << endl;
    imshow("Annotated Image", origin);
    imwrite("Annotated_Image.jpg", origin);
    waitKey(0);
    return 0;
}
