#include <vector>
#include "image_opencv.h"
#include "matrix_calculate.h"


using std::cin;
using std::cout;
using std::endl;
using std::vector;
using cv::imread;
using cv::Scalar;
using cv::imshow;

// 重载函数
void conv(double** A, int m, int n, double** K, int r, int c, const string& out_name)
{
    int out_m = m - r + 1;
    int out_n = n - c + 1;

    double** C = new double* [out_m];
    for (int i = 0; i < out_m; i++)
        C[i] = new double[out_n];

    // 执行卷积
    for (int i = 0; i < out_m; i++) {
        for (int j = 0; j < out_n; j++) {
            C[i][j] = 0;
            for (int u = 0; u < r; u++) {
                for (int v = 0; v < c; v++) {
                    C[i][j] += A[i + u][j + v] * K[u][v];
                }
            }
        }
    }

    // 卷积核求和
    double kernel_sum = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            kernel_sum += K[i][j];

    // 若总和不为0，归一化
    if (kernel_sum != 0) {
        for (int i = 0; i < out_m; i++)
            for (int j = 0; j < out_n; j++)
                C[i][j] /= kernel_sum;
    }

    // 保存为灰度图
    Mat result(out_m, out_n, CV_8U);
    double minVal = C[0][0], maxVal = C[0][0];
    for (int i = 0; i < out_m; i++)
        for (int j = 0; j < out_n; j++) {
            if (C[i][j] < minVal) minVal = C[i][j];
            if (C[i][j] > maxVal) maxVal = C[i][j];
        }

    for (int i = 0; i < out_m; i++)
        for (int j = 0; j < out_n; j++)
            result.at<uchar>(i, j) = static_cast<uchar>(
                255.0 * (C[i][j] - minVal) / (maxVal - minVal + 1e-6));

    imwrite(out_name, result);


    // 释放内存
    for (int i = 0; i < out_m; i++) delete[] C[i];
    delete[] C;
}


/***************************************************************************
  函数名称：conv_image
  功    能：读取一幅灰度图，分别应用6个卷积核并保存结果
  输入参数：img_path —— 图像路径（例如 "demolena.jpg"）
  返 回 值：无
  说    明：
***************************************************************************/
void conv_image(const string& img_path)
{
    Mat src = imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "无法读取图像：" << img_path << endl;
        return;
    }

    int m = src.rows;
    int n = src.cols;

    // 转成二维数组 double**
    double** A = new double* [m];
    for (int i = 0; i < m; i++) {
        A[i] = new double[n];
        for (int j = 0; j < n; j++)
            A[i][j] = src.at<uchar>(i, j);
    }

    // 六个卷积核
    double B1[3][3] = { {1,1,1},{1,1,1},{1,1,1} };      // 平滑
    double B2[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };  // 垂直边缘
    double B3[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };  // 水平边缘
    double B4[3][3] = { {-1,-1,-1},{-1,9,-1},{-1,-1,-1} }; // 锐化
    double B5[3][3] = { {-1,-1,0},{-1,0,1},{0,1,1} };  // 对角线增强
    double B6[3][3] = { {1,2,1},{2,4,2},{1,2,1} };     // 高斯模糊近似
    double (*kernels[6])[3] = { B1,B2,B3,B4,B5,B6 };

    cout << "原图尺寸: " << m << " x " << n << endl;
    cout << "开始执行 6 个卷积核处理..." << endl;

    for (int k = 0; k < 6; k++) {
        // 转成 double**
        double** K = new double* [3];
        for (int i = 0; i < 3; i++) {
            K[i] = new double[3];
            for (int j = 0; j < 3; j++)
                K[i][j] = kernels[k][i][j];
        }

        string out_name = "result_B" + std::to_string(k + 1) + ".jpg";
        conv(A, m, n, K, 3, 3, out_name);
        cout << "卷积核 B" << (k + 1) << " 处理完成 -> " << out_name << endl;

        // 读取并显示结果图
        Mat result = imread(out_name, cv::IMREAD_GRAYSCALE);
        if (!result.empty()) {
            string win_name = "Result B" + std::to_string(k + 1);
            imshow(win_name, result);
        }
        else {
            cout << "无法读取结果图：" << out_name << endl;
        }

        for (int i = 0; i < 3; i++)
            delete[] K[i];
        delete[] K;
    }

    for (int i = 0; i < m; i++)
        delete[] A[i];
    delete[] A;

    cout << "所有卷积结果已保存！" << endl;
    cout << "按任意键关闭所有窗口..." << endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void otsu_binarization(const string& img_path)
{
    Mat src = imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "无法读取图像：" << img_path << endl;
        return;
    }

    int rows = src.rows;
    int cols = src.cols;

    // 统计灰度直方图
    int hist[256] = { 0 };
    for (int i = 0; i < rows; i++) {
        const uchar* row_ptr = src.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            hist[row_ptr[j]]++;
        }
    }

    int total = rows * cols;
    double sum_all = 0.0;
    for (int t = 0; t < 256; t++)
        sum_all += t * hist[t];

    double sumB = 0.0;
    int wB = 0;  // 背景像素数
    int wF = 0;  // 前景像素数

    double max_var = 0.0;
    int threshold_val = 0;

    // 计算类间方差，找到最大值对应的阈值
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;

        wF = total - wB;
        if (wF == 0) break;

        sumB += (double)(t * hist[t]);
        double mB = sumB / wB;
        double mF = (sum_all - sumB) / wF;

        double between_var = (double)wB * (double)wF * (mB - mF) * (mB - mF);

        if (between_var > max_var) {
            max_var = between_var;
            threshold_val = t;
        }
    }

    cout << "OTSU 自动计算阈值为: " << threshold_val << endl;

    // 创建输出图像
    Mat dst = Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        const uchar* src_row = src.ptr<uchar>(i);
        uchar* dst_row = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            dst_row[j] = (src_row[j] > threshold_val) ? 255 : 0;
        }
    }

    // 保存结果
    string out_name = "result_otsu.jpg";
    imwrite(out_name, dst);
    cout << "已保存二值化结果 -> " << out_name << endl;

    // 显示效果
    imshow("Original Image", src);
    imshow("OTSU Result", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// 重载函数
double otsu_binarization(const Mat& src, Mat& dst)
{
    int rows = src.rows;
    int cols = src.cols;

    // 统计灰度直方图
    int hist[256] = { 0 };
    for (int i = 0; i < rows; i++) {
        const uchar* row_ptr = src.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            hist[row_ptr[j]]++;
        }
    }

    int total = rows * cols;
    double sum_all = 0.0;
    for (int t = 0; t < 256; t++)
        sum_all += t * hist[t];

    double sumB = 0.0;
    int wB = 0, wF = 0;
    double max_var = 0.0;
    int threshold_val = 0;

    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        wF = total - wB;
        if (wF == 0) break;

        sumB += t * hist[t];
        double mB = sumB / wB;
        double mF = (sum_all - sumB) / wF;
        double between_var = (double)wB * (double)wF * (mB - mF) * (mB - mF);

        if (between_var > max_var) {
            max_var = between_var;
            threshold_val = t;
        }
    }

    // 根据阈值生成二值图
    dst = Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        const uchar* src_row = src.ptr<uchar>(i);
        uchar* dst_row = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            dst_row[j] = (src_row[j] > threshold_val) ? 255 : 0;
        }
    }

    return (double)threshold_val;
}

void extract_object_multi(const string& img_path)
{
    // 1. 读取灰度图
    Mat src = imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "无法读取图像：" << img_path << endl;
        return;
    }

    //// 2. 使用 OTSU 算法
    char useOTSU;
    cout << "是否使用 OTSU 自动阈值法？(y/n): ";
    while (true)
    {
        cin >> useOTSU;
        if (useOTSU == 'y' || useOTSU == 'Y') {
            threshold(src, src, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        }
        if (useOTSU == 'n' || useOTSU == 'N') {
            break;
        }
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效输入，输入 y/n，请重新输入：";
        }
    }

    // 3. 菜单提示
    cout << "\n=== 形态学操作菜单 ===" << endl;
    cout << "可选择的形态学操作（可多选，用空格分隔）：" << endl;
    cout << "1. 腐蚀 (Erode)" << endl;
    cout << "2. 膨胀 (Dilate)" << endl;
    cout << "3. 开运算 (Open)" << endl;
    cout << "4. 闭运算 (Close)" << endl;
    cout << "5. 形态梯度 (Gradient)" << endl;
    cout << "0. 不进行操作" << endl;
    cout << "请输入选择（例如：1 4 表示先腐蚀再闭运算）: ";

    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    string input;
    getline(cin, input);
    std::stringstream ss(input);

    int ops[10], count = 0, val;
    while (ss >> val)
    {
        if (val >= 0 && val <= 5 && count < 10) ops[count++] = val;
        else cout << "忽略无效输入：" << val << endl;
    }
    if (count == 0) {
        ops[0] = 0;
        count = 1;
    }

    // 4. 执行形态学
    Mat morph = src.clone();
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    for (int i = 0; i < count; i++) {
        switch (ops[i]) {
            case 1: erode(morph, morph, kernel);  cout << "执行腐蚀\n"; break;
            case 2: dilate(morph, morph, kernel); cout << "执行膨胀\n"; break;
            case 3: morphologyEx(morph, morph, cv::MORPH_OPEN, kernel);  cout << "执行开运算\n"; break;
            case 4: morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel); cout << "执行闭运算\n"; break;
            case 5: morphologyEx(morph, morph, cv::MORPH_GRADIENT, kernel); cout << "执行形态梯度\n"; break;
        }
    }

    // 5. 读取彩色图
    Mat src_color = imread(img_path);
    if (src_color.empty()) {
        cout << "无法读取彩色图：" << img_path << endl;
        return;
    }

    // ==============================
    //    下面开始加入你需要的三种展示效果
    // ==============================

    // 1) 原图 + 只保留前景区域（背景黑色）
    Mat masked;
    src_color.copyTo(masked, morph);  // morph = 255 的像素被保留

    // 2) 原图 + 红色轮廓勾勒
    vector<vector<cv::Point>> contours;
    findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    Mat contour_show = src_color.clone();
    drawContours(contour_show, contours, -1, Scalar(0, 0, 255), 2);  // 红色

    // 3) 原图 + 半透明黄色高亮 overlay
    Mat overlay = src_color.clone();
    Mat highlight = src_color.clone();
    for (int y = 0; y < morph.rows; y++) {
        for (int x = 0; x < morph.cols; x++) {
            if (morph.at<uchar>(y, x) == 255) {
                highlight.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 255); // Yellow
            }
        }
    }
    addWeighted(highlight, 0.4, src_color, 0.6, 0, overlay);

    // 6. 显示窗口
    imshow("Original", src_color);
    imshow("Binary / After Morphology", morph);
    imshow("1. Masked Foreground", masked);
    imshow("2. Contour on Original", contour_show);
    imshow("3. Yellow Overlay Highlight", overlay);

    // 7. 保存结果
    imwrite("result_mask_multi.jpg", morph);
    imwrite("result_masked.png", masked);
    imwrite("result_contour.png", contour_show);
    imwrite("result_overlay.png", overlay);

    cout << "\n处理完成！保存为：" << endl;
    cout << " - result_mask_multi.jpg" << endl;
    cout << " - result_masked.png" << endl;
    cout << " - result_contour.png" << endl;
    cout << " - result_overlay.png" << endl;

    cv::waitKey(0);
    cv::destroyAllWindows();
}

