#include <iomanip>
#include <stdlib.h>
#include "matrix_calculate.h"

using std::cin;
using std::cout;
using std::endl;
using std::setw;
using std::fixed;
using std::setprecision;

/***************************************************************************
  函数名称：matriplus
  功    能：矩阵加法函数：读取两个矩阵，进行加法运算并输出结果
  输入参数：
  返 回 值：
  说    明：支持可变大小的矩形矩阵（m x n），使用动态数组
***************************************************************************/
void matriplus() {
    int m, n; // 矩阵行数和列数
    while (true)
    {
        cout << "请输入矩阵的行数 m：";
        cin >> m;
        if ((m > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();   //重置输入流的状态
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }
    while (true)
    {
        cout << "请输入矩阵的列数 n：";
        cin >> n;
        if ((n > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();   //重置输入流的状态
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    // 动态分配第一个矩阵
    double** mat1 = new double* [m];
    for (int i = 0; i < m; i++) {
        mat1[i] = new double[n];
    }
    // 动态分配第二个矩阵
    double** mat2 = new double* [m];
    for (int i = 0; i < m; i++) {
        mat2[i] = new double[n];
    }
    // 动态分配结果矩阵
    double** result = new double* [m];
    for (int i = 0; i < m; i++) {
        result[i] = new double[n];
    }
    // 输入第一个矩阵
    cout << "请输入第一个 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat1[" << i << "][" << j << "] = ";
            cin >> mat1[i][j];
        }
    }
    // 输入第二个矩阵
    cout << "请输入第二个 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat2[" << i << "][" << j << "] = ";
            cin >> mat2[i][j];
        }
    }
    // 计算加法
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    // 输出结果
    cout << "矩阵加法结果：" << endl;
    cout << fixed << setprecision(2); // 设置输出精度
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(8) << result[i][j] << " ";
        }
        cout << endl;
    }
    // 释放内存
    for (int i = 0; i < m; i++) {
        delete[] mat1[i];
        delete[] mat2[i];
        delete[] result[i];
    }
    delete[] mat1;
    delete[] mat2;
    delete[] result;
}

/***************************************************************************
  函数名称：matriplus
  功    能：矩阵数乘函数：读取一个标量和一个矩阵，进行数乘运算并输出结果
  输入参数：
  返 回 值：
  说    明：支持可变大小的矩形矩阵（m x n），使用动态数组
***************************************************************************/
void nummulti() {
    int m, n; // 矩阵行数和列数
    double scalar; // 标量（支持浮点数，兼容整数输入）
    cout << "请输入矩阵的行数 m：";
    cin >> m;
    cout << "请输入矩阵的列数 n：";
    cin >> n;
    if (m <= 0 || n <= 0) {
        cout << "无效的尺寸，请输入正整数。" << endl;
        return;
    }
    cout << "请输入标量值：";
    cin >> scalar;
    // 动态分配矩阵
    double** mat = new double* [m];
    for (int i = 0; i < m; i++) {
        mat[i] = new double[n];
    }
    // 动态分配结果矩阵
    double** result = new double* [m];
    for (int i = 0; i < m; i++) {
        result[i] = new double[n];
    }
    // 输入矩阵
    cout << "请输入 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat[" << i << "][" << j << "] = ";
            cin >> mat[i][j];
        }
    }
    // 计算数乘
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = mat[i][j] * scalar;
        }
    }
    // 输出结果
    cout << "矩阵数乘结果（标量 " << scalar << " 乘以矩阵）：" << endl;
    cout << fixed << setprecision(2); // 设置输出精度
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(8) << result[i][j] << " ";
        }
        cout << endl;
    }
    // 释放内存
    for (int i = 0; i < m; i++) {
        delete[] mat[i];
        delete[] result[i];
    }
    delete[] mat;
    delete[] result;
}

/***************************************************************************
  函数名称：matritrans
  功    能：矩阵转置函数：读取一个矩阵，生成其转置矩阵并输出
  输入参数：
  返 回 值：
  说    明：支持可变大小的矩形矩阵（m x n），使用动态数组
***************************************************************************/
void matritrans() {
    int m, n; // 原矩阵的行列
    while (true)
    {
        cout << "请输入矩阵的行数 m：";
        cin >> m;
        if ((m > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }
    while (true)
    {
        cout << "请输入矩阵的列数 n：";
        cin >> n;
        if ((n > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    // 动态分配原矩阵
    double** mat = new double* [m];
    for (int i = 0; i < m; i++) {
        mat[i] = new double[n];
    }

    // 动态分配转置矩阵（列变行，行变列）
    double** trans = new double* [n];
    for (int i = 0; i < n; i++) {
        trans[i] = new double[m];
    }

    // 输入原矩阵
    cout << "请输入 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat[" << i << "][" << j << "] = ";
            cin >> mat[i][j];
        }
    }

    // 执行转置操作
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            trans[j][i] = mat[i][j];
        }
    }

    // 输出结果
    cout << "矩阵转置结果（" << m << "x" << n << " → " << n << "x" << m << "）：" << endl;
    cout << fixed << setprecision(2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << setw(8) << trans[i][j] << " ";
        }
        cout << endl;
    }

    // 释放内存
    for (int i = 0; i < m; i++) {
        delete[] mat[i];
    }
    delete[] mat;

    for (int i = 0; i < n; i++) {
        delete[] trans[i];
    }
    delete[] trans;
}

/***************************************************************************
  函数名称： matrimulti
  功    能：矩阵乘法函数：读取两个矩阵，进行标准矩阵乘法并输出结果
  输入参数：
  返 回 值：
  说    明：A为 m×n，B为 n×p，结果C为 m×p
***************************************************************************/
void  matrimulti() {
    int m, n, p; // 矩阵A的行列数、矩阵B的列数

    // 输入矩阵A的尺寸
    while (true)
    {
        cout << "请输入矩阵A的行数 m：";
        cin >> m;
        if (m > 0 && cin.good())
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    while (true)
    {
        cout << "请输入矩阵A的列数 n（即矩阵B的行数）：";
        cin >> n;
        if (n > 0 && cin.good())
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    while (true)
    {
        cout << "请输入矩阵B的列数 p：";
        cin >> p;
        if (p > 0 && cin.good())
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    // 动态分配矩阵A (m x n)
    double** A = new double* [m];
    for (int i = 0; i < m; i++) {
        A[i] = new double[n];
    }

    // 动态分配矩阵B (n x p)
    double** B = new double* [n];
    for (int i = 0; i < n; i++) {
        B[i] = new double[p];
    }

    // 动态分配结果矩阵C (m x p)
    double** C = new double* [m];
    for (int i = 0; i < m; i++) {
        C[i] = new double[p];
    }

    // 输入矩阵A
    cout << "请输入矩阵A (" << m << "x" << n << ") 的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "A[" << i << "][" << j << "] = ";
            cin >> A[i][j];
        }
    }

    // 输入矩阵B
    cout << "请输入矩阵B (" << n << "x" << p << ") 的元素：" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            cout << "B[" << i << "][" << j << "] = ";
            cin >> B[i][j];
        }
    }

    // 计算矩阵乘积
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // 输出结果
    cout << "矩阵乘法结果 (A × B)：" << endl;
    cout << fixed << setprecision(2);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            cout << setw(10) << C[i][j] << " ";
        }
        cout << endl;
    }

    // 释放内存
    for (int i = 0; i < m; i++) delete[] A[i];
    delete[] A;

    for (int i = 0; i < n; i++) delete[] B[i];
    delete[] B;

    for (int i = 0; i < m; i++) delete[] C[i];
    delete[] C;
}


void hadamulti()
{
    int m, n; // 矩阵行数和列数
    while (true)
    {
        cout << "请输入矩阵的行数 m：";
        cin >> m;
        if ((m > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();   //重置输入流的状态
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }
    while (true)
    {
        cout << "请输入矩阵的列数 n：";
        cin >> n;
        if ((n > 0 && cin.good() == 1))
            break;
        else
        {
            cin.clear();   //重置输入流的状态
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效的尺寸，请输入正整数。" << endl;
        }
    }

    // 动态分配第一个矩阵
    double** mat1 = new double* [m];
    for (int i = 0; i < m; i++) {
        mat1[i] = new double[n];
    }
    // 动态分配第二个矩阵
    double** mat2 = new double* [m];
    for (int i = 0; i < m; i++) {
        mat2[i] = new double[n];
    }
    // 动态分配结果矩阵
    double** result = new double* [m];
    for (int i = 0; i < m; i++) {
        result[i] = new double[n];
    }
    // 输入第一个矩阵
    cout << "请输入第一个 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat1[" << i << "][" << j << "] = ";
            cin >> mat1[i][j];
        }
    }
    // 输入第二个矩阵
    cout << "请输入第二个 " << m << "x" << n << " 矩阵的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "mat2[" << i << "][" << j << "] = ";
            cin >> mat2[i][j];
        }
    }
    // 计算乘法
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    // 输出结果
    cout << "矩阵Hadamard乘积结果：" << endl;
    cout << fixed << setprecision(2); // 设置输出精度
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(8) << result[i][j] << " ";
        }
        cout << endl;
    }
    // 释放内存
    for (int i = 0; i < m; i++) {
        delete[] mat1[i];
        delete[] mat2[i];
        delete[] result[i];
    }
    delete[] mat1;
    delete[] mat2;
    delete[] result;
}


/***************************************************************************
  函数名称：conv
  功    能：矩阵卷积函数：读取一个矩阵与卷积核，执行二维卷积运算并输出结果
  输入参数：用户可选择是否使用默认参数（kernelsize=3, padding=1, stride=1, dilation=1）
  返 回 值：无
  说    明：支持padding（补零），卷积核逐点滑动
***************************************************************************/
void conv() {
    int m, n; // 输入矩阵尺寸
    int kernel_size = 3;
    int padding = 1;
    int stride = 1;
    int dilation = 1;

    // 参数选择
    char choice;
    cout << "是否使用默认参数 (kernelsize=3, padding=1, stride=1, dilation=1)? (y/n): ";
    while (true) {
        cin >> choice;
        if (choice == 'Y' || choice == 'y' || choice == 'n' || choice == 'N')
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效输入，输入 y/n，请重新输入：";
        }
    }
    if (choice == 'n' || choice == 'N') {
        // 卷积核大小
        while (true) {
            cout << "请输入卷积核大小（奇数，建议3）: ";
            cin >> kernel_size;
            if (cin.good() && kernel_size > 0 && kernel_size % 2 == 1) {
                break;
            }
            else {
                cin.clear();
                cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                cout << "无效输入，请输入正的奇数（例如3、5、7）" << endl;
            }
        }

        // padding
        while (true) {
            cout << "请输入padding（边缘补零层数）: ";
            cin >> padding;
            if (cin.good() && padding >= 0) {
                break;
            }
            else {
                cin.clear();
                cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                cout << "无效输入，请输入非负整数" << endl;
            }
        }

        // stride
        while (true) {
            cout << "请输入stride（步幅）: ";
            cin >> stride;
            if (cin.good() && stride > 0) {
                break;
            }
            else {
                cin.clear();
                cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                cout << "无效输入，请输入正整数（如1、2、3）" << endl;
            }
        }

        // dilation
        while (true) {
            cout << "请输入dilation（膨胀系数）: ";
            cin >> dilation;
            if (cin.good() && dilation > 0) {
                break;
            }
            else {
                cin.clear();
                cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                cout << "无效输入，请输入正整数（如1、2、3）" << endl;
            }
        }
    }


    // 输入矩阵尺寸
    while (true) {
        cout << "请输入输入矩阵的行数 m：";
        cin >> m;
        if (m > 0 && cin.good()) 
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效输入，请输入正整数。" << endl;
        }
    }

    while (true) {
        cout << "请输入输入矩阵的列数 n：";
        cin >> n;
        if (n > 0 && cin.good()) 
            break;
        else {
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "无效输入，请输入正整数。" << endl;
        }
    }

    int effective_kernel = dilation * (kernel_size - 1) + 1;
    int out_m = (m + 2 * padding - effective_kernel) / stride + 1;
    int out_n = (n + 2 * padding - effective_kernel) / stride + 1;

    if (out_m <= 0 || out_n <= 0 || m < kernel_size || n < kernel_size) {
		cout << endl;
        cout << "参数组合不合理：无法在该矩阵上执行卷积！" << endl;
        cout << "请检查：矩阵尺寸(" << m << "x" << n << ")、卷积核=" << kernel_size
            << "、padding=" << padding << "、stride=" << stride
            << "、dilation=" << dilation << endl;
        return;
    }

    // 分配输入矩阵
    double** A = new double* [m];
    for (int i = 0; i < m; i++) {
        A[i] = new double[n];
    }

    // 输入A矩阵
    cout << "请输入输入矩阵A (" << m << "x" << n << ") 的元素：" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << "A[" << i << "][" << j << "] = ";
            cin >> A[i][j];
        }
    }

    // 卷积核矩阵分配
    double** K = new double* [kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        K[i] = new double[kernel_size];
    }

    cout << "请输入卷积核矩阵K (" << kernel_size << "x" << kernel_size << ") 的元素：" << endl;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            cout << "K[" << i << "][" << j << "] = ";
            cin >> K[i][j];
        }
    }


    // 分配输出矩阵
    double** C = new double* [out_m];
    for (int i = 0; i < out_m; i++) {
        C[i] = new double[out_n];
    }

    // 执行卷积运算
    for (int i = 0; i < out_m; i++) {
        for (int j = 0; j < out_n; j++) {
            double sum = 0.0;
            for (int u = 0; u < kernel_size; u++) {
                for (int v = 0; v < kernel_size; v++) {
                    int x = i * stride + u * dilation - padding;
                    int y = j * stride + v * dilation - padding;
                    if (x >= 0 && x < m && y >= 0 && y < n)
                        sum += A[x][y] * K[u][v];
                }
            }
            C[i][j] = sum;
        }
    }

    // 输出结果
    cout << "\n卷积结果矩阵 (" << out_m << "x" << out_n << ")：" << endl;
    cout << fixed << setprecision(2);
    for (int i = 0; i < out_m; i++) {
        for (int j = 0; j < out_n; j++) {
            cout << setw(10) << C[i][j] << " ";
        }
        cout << endl;
    }

    // 释放内存
    for (int i = 0; i < m; i++) delete[] A[i];
    delete[] A;
    for (int i = 0; i < kernel_size; i++) delete[] K[i];
    delete[] K;
    for (int i = 0; i < out_m; i++) delete[] C[i];
    delete[] C;
}



