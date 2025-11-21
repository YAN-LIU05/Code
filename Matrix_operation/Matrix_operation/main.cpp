#include <conio.h>
#include <limits>
#include <opencv2/opencv.hpp>
#include "matrix_calculate.h"
#include "image_opencv.h"


using std::cin;
using std::cout;
using std::endl;
using cv::Mat;
using cv::imread;

void wait_for_enter()
{
    cout << endl
        << "按回车键继续";
    while (_getch() != '\r')
        ;
    cout << endl
        << endl;
}

void menu()
{
    cout << "**************************************************************" << endl;
    cout << "*      1 矩阵加法        2 矩阵数乘        3 矩阵转置        *" << endl;
    cout << "*      4 矩阵乘法        5 Hadmard乘积     6 矩阵卷积        *" << endl;
    cout << "*      7 卷积应用        8 OTSU算法        9 目标提取        *" << endl;
    cout << "*                        0 退出程序                          *" << endl;
    cout << "**************************************************************" << endl;
}

void demo()
{
     /* 对vs+opencv正确配置后方可使用，此处只给出一段读取并显示图像的参考代码，其余功能流程自行设计和查阅文献 */
     Mat image =
         imread("demolena.jpg"); // 图像的灰度值存放在格式为Mat的变量image中
     imshow("Image-original", image);
     cv::waitKey(0);

     //提示：Mat格式可与数组相互转换

    return;
}

int main()
{
    // 定义相关变量

    wait_for_enter();
    while (true) // 注意该循环退出的条件
    {
        system("cls"); // 清屏函数

        menu(); // 调用菜单显示函数，自行补充完成

        // 按要求输入菜单选择项choice
        int choice = 0;
        char ch = '0';
        while (true)
        {
            cout << "选择菜单项<0~9>:";
            cin >> choice;
            if ((choice >= 0 && choice <= 9) && cin.good() == 1)
                break;
            else
            {
                cin.clear();   //重置输入流的状态
                cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                cout << "输入格式错误，请重新输入！" << endl;
            }
        }

        if (choice == 0) // 选择退出
        {
            cout << "\n 确定退出吗?" << endl;
            cin >> ch;
            if (ch == 'y' || ch == 'Y')
                break;
            else
                continue;
        }

        switch (choice)
        {
            // 下述矩阵操作函数自行设计并完成（包括函数参数及返回类型等），若选择加分项，请自行补充
            case 1:
                matriplus();
                break;
            case 2:
                nummulti();
                break;
            case 3:
                matritrans();
                break;
            case 4:
                matrimulti();
                break;
            case 5:
                hadamulti();
                break;
            case 6:
                conv();
                break;
            case 7:
                conv_image("demolena.jpg");
                break;
			case 8:
                otsu_binarization("demolena.jpg");
                break;
			case 9:
                extract_object_multi("snowball.jpg");
				extract_object_multi("polyhedrosis.jpg");
                break;
        }
        wait_for_enter();
    }
    return 0;
}