// 编译命令 g++ -o libtryPython.so -shared -fPIC test.cpp
//g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` 要编译的源代码 -o 模块名`python3-config --extension-suffix` -I /path/to/python3
#include<iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
using namespace std;
namespace py = pybind11;


vector<int> test(vector<double>xs_008, vector<double>ys_008, vector<vector<int> >radar_pillars_matrix)
{
    vector<int>pillars_row_index;
    for(int i = 0; i < xs_008.size(); ++i)
    {
        int x_index_min = int(xs_008[i] + 0.5) - 24 >= 0? int(xs_008[i] + 0.5) - 24: 0;
        int x_index_max = int(xs_008[i] + 0.5) + 24 < radar_pillars_matrix.size()? int(xs_008[i] + 0.5) + 24: radar_pillars_matrix.size();
        int y_index_min = int(ys_008[i] + 0.5) - 24 >= 0? int(ys_008[i] + 0.5) - 24: 0;
        int y_index_max = int(ys_008[i] + 0.5) + 24 < radar_pillars_matrix.size()? int(ys_008[i] + 0.5) + 24: radar_pillars_matrix.size();
        for(int p =  x_index_min; p < x_index_max; ++p)
        {
            for(int q = y_index_min; q < y_index_max; ++q)
            {
                if(radar_pillars_matrix[p][q] != 0)
                    pillars_row_index.insert(pillars_row_index.end(), radar_pillars_matrix[p][q]);
            }
        }
    }
    return pillars_row_index;
}

void testb()
{
    std::cout << "a" << std::endl;
}

PYBIND11_MODULE(libselect, m)
{
    // 可选，说明这个模块是做什么的
    m.doc() = "pybind11 example plugin";
    //def( "给python调用方法名"， &实际操作的函数， "函数功能说明" ). 其中函数功能说明为可选
    m.def("test", &test, "A function which adds two numbers");
    m.def("testb", &testb, "A function which adds two numbers");
}
