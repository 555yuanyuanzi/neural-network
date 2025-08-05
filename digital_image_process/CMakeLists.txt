# CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 定义项目名称
project(MyOpenCVProject)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ 核心部分 ------------------
# 查找 OpenCV 包。CMake 会自动搜索！
# REQUIRED 表示如果找不到 OpenCV，就直接报错停止。
find_package(OpenCV REQUIRED)
# ----------------------------------------------

# 将 OpenCV 的头文件目录添加到项目中
# ${OpenCV_INCLUDE_DIRS} 是 find_package 成功后自动创建的变量
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(main main.cpp)



# 将你的可执行文件与 OpenCV 的库链接起来
# ${OpenCV_LIBS} 也是 find_package 成功后自动创建的变量
target_link_libraries(main ${OpenCV_LIBS})


