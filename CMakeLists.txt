cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CRTREES LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_library(CRTrees STATIC CRTrees.cu CRTrees.hpp)
target_compile_features(CRTrees PUBLIC cxx_std_11)
target_compile_options(CRTrees PUBLIC -O3)

add_library(CRTREES STATIC segment.cu segment.hpp imutil.cu util.h)
target_compile_features(CRTREES PUBLIC cxx_std_11)
target_compile_options(CRTREES PUBLIC -O3)
target_link_libraries(CRTREES PUBLIC ${OpenCV_LIBS} CRTrees)

add_executable(test_CRTrees test_CRTrees.cu)
target_link_libraries(test_CRTrees CRTrees)

add_executable(CRTREES_img main_img.cu)
target_link_libraries(CRTREES_img PUBLIC ${OpenCV_LIBS} CRTREES)

add_executable(CRTREES_video main_video.cu)
target_link_libraries(CRTREES_video PUBLIC ${OpenCV_LIBS} CRTREES)

add_subdirectory(lib_eval)
include_directories(./lib_eval/)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

add_executable(CRTREES_img_seq main_img_seq.cu)
target_link_libraries(CRTREES_img_seq eval ${Boost_LIBRARIES} ${OpenCV_LIBS} CRTREES)
