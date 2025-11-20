#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;

int main(){
    int frame_to_stitch = 30;
    double fps = 30.0;
    Mat frame0 = imread("imagesVideo/frame_0.png");
    Size frame_size(frame0.cols, frame0.rows);
    VideoWriter video_writer("output_video.mp4",
                             VideoWriter::fourcc('m', 'p', '4', 'v'),
                             fps,
                             frame_size);
    for(int i = 0 ;i < frame_to_stitch; ++i){
        std::string filename = "imagesVideo/frame_" + std::to_string(i) + ".png";
        Mat frame = imread(filename);
        if(frame.empty()){
            std::cerr << "Failed to read " << filename << std::endl;
            continue;
        }
        if (frame.size() != frame_size) {
            std::cerr << "Size mismatch at " << filename << std::endl;
            break;
        }
        video_writer.write(frame);
    }
                          
    video_writer.release();
    return 0;  

}