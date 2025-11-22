#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
//video used https://www.youtube.com/watch?v=PhAsdlmY0o8 I downloaded for testing it is 8k video. you can use yt-dlp to download it.
using namespace cv;
namespace fs = std::filesystem;

int main(int, char **) {
    std::string path = "imagesVideo";
    Mat frame;
    
    double fps;
    int time = 0;
    int frames_to_process = 80;

    for (const auto &entry : fs::directory_iterator(path)) {
        fs::remove(entry.path());
    }

    VideoCapture cap("./videoplayback.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    fps = cap.get(CAP_PROP_FPS);
    long start_frame = static_cast<long>(fps * time);
    cap.set(CAP_PROP_POS_FRAMES, start_frame);

    std::vector<int> params = { IMWRITE_PNG_COMPRESSION, 0 };

    for (int i = 0; i < frames_to_process; ++i) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "No more frames or read error at i = " << i << std::endl;
            break;
        }

        std::string filename = "imagesVideo/frame_" + std::to_string(i) + ".png";
        if (!imwrite(filename, frame, params)) {
            std::cerr << "Failed to write " << filename << std::endl;
        }
    }

    return 0;
}