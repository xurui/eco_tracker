#include <iostream>
#include <fstream>
#include <string>

#include "feature_extractor.hpp"
#include "parameters.hpp"
#include "eco_util.hpp"
#include "eco.hpp"
#include "time_log.hpp"

using namespace std;
using namespace cv;
using namespace eco_tracker;
using json = nlohmann::json;

void readGroundTruthFromFile(
    ifstream* groundtruth,
    Rect2f& bboxGroundtruth) {
    float x, y, w, h;
    std::string s;
    getline(*groundtruth, s, '\t');
    x = atof(s.c_str());
    getline(*groundtruth, s, '\t');
    y = atof(s.c_str());
    getline(*groundtruth, s, '\t');
    w = atof(s.c_str());
    getline(*groundtruth, s);
    h = atof(s.c_str());
    cout << "Bounding box:" << x << " " << y << " " << w << " " << h << " " << endl;
    bboxGroundtruth.x = x;
    bboxGroundtruth.y = y;
    bboxGroundtruth.width = w;
    bboxGroundtruth.height = h;
}

int main(int argc, char **argv) {
    // step 1. Read the groundtruth and image
    // step 1.1 Read the groundtruth bbox
    std::string path = "../sequences/Jogging";
    std::string gt_file_name = path + "/groundtruth_rect_2.txt";

    std::ifstream cfg_file_in(path + "/cfg.json");
    json cfg_param;
    cfg_file_in >> cfg_param;
    // Read the groundtruth bbox
    ifstream *groundtruth;
    groundtruth = new ifstream(path + "/groundtruth_rect_2.txt");
    Rect2f bboxGroundtruth;
    readGroundTruthFromFile(groundtruth, bboxGroundtruth);
    Rect2f ecobbox = bboxGroundtruth;

    int f = 1;
    std::string img_file_name = path + "/img/" + cv::format("%04d", f) + ".jpg";
    cv::Mat frame = cv::imread(img_file_name, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat frameDraw;
    frame.copyTo(frameDraw);
    if (!frame.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

    // step 2. eco tracking process
    // step 2.1 eco tracking init process
    timer eco_time_probe;
    ECOTracker ecotracker;
    ecotracker.init(frame, ecobbox, cfg_param);
    double total_cost_time = 0.0;
    int cnt = 0;
    // step 2.2 eco tracking update process
    while (frame.data) {
        frame.copyTo(frameDraw); 
        eco_time_probe.reset();
        bool okeco = ecotracker.update(frame, ecobbox);
        total_cost_time += eco_time_probe.ms_delay();
        cnt++;
        if (okeco) {
            rectangle(frameDraw, ecobbox, Scalar(255, 0, 255), 2, 1); //blue
        } else {
            putText(frameDraw, "ECO tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(255, 0, 255), 2);
        }
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        imshow("EcoTracker", frameDraw);

        std::cout << "--bboxGroundtruth x: " << bboxGroundtruth.x << " y:" << bboxGroundtruth.y
        << " width: " << bboxGroundtruth.width << " height: " << bboxGroundtruth.height << std::endl;
        std::cout << "--ecobbox x: " << ecobbox.x << " y:" << ecobbox.y
        << " width: " << ecobbox.width << " height: " << ecobbox.height << std::endl;
        waitKey(5);
        f++;
        readGroundTruthFromFile(groundtruth, bboxGroundtruth);
        img_file_name = path + "/img/" + cv::format("%04d", f) + ".jpg";
        frame = cv::imread(img_file_name, CV_LOAD_IMAGE_UNCHANGED);
        if (!frame.data) {
            break;
        }
    }
    double avg_cost_time = total_cost_time / cnt;
    std::cout << "---avg cost time: " << avg_cost_time << std::endl;

    return 0;
}
