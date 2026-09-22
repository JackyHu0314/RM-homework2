#include "vision_utils.hpp"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const std::string inputPath = argc > 1 ? argv[1] : "resources/test_image.jpg";
    const std::string outputDir = argc > 2 ? argv[2] : "result/task1_images";

    try {
        vision::ensureDirectory(outputDir);
        const cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Cannot read image: " << inputPath << '\n';
            return 1;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        vision::saveImage(outputDir + "/gray.png", gray);

        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsv, hsvChannels);
        vision::saveImage(outputDir + "/hsv_h.png", hsvChannels[0]);
        vision::saveImage(outputDir + "/hsv_s.png", hsvChannels[1]);
        vision::saveImage(outputDir + "/hsv_v.png", hsvChannels[2]);

        cv::Mat meanFiltered, gaussianFiltered, medianFiltered;
        cv::blur(image, meanFiltered, cv::Size(5, 5));
        cv::GaussianBlur(image, gaussianFiltered, cv::Size(5, 5), 1.5);
        cv::medianBlur(image, medianFiltered, 5);
        vision::saveImage(outputDir + "/mean_filter.png", meanFiltered);
        vision::saveImage(outputDir + "/gaussian_filter.png", gaussianFiltered);
        vision::saveImage(outputDir + "/median_filter.png", medianFiltered);

        cv::Mat redLow, redHigh, redMask;
        cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(12, 255, 255), redLow);
        cv::inRange(hsv, cv::Scalar(165, 70, 50), cv::Scalar(179, 255, 255), redHigh);
        cv::bitwise_or(redLow, redHigh, redMask);
        vision::saveImage(outputDir + "/red_mask.png", redMask);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat eroded, dilated, opened, closed;
        cv::erode(redMask, eroded, kernel);
        cv::dilate(redMask, dilated, kernel);
        cv::morphologyEx(redMask, opened, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(redMask, closed, cv::MORPH_CLOSE, kernel);
        vision::saveImage(outputDir + "/erode.png", eroded);
        vision::saveImage(outputDir + "/dilate.png", dilated);
        vision::saveImage(outputDir + "/open.png", opened);
        vision::saveImage(outputDir + "/close.png", closed);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(closed.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat contourResult = image.clone();
        std::vector<double> acceptedAreas;
        for (const auto& contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < 300.0) continue;
            acceptedAreas.push_back(area);
            const cv::Rect box = cv::boundingRect(contour);
            std::vector<std::vector<cv::Point>> one{contour};
            cv::drawContours(contourResult, one, 0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::rectangle(contourResult, box, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            const int labelY = std::max(22, box.y - 6);
            cv::putText(contourResult, cv::format("area=%.0f", area),
                        cv::Point(box.x, labelY), cv::FONT_HERSHEY_SIMPLEX,
                        0.55, cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
            cv::putText(contourResult, cv::format("area=%.0f", area),
                        cv::Point(box.x, labelY), cv::FONT_HERSHEY_SIMPLEX,
                        0.55, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        }
        vision::saveImage(outputDir + "/contours_boxes.png", contourResult);

        cv::Mat drawing = image.clone();
        cv::circle(drawing, cv::Point(image.cols / 4, image.rows / 3),
                   std::min(image.cols, image.rows) / 12, cv::Scalar(255, 0, 0), 4, cv::LINE_AA);
        cv::rectangle(drawing,
                      cv::Rect(image.cols / 2, image.rows / 4, image.cols / 4, image.rows / 3),
                      cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
        cv::putText(drawing, "RoboMaster OpenCV", cv::Point(40, image.rows - 45),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
        cv::putText(drawing, "RoboMaster OpenCV", cv::Point(40, image.rows - 45),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 80, 255), 2, cv::LINE_AA);
        vision::saveImage(outputDir + "/drawing.png", drawing);

        const cv::Point2f center(image.cols / 2.0F, image.rows / 2.0F);
        const cv::Mat rotation = cv::getRotationMatrix2D(center, 35.0, 1.0);
        cv::Mat rotated;
        cv::warpAffine(image, rotated, rotation, image.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        vision::saveImage(outputDir + "/rotated_35deg.png", rotated);

        const cv::Mat crop = image(cv::Rect(0, 0, image.cols / 2, image.rows / 2)).clone();
        vision::saveImage(outputDir + "/crop_top_left.png", crop);

        std::ofstream report(outputDir + "/task1_analysis.txt");
        report << "input=" << inputPath << "\n"
               << "size=" << image.cols << "x" << image.rows << "\n"
               << "mean_kernel=5x5\n"
               << "gaussian_kernel=5x5 sigma=1.5\n"
               << "median_kernel=5\n"
               << "red_hsv=[0,12]U[165,179], S=[70,255], V=[50,255]\n"
               << "morphology_kernel=ellipse 5x5\n"
               << "minimum_contour_area=300 px^2\n"
               << "accepted_contours=" << acceptedAreas.size() << "\n";
        for (std::size_t i = 0; i < acceptedAreas.size(); ++i) {
            report << "area_" << i + 1 << "=" << acceptedAreas[i] << " px^2\n";
        }

        std::cout << "Task 1 complete: wrote 16 required images to " << outputDir << '\n';
        std::cout << "Accepted contour areas (px^2):";
        for (double area : acceptedAreas) std::cout << ' ' << cvRound(area);
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
