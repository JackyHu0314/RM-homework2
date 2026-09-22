#include "vision_utils.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Candidate {
    cv::Point2f center{};
    float radius = 0.0F;
    double area = 0.0;
    double circularity = 0.0;
};

struct TrackerState {
    std::optional<cv::Point2f> windmillCenter;
    std::optional<double> targetAngle;
    std::optional<double> targetRadius;
    std::optional<cv::Point2f> lastPoint;
    int targetId = 0;
    int lostFrames = 0;
    int reselectCount = 0;
};

cv::Mat makeColorMask(const cv::Mat& frame) {
    cv::Mat hsv, redLow, redHigh, blue, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 90, 80), cv::Scalar(15, 255, 255), redLow);
    cv::inRange(hsv, cv::Scalar(160, 90, 80), cv::Scalar(179, 255, 255), redHigh);
    cv::inRange(hsv, cv::Scalar(85, 80, 70), cv::Scalar(135, 255, 255), blue);
    cv::bitwise_or(redLow, redHigh, mask);
    cv::bitwise_or(mask, blue, mask);
    const cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    const cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel3);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel7);
    return mask;
}

std::vector<Candidate> findCandidates(const cv::Mat& mask, double frameArea) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Candidate> result;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < std::max(35.0, frameArea * 0.00004) || area > frameArea * 0.15) continue;
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 1e-6) continue;
        const cv::Moments moments = cv::moments(contour);
        if (std::abs(moments.m00) < 1e-9) continue;
        Candidate candidate;
        candidate.center = cv::Point2f(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00));
        cv::minEnclosingCircle(contour, candidate.center, candidate.radius);
        candidate.area = area;
        candidate.circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        result.push_back(candidate);
    }
    return result;
}

std::optional<cv::Point2f> detectCenter(
    const std::vector<Candidate>& candidates,
    const cv::Size& size,
    const std::optional<cv::Point2f>& previous) {
    if (candidates.empty()) return previous;
    const cv::Point2f expected = previous.value_or(cv::Point2f(size.width / 2.0F, size.height / 2.0F));
    double bestScore = std::numeric_limits<double>::infinity();
    std::optional<cv::Point2f> best;
    const double diagonal = std::hypot(size.width, size.height);
    for (const auto& candidate : candidates) {
        const double positionCost = cv::norm(candidate.center - expected) / diagonal;
        const double roundnessCost = 1.0 - std::clamp(candidate.circularity, 0.0, 1.0);
        const double score = positionCost + 0.18 * roundnessCost + 0.000002 * candidate.area;
        if (score < bestScore) {
            bestScore = score;
            best = candidate.center;
        }
    }
    if (!best) return previous;
    if (previous && cv::norm(*best - *previous) > diagonal * 0.12) return previous;
    if (previous) return *previous * 0.75F + *best * 0.25F;
    return best;
}

std::vector<Candidate> bladeCandidates(
    const std::vector<Candidate>& all,
    const cv::Point2f& center,
    const cv::Size& size) {
    const double minRadius = std::min(size.width, size.height) * 0.06;
    const double maxRadius = std::min(size.width, size.height) * 0.48;
    std::vector<Candidate> blades;
    for (const auto& candidate : all) {
        const double distance = cv::norm(candidate.center - center);
        if (distance >= minRadius && distance <= maxRadius) blades.push_back(candidate);
    }
    return blades;
}

double angleOf(const cv::Point2f& center, const cv::Point2f& point) {
    return std::atan2(center.y - point.y, point.x - center.x);
}

std::optional<Candidate> chooseTarget(
    const std::vector<Candidate>& candidates,
    const cv::Point2f& center,
    const TrackerState& state) {
    if (candidates.empty()) return std::nullopt;
    if (!state.targetAngle || !state.targetRadius) {
        return *std::max_element(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.area < b.area; });
    }
    double bestScore = std::numeric_limits<double>::infinity();
    std::optional<Candidate> best;
    for (const auto& candidate : candidates) {
        const double angle = angleOf(center, candidate.center);
        const double radius = cv::norm(candidate.center - center);
        const double angleCost = vision::angularDistance(angle, *state.targetAngle);
        const double radiusCost = std::abs(radius - *state.targetRadius) / std::max(1.0, *state.targetRadius);
        const double positionCost = state.lastPoint ? cv::norm(candidate.center - *state.lastPoint) / std::max(1.0, *state.targetRadius) : 0.0;
        const double score = angleCost + 0.55 * radiusCost + 0.25 * positionCost;
        if (score < bestScore) {
            bestScore = score;
            best = candidate;
        }
    }
    return bestScore < 1.05 ? best : std::nullopt;
}

std::string baseName(const std::string& path) {
    return std::filesystem::path(path).stem().string();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: task3_windmill <input.mp4> <output-directory>\n";
        return 1;
    }
    const std::string inputPath = argv[1];
    const std::string outputDir = argv[2];
    constexpr int lostTolerance = 12;
    constexpr int reselectAfter = 18;

    try {
        vision::ensureDirectory(outputDir);
        cv::VideoCapture capture(inputPath);
        if (!capture.isOpened()) throw std::runtime_error("Cannot open video: " + inputPath);
        const double fps = capture.get(cv::CAP_PROP_FPS);
        const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        const int expectedFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        if (fps <= 0.0 || width <= 0 || height <= 0) throw std::runtime_error("Invalid video metadata");

        auto overlayWriter = vision::makeMp4Writer(
            outputDir + "/recognition_overlay.mp4", fps, cv::Size(width, height));
        auto binaryWriter = vision::makeMp4Writer(
            outputDir + "/binary_process.mp4", fps, cv::Size(width, height));

        TrackerState state;
        int frameIndex = 0;
        int detectedFrames = 0;
        int lostFramesTotal = 0;
        cv::Mat frame;
        while (capture.read(frame)) {
            cv::Mat mask = makeColorMask(frame);
            auto candidates = findCandidates(mask, static_cast<double>(width) * height);
            state.windmillCenter = detectCenter(candidates, frame.size(), state.windmillCenter);

            std::optional<Candidate> target;
            if (state.windmillCenter) {
                const auto blades = bladeCandidates(candidates, *state.windmillCenter, frame.size());
                target = chooseTarget(blades, *state.windmillCenter, state);
                if (!target && state.lostFrames >= reselectAfter && !blades.empty()) {
                    TrackerState resetState;
                    target = chooseTarget(blades, *state.windmillCenter, resetState);
                    if (target) {
                        ++state.targetId;
                        ++state.reselectCount;
                    }
                }
            }

            std::string status;
            if (target && state.windmillCenter) {
                const double currentAngle = angleOf(*state.windmillCenter, target->center);
                const double currentRadius = cv::norm(target->center - *state.windmillCenter);
                state.targetAngle = currentAngle;
                state.targetRadius = state.targetRadius ? 0.8 * *state.targetRadius + 0.2 * currentRadius : currentRadius;
                state.lastPoint = target->center;
                state.lostFrames = 0;
                if (state.targetId == 0) state.targetId = 1;
                status = "detected";
                ++detectedFrames;

                cv::circle(frame, target->center, std::max(10, cvRound(target->radius)),
                           cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
                cv::circle(frame, target->center, 5, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
                cv::line(frame, *state.windmillCenter, target->center, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            } else {
                ++state.lostFrames;
                ++lostFramesTotal;
                status = "lost";
                if (state.lastPoint && state.lostFrames <= lostTolerance) {
                    cv::circle(frame, *state.lastPoint, 14, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
                }
            }

            if (state.windmillCenter) {
                cv::drawMarker(frame, *state.windmillCenter, cv::Scalar(255, 255, 255),
                               cv::MARKER_CROSS, 28, 3, cv::LINE_AA);
                cv::putText(frame, "R", *state.windmillCenter + cv::Point2f(12.0F, -12.0F),
                            cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            }
            const cv::Scalar statusColor = status == "detected" ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame, cv::format("target ID: %d", state.targetId), cv::Point(24, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
            cv::putText(frame, cv::format("target ID: %d", state.targetId), cv::Point(24, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.85, statusColor, 2, cv::LINE_AA);
            cv::putText(frame, "status: " + status, cv::Point(24, 78),
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, statusColor, 2, cv::LINE_AA);
            cv::putText(frame, cv::format("frame %d", frameIndex), cv::Point(24, 112),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            overlayWriter.write(frame);

            cv::Mat binaryColor;
            cv::cvtColor(mask, binaryColor, cv::COLOR_GRAY2BGR);
            cv::putText(binaryColor, cv::format("frame %d", frameIndex), cv::Point(24, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            binaryWriter.write(binaryColor);
            ++frameIndex;
        }

        std::ofstream metrics(outputDir + "/metrics.txt");
        metrics << "input=" << inputPath << "\n"
                << "width=" << width << "\nheight=" << height << "\nfps=" << fps << "\n"
                << "reported_input_frames=" << expectedFrames << "\n"
                << "written_frames=" << frameIndex << "\n"
                << "detected_frames=" << detectedFrames << "\n"
                << "lost_frames=" << lostFramesTotal << "\n"
                << "lost_tolerance_frames=" << lostTolerance << "\n"
                << "reselect_after_frames=" << reselectAfter << "\n"
                << "reselect_count=" << state.reselectCount << "\n";
        std::cout << baseName(inputPath) << ": wrote " << frameIndex << " frames at " << fps
                  << " FPS; detected=" << detectedFrames << " lost=" << lostFramesTotal
                  << " reselect=" << state.reselectCount << '\n';
        return frameIndex == expectedFrames || expectedFrames <= 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
