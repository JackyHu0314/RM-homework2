#include "vision_utils.hpp"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace {

struct Observation {
    int frame = 0;
    double time = 0.0;
    cv::Point2f point{};
    double angle = 0.0;
};

struct FitResult {
    double amplitude = 0.0;
    double meanSpeed = 0.0;
    double frequency = 0.0;
    double phase = 0.0;
    double rmse = std::numeric_limits<double>::infinity();
    int sampleCount = 0;
};

std::optional<cv::Point2f> detectCyanTarget(const cv::Mat& frame, cv::Mat* outputMask = nullptr) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(78, 80, 80), cv::Scalar(105, 255, 255), mask);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    if (outputMask) *outputMask = mask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double bestArea = 0.0;
    cv::Point2f bestPoint;
    bool found = false;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 15.0 || area <= bestArea) continue;
        const cv::Moments moments = cv::moments(contour);
        if (std::abs(moments.m00) < 1e-9) continue;
        bestArea = area;
        bestPoint = cv::Point2f(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00));
        found = true;
    }
    return found ? std::optional<cv::Point2f>(bestPoint) : std::nullopt;
}

std::vector<double> unwrapAngles(const std::vector<Observation>& observations) {
    std::vector<double> result;
    if (observations.empty()) return result;
    result.reserve(observations.size());
    result.push_back(observations.front().angle);
    for (std::size_t i = 1; i < observations.size(); ++i) {
        const double delta = vision::wrapPi(observations[i].angle - observations[i - 1].angle);
        result.push_back(result.back() + delta);
    }
    return result;
}

std::vector<double> movingAverage(const std::vector<double>& values, int halfWindow) {
    std::vector<double> smoothed(values.size(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int begin = std::max(0, static_cast<int>(i) - halfWindow);
        const int end = std::min(static_cast<int>(values.size()) - 1, static_cast<int>(i) + halfWindow);
        double sum = 0.0;
        int count = 0;
        for (int j = begin; j <= end; ++j) {
            if (std::isfinite(values[j])) {
                sum += values[j];
                ++count;
            }
        }
        smoothed[i] = count > 0 ? sum / count : std::numeric_limits<double>::quiet_NaN();
    }
    return smoothed;
}

std::vector<double> estimateAngularSpeed(
    const std::vector<Observation>& observations,
    const std::vector<double>& unwrapped) {
    std::vector<double> speed(unwrapped.size(), std::numeric_limits<double>::quiet_NaN());
    if (unwrapped.size() < 3) return speed;
    for (std::size_t i = 1; i + 1 < unwrapped.size(); ++i) {
        const double dt = observations[i + 1].time - observations[i - 1].time;
        if (dt > 1e-9) speed[i] = (unwrapped[i + 1] - unwrapped[i - 1]) / dt;
    }
    speed.front() = speed[1];
    speed.back() = speed[speed.size() - 2];
    return movingAverage(speed, 5);
}

FitResult fitForFrequency(
    const std::vector<double>& time,
    const std::vector<double>& speed,
    double frequency) {
    cv::Matx33d normal = cv::Matx33d::zeros();
    cv::Vec3d rhs(0.0, 0.0, 0.0);
    int count = 0;
    for (std::size_t i = 0; i < time.size(); ++i) {
        if (!std::isfinite(speed[i])) continue;
        const cv::Vec3d row(1.0, std::sin(frequency * time[i]), std::cos(frequency * time[i]));
        normal += row * row.t();
        rhs += row * speed[i];
        ++count;
    }
    cv::Vec3d parameters;
    if (count < 10 || !cv::solve(normal, rhs, parameters, cv::DECOMP_SVD)) return {};

    const double amplitude = std::hypot(parameters[1], parameters[2]);
    if (amplitude <= 0.0 || parameters[0] <= amplitude) return {};
    double squaredError = 0.0;
    for (std::size_t i = 0; i < time.size(); ++i) {
        if (!std::isfinite(speed[i])) continue;
        const double prediction = parameters[0] + parameters[1] * std::sin(frequency * time[i])
                                + parameters[2] * std::cos(frequency * time[i]);
        const double residual = speed[i] - prediction;
        squaredError += residual * residual;
    }
    FitResult fit;
    fit.amplitude = amplitude;
    fit.meanSpeed = parameters[0];
    fit.frequency = frequency;
    fit.phase = vision::wrapPi(std::atan2(parameters[2], parameters[1]));
    fit.rmse = std::sqrt(squaredError / count);
    fit.sampleCount = count;
    return fit;
}

FitResult fitModel(const std::vector<double>& time, const std::vector<double>& speed) {
    FitResult best;
    for (int i = 0; i <= 1200; ++i) {
        const double frequency = 0.08 + i * (4.0 - 0.08) / 1200.0;
        const FitResult candidate = fitForFrequency(time, speed, frequency);
        if (candidate.sampleCount > 0 && candidate.rmse < best.rmse) best = candidate;
    }
    if (!std::isfinite(best.rmse)) return best;
    const double coarseStep = (4.0 - 0.08) / 1200.0;
    for (int i = -100; i <= 100; ++i) {
        const double frequency = best.frequency + i * coarseStep / 100.0;
        if (frequency <= 0.0) continue;
        const FitResult candidate = fitForFrequency(time, speed, frequency);
        if (candidate.sampleCount > 0 && candidate.rmse < best.rmse) best = candidate;
    }
    return best;
}

double predictSpeed(const FitResult& fit, double time) {
    return fit.meanSpeed + fit.amplitude * std::sin(fit.frequency * time + fit.phase);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string inputPath = argc > 1 ? argv[1] : "resources/task_2.mp4";
    const std::string outputDir = argc > 2 ? argv[2] : "result/task2_fit";
    try {
        vision::ensureDirectory(outputDir);
        cv::VideoCapture capture(inputPath);
        if (!capture.isOpened()) {
            std::cerr << "Cannot open video: " << inputPath << '\n';
            return 1;
        }
        const double fps = capture.get(cv::CAP_PROP_FPS);
        const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (fps <= 0.0 || width <= 0 || height <= 0) throw std::runtime_error("Invalid video metadata");
        const cv::Point2f rotationCenter(width / 2.0F, height / 2.0F);

        std::vector<Observation> observations;
        cv::Mat frame;
        int frameIndex = 0;
        while (capture.read(frame)) {
            const auto target = detectCyanTarget(frame);
            if (target) {
                const double angle = std::atan2(rotationCenter.y - target->y, target->x - rotationCenter.x);
                observations.push_back({frameIndex, frameIndex / fps, *target, angle});
            }
            ++frameIndex;
        }
        if (observations.size() < 30) throw std::runtime_error("Too few cyan-target observations for fitting");

        const std::vector<double> unwrapped = unwrapAngles(observations);
        const std::vector<double> speed = estimateAngularSpeed(observations, unwrapped);
        std::vector<double> time;
        time.reserve(observations.size());
        for (const auto& observation : observations) time.push_back(observation.time);
        const FitResult fit = fitModel(time, speed);
        if (!std::isfinite(fit.rmse)) throw std::runtime_error("Model fitting failed");

        std::vector<double> fitted(speed.size()), residuals(speed.size());
        for (std::size_t i = 0; i < speed.size(); ++i) {
            fitted[i] = predictSpeed(fit, time[i]);
            residuals[i] = speed[i] - fitted[i];
        }
        const cv::Mat comparison = vision::makePlot(
            time, {speed, fitted}, {cv::Scalar(30, 120, 220), cv::Scalar(20, 150, 40)},
            "Observed and fitted angular velocity", "time (s)", "omega (rad/s)", {"observed", "fitted"});
        vision::saveImage(outputDir + "/fit_comparison.png", comparison);
        vision::saveImage(outputDir + "/angular_velocity.png", comparison);
        const cv::Mat residualPlot = vision::makePlot(
            time, {residuals}, {cv::Scalar(180, 60, 80)},
            "Angular velocity residuals", "time (s)", "residual (rad/s)", {"residual"});
        vision::saveImage(outputDir + "/residuals.png", residualPlot);

        capture.open(inputPath);
        auto writer = vision::makeMp4Writer(outputDir + "/tracking_overlay.mp4", fps, cv::Size(width, height));
        frameIndex = 0;
        while (capture.read(frame)) {
            const auto target = detectCyanTarget(frame);
            cv::circle(frame, rotationCenter, 7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            if (target) {
                cv::circle(frame, *target, 13, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
                cv::line(frame, rotationCenter, *target, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::putText(frame, "detected", cv::Point(25, 45), cv::FONT_HERSHEY_SIMPLEX,
                            0.9, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            } else {
                cv::putText(frame, "lost", cv::Point(25, 45), cv::FONT_HERSHEY_SIMPLEX,
                            0.9, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            cv::putText(frame, cv::format("frame=%d  t=%.3fs", frameIndex, frameIndex / fps),
                        cv::Point(25, 82), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            writer.write(frame);
            ++frameIndex;
        }

        std::ofstream report("result/task2_fit_result.md");
        report << "# 任务 2：合成旋转视频参数拟合\n\n"
               << "## 方法\n\n"
               << "通过 HSV 区间 `H=[78,105], S=[80,255], V=[80,255]` 提取青色目标，"
                  "以画面中心为旋转中心，按 `atan2(cy-y, x-cx)` 计算并展开角度。"
                  "对中心差分得到的角速度做 11 帧移动平均，再对频率进行一维搜索；"
                  "固定频率时使用线性最小二乘求解常数项、正弦项和余弦项。\n\n"
               << "约束为 `A > 0`、`b > A`、`Omega > 0`，相位归一化到 `[-pi, pi)`。\n\n"
               << "## 估计结果\n\n"
               << std::fixed << std::setprecision(8)
               << "- A = " << fit.amplitude << " rad/s\n"
               << "- b = " << fit.meanSpeed << " rad/s\n"
               << "- Omega = " << fit.frequency << " rad/s\n"
               << "- phi = " << fit.phase << " rad\n"
               << "- 角速度 RMSE = " << fit.rmse << " rad/s\n"
               << "- 有效样本数 = " << fit.sampleCount << "\n"
               << "- 参与计算帧范围 = " << observations.front().frame << " - " << observations.back().frame << "\n"
               << "- 时间原点 = 视频第 0 帧\n\n"
               << "## 结果文件\n\n"
               << "- [跟踪标注视频](task2_fit/tracking_overlay.mp4)\n"
               << "- [观测与拟合对比](task2_fit/fit_comparison.png)\n"
               << "- [角速度曲线](task2_fit/angular_velocity.png)\n"
               << "- [残差曲线](task2_fit/residuals.png)\n";

        std::cout << std::fixed << std::setprecision(8)
                  << "A=" << fit.amplitude << " b=" << fit.meanSpeed
                  << " Omega=" << fit.frequency << " phi=" << fit.phase
                  << " RMSE=" << fit.rmse << " rad/s samples=" << fit.sampleCount << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
