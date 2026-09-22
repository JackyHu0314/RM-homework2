#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vision {

inline void ensureDirectory(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error("Cannot create directory: " + path + ": " + error.message());
    }
}

inline void saveImage(const std::string& path, const cv::Mat& image) {
    if (image.empty() || !cv::imwrite(path, image)) {
        throw std::runtime_error("Cannot save image: " + path);
    }
}

inline cv::VideoWriter makeMp4Writer(
    const std::string& path,
    double fps,
    const cv::Size& size,
    bool color = true) {
    cv::VideoWriter writer(
        path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        size,
        color);
    if (!writer.isOpened()) {
        throw std::runtime_error("Cannot create video: " + path);
    }
    return writer;
}

inline double wrapPi(double angle) {
    constexpr double pi = 3.14159265358979323846;
    while (angle >= pi) angle -= 2.0 * pi;
    while (angle < -pi) angle += 2.0 * pi;
    return angle;
}

inline double angularDistance(double a, double b) {
    return std::abs(wrapPi(a - b));
}

inline cv::Mat makePlot(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& series,
    const std::vector<cv::Scalar>& colors,
    const std::string& title,
    const std::string& xLabel,
    const std::string& yLabel,
    const std::vector<std::string>& legends = {}) {
    const int width = 1280;
    const int height = 720;
    const int left = 105;
    const int right = 40;
    const int top = 75;
    const int bottom = 90;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(250, 250, 250));

    if (x.empty() || series.empty()) return canvas;
    double xMin = *std::min_element(x.begin(), x.end());
    double xMax = *std::max_element(x.begin(), x.end());
    double yMin = std::numeric_limits<double>::infinity();
    double yMax = -std::numeric_limits<double>::infinity();
    for (const auto& values : series) {
        for (double value : values) {
            if (std::isfinite(value)) {
                yMin = std::min(yMin, value);
                yMax = std::max(yMax, value);
            }
        }
    }
    if (!std::isfinite(yMin) || !std::isfinite(yMax)) {
        yMin = -1.0;
        yMax = 1.0;
    }
    if (std::abs(xMax - xMin) < 1e-12) xMax = xMin + 1.0;
    if (std::abs(yMax - yMin) < 1e-12) {
        yMin -= 1.0;
        yMax += 1.0;
    }
    const double margin = 0.08 * (yMax - yMin);
    yMin -= margin;
    yMax += margin;

    auto mapPoint = [&](double xv, double yv) {
        const int px = left + static_cast<int>((xv - xMin) / (xMax - xMin) * (width - left - right));
        const int py = height - bottom - static_cast<int>((yv - yMin) / (yMax - yMin) * (height - top - bottom));
        return cv::Point(px, py);
    };

    cv::rectangle(canvas, cv::Point(left, top), cv::Point(width - right, height - bottom), cv::Scalar(60, 60, 60), 1);
    for (int i = 0; i <= 5; ++i) {
        const double ratio = i / 5.0;
        const int px = left + static_cast<int>(ratio * (width - left - right));
        const int py = height - bottom - static_cast<int>(ratio * (height - top - bottom));
        cv::line(canvas, cv::Point(px, top), cv::Point(px, height - bottom), cv::Scalar(220, 220, 220), 1);
        cv::line(canvas, cv::Point(left, py), cv::Point(width - right, py), cv::Scalar(220, 220, 220), 1);
        cv::putText(canvas, cv::format("%.1f", xMin + ratio * (xMax - xMin)),
                    cv::Point(px - 18, height - bottom + 28), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(40, 40, 40), 1);
        cv::putText(canvas, cv::format("%.3f", yMin + ratio * (yMax - yMin)),
                    cv::Point(8, py + 5), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(40, 40, 40), 1);
    }

    for (std::size_t s = 0; s < series.size(); ++s) {
        std::vector<cv::Point> points;
        const std::size_t count = std::min(x.size(), series[s].size());
        points.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (std::isfinite(series[s][i])) points.push_back(mapPoint(x[i], series[s][i]));
        }
        if (points.size() > 1) cv::polylines(canvas, points, false, colors[s % colors.size()], s == 0 ? 1 : 3, cv::LINE_AA);
        if (s == 0) {
            const std::size_t stride = std::max<std::size_t>(1, points.size() / 240);
            for (std::size_t i = 0; i < points.size(); i += stride) cv::circle(canvas, points[i], 2, colors[s % colors.size()], -1, cv::LINE_AA);
        }
    }

    cv::putText(canvas, title, cv::Point(left, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    cv::putText(canvas, xLabel, cv::Point(width / 2 - 35, height - 28), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    cv::putText(canvas, yLabel, cv::Point(12, 35), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    for (std::size_t i = 0; i < legends.size(); ++i) {
        const int x0 = width - right - 245;
        const int y0 = top + 25 + static_cast<int>(i) * 28;
        cv::line(canvas, cv::Point(x0, y0), cv::Point(x0 + 30, y0), colors[i % colors.size()], 3, cv::LINE_AA);
        cv::putText(canvas, legends[i], cv::Point(x0 + 42, y0 + 5), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    }
    return canvas;
}

}  // namespace vision
