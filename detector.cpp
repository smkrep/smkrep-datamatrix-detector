#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <fstream>
namespace fs = std::filesystem;



const int area_threshold = 1000;
const int size_threshold = 1200;
const double distance_threshold = 10;
const double vertex_threshold = 10;
const double iou_filter_threshold = 0.7;



// Function that decides if the input image is grayscale 
bool is_grayscale(const cv::Mat& image) {
    if (image.channels() == 1) 
        return true;
    
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            if (pixel[0] != pixel[1] || pixel[1] != pixel[2])
                return false; 
        }
    }
    return true;
}


// Function that decides if the component is considered "small"
bool is_small_component(const int area){
    return area < area_threshold;
}


// Function that removes small components from the original input image
cv::Mat remove_small_components(cv::Mat img, cv::Mat stats){
    cv::Mat res = img.clone();
    res = (res > 0) * 255;
    for(int i = 1; i < stats.rows; i++){
        if(is_small_component(stats.at<int>(i, cv::CC_STAT_AREA))){
            cv::Mat mask = (img == i);
            mask = ~mask;
            cv::bitwise_and(res, mask, res);
        }
    }
    return res;
}


// Function that fits the region of interest rectangle into the image, if the rectangle is out of bounds
void fit_roi_into_image(cv::Rect& roi_rect, int rows, int cols){
    if (roi_rect.x < 0) roi_rect.x = 0;
    if (roi_rect.y < 0) roi_rect.y = 0;
    if (roi_rect.x + roi_rect.width > cols) roi_rect.width = cols - roi_rect.x;
    if (roi_rect.y + roi_rect.height > rows) roi_rect.height = rows - roi_rect.y;
    
}


// Function that calculates distance between two points
double distance(cv::Point pt1, cv::Point pt2) {
    return sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));
}


// Function that checks if the given point is close to any given line
bool is_point_near_line_edge(cv::Point pt, cv::Point line_pt1, cv::Point line_pt2, double threshold) {
    double dist1 = distance(pt, line_pt1);
    double dist2 = distance(pt, line_pt2);

    double line_length = distance(line_pt1, line_pt2);
    if (line_length == 0) return false; 
    return dist1 < threshold || dist2 < threshold;
}


// Function that calculates the angle between two given lines. The returned angle is [0;180] in degrees.
double angle_between_lines(cv::Vec4f line1, cv::Vec4f line2) {
    cv::Point v1(line1[2] - line1[0], line1[3] - line1[1]);
    cv::Point v2(line2[2] - line2[0], line2[3] - line2[1]);
    double angle = atan2(v2.y, v2.x) - atan2(v1.y, v1.x);
    if (angle < 0) angle += 2 * CV_PI;
    double ans = angle * 180.0 / CV_PI;
    if(ans > 180 && ans <= 270){
        ans-=180;
    }
    if(ans > 270 && ans <= 360){
        ans-=360;
        ans = -ans;
    }
    return ans;
}


// Function that loads all .png and .jpg files from the input directory
std::vector<std::pair<cv::Mat, std::string>> get_pictures_from_input_dir(std::filesystem::path input_folder){

    std::vector<std::pair<cv::Mat, std::string>> pics;
    for (const auto& file : fs::directory_iterator(input_folder)) {
        if (!fs::is_directory(file)) {

            if (fs::path(file).extension() == ".png" || fs::path(file).extension() == ".jpg")
            {
                fs::path img = fs::path(file);
                cv::Mat tmp = cv::imread(img.string());
                if(is_grayscale(tmp)){
                    tmp = cv::imread(img.string(), cv::IMREAD_GRAYSCALE);
                    cv::cvtColor(tmp, tmp,  cv::COLOR_GRAY2RGB);
                }
                else{
                    tmp = cv::imread(img.string(), cv::IMREAD_COLOR);
                }
                pics.push_back(std::pair<cv::Mat, std::string>(tmp, img.filename().string()));
            }
        }
    }
    return pics;

}


// Function that performs Canny egde detection algorithm 
cv::Mat extract_edges(const cv::Mat& grayscale_image, const int& canny_thresh){
    cv::Mat edges;
    cv::blur(grayscale_image, edges, cv::Size(3,3));
    cv::Canny(edges, edges, canny_thresh, canny_thresh * 3, 3);
    return edges;
}


// Function that performs morphological opening and dilation to distinguish possible datamatrixes from the background
cv::Mat perform_morphology_and_component_filtering(cv::Mat edges){
    // cv::adaptiveThreshold(elem.first, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 9, 0);
    cv::Mat opelem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)), //3
            dilelem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9)); //4
    cv::morphologyEx(edges, edges, cv::MORPH_DILATE, dilelem);
    cv::morphologyEx(edges, edges, cv::MORPH_OPEN, opelem);
    //cv::imwrite(output_folder.string() + "processed_" + elem.second, dst);
    
    cv::Mat components, stats, centroids;
    cv::connectedComponentsWithStats(edges, components, stats, centroids, 8);
    cv::Mat filtered = remove_small_components(components, stats);
    //cv::imwrite(output_folder.string() + "filtered" + elem.second, filtered);

    return filtered;

}


// Function that implements the algorithm of finding pairs of lines that are possible L-shape finder patterns
std::vector<std::pair<cv::Vec4f, cv::Vec4f>> find_l_shape_finder_patterns(std::vector<std::pair<cv::Vec4f, bool>> marked_lines){

    std::vector<std::pair<cv::Vec4f, cv::Vec4f>> l_shape_pattern_candidates;

    for(int i = 0; i < marked_lines.size(); i++){
        cv::Vec4f line_i = marked_lines[i].first;
        bool mark_i = marked_lines[i].second;

        if(mark_i == false){
            
            cv::Point first_i = cv::Point(line_i[0], line_i[1]);
            cv::Point second_i = cv::Point(line_i[2], line_i[3]);
            double line_i_length = distance(first_i, second_i);

            for(int j = 0; j < marked_lines.size(); j++){
                
                bool mark_j = marked_lines[j].second;
                if ((i == j) || mark_j == true) continue;

                cv::Vec4f line_j = marked_lines[j].first;
                cv::Point first_j = cv::Point(line_j[0], line_j[1]);
                cv::Point second_j = cv::Point(line_j[2], line_j[3]);

                double line_j_length = distance(first_j, second_j);

                bool is_close_to_line_i = is_point_near_line_edge(first_j, first_i, second_i, distance_threshold) ||
                is_point_near_line_edge(second_j, first_i, second_i, distance_threshold);

                double angle = angle_between_lines(line_i, line_j);

                bool angle_is_valid = angle > 60 && angle < 120;

                double length_ratio = std::max(line_i_length, line_j_length) / std::min(line_i_length, line_j_length);

                if(is_close_to_line_i && angle_is_valid && length_ratio < 5){
                    l_shape_pattern_candidates.push_back({line_i, line_j});
                    marked_lines[j].second = true;
                    break;
                }

                marked_lines[i].second = true;
            }
        }
    }

    return l_shape_pattern_candidates;
}


// Function that calculates the intersection point of two lines
cv::Point2f get_intersection_point(const cv::Vec4f& line1, const cv::Vec4f& line2) {

    float x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
    
    float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    
    float t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    float u_num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);

    float t = t_num / denom;
    float u = -u_num / denom;

    float intersect_x = x1 + t * (x2 - x1);
    float intersect_y = y1 + t * (y2 - y1);

    return cv::Point2f(intersect_x, intersect_y);
}


// Function that extends the given line to the given point
cv::Vec4f extend_line_to_a_point(const cv::Vec4f& line, const cv::Point2f& point) {
    cv::Vec4f extended_line;
    if(distance(cv::Point(line[0], line[1]), point) > distance(cv::Point(line[2], line[3]), point)){
        extended_line = {line[0], line[1], point.x, point.y};
    }
    else{
        extended_line = {point.x, point.y, line[2], line[3]};
    }
    return extended_line;
}


// Function that draws a set of given points on a given image
void draw_points(const cv::Mat &image, const std::vector<cv::Point> &points, const cv::Scalar &color) {
    for (const auto &point : points) {
        cv::circle(image, point, 1, color, -1);
    }
}


// Function that calculates all points of the line to scan for dashed border
std::vector<cv::Point2f> get_scan_line_points(const cv::Point2f& pt1, const cv::Point2f& pt2) {

    std::vector<cv::Point2f> points;

    float x1 = pt1.x, y1 = pt1.y, x2 = pt2.x, y2 = pt2.y;

    float dx = std::abs(x2 - x1), dy = std::abs(y2 - y1);

    int steps = static_cast<int>(std::max(dx, dy));
    float x_increment = (x2 - x1) / steps, y_increment = (y2 - y1) / steps;

    float x = x1, y = y1;

    for (int i = 0; i <= steps; i++) {
        points.push_back(cv::Point2f(x, y));
        x += x_increment;
        y += y_increment;
    }

    return points;
}


// Function that calculates the coordinates of the vertex diagonal to the intersection point
void find_diagonal_vertex(const cv::Vec4f& line1, const cv::Vec4f& line2, const cv::Point2f& intersection,
                   cv::Point2f& vertex1, cv::Point2f& vertex2, 
                   cv::Point2f& vertex_diagonal) {


    cv::Point2f p1(line1[0], line1[1]);
    cv::Point2f p2(line1[2], line1[3]);
    cv::Point2f p3(line2[0], line2[1]);
    cv::Point2f p4(line2[2], line2[3]);

    std::vector<cv::Point2f> points_init = {p1, p2, p3, p4};
    std::vector<cv::Point2f> points;


    for(int i = 0; i < points_init.size(); i++){
        if(points_init[i] != intersection){
            points.push_back(points_init[i]);
        }
    }
    
    
    std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y > b.y) || (a.y == b.y && a.x < b.x);
    });

    cv::Vec2f AB = {points[1].x - intersection.x, points[1].y - intersection.y};
    cv::Vec2f AD = {points[0].x - intersection.x, points[0].y - intersection.y};
    cv::Vec2f AC = {AD[0] + AB[0], AD[1] + AB[1]};

    vertex1 = points[0];
    vertex2 = points[1];
    vertex_diagonal = cv::Point2f(AC[0] + intersection.x, AC[1] + intersection.y);

}


// Function that decides if the detected pattern is a dashed border
bool dashed_border_is_valid(const std::vector<cv::Point>& timing_pattern_points1, const std::vector<cv::Point>& timing_pattern_points2){
    int size_1 = timing_pattern_points1.size();
    int size_2 = timing_pattern_points2.size();

    float ratio = (float) std::max(size_1, size_2) / std::min(size_1, size_2);

    return ratio <= 2  && timing_pattern_points1.size() > 3 && timing_pattern_points2.size() > 3;
}


// Function to order vertices in counterclockwise order
std::vector<cv::Point> order_vertices_counterclockwise(const std::vector<cv::Point>& vertices) {
    cv::Point centroid(0, 0);
    for (const auto& vertex : vertices) {
        centroid += vertex;
    }
    centroid.x /= vertices.size();
    centroid.y /= vertices.size();

    std::vector<cv::Point> ordered_vertices = vertices;

    std::sort(ordered_vertices.begin(), ordered_vertices.end(),
              [&centroid](const cv::Point& a, const cv::Point& b) {
                  return atan2(a.y - centroid.y, a.x - centroid.x) < atan2(b.y - centroid.y, b.x - centroid.x);
              });

    return ordered_vertices;
}


// Function that detects dashed border near the given L-finder pattern
std::vector<cv::Point> detect_dashed_border(const cv::Vec4f& first_line, const cv::Vec4f& second_line, 
                                            const cv::Point2f& intersection, const cv::Mat& edges){
    
    cv::Point2f vertex1, vertex2, vertex_diagonal;

    find_diagonal_vertex(first_line, second_line, intersection, vertex1, vertex2, vertex_diagonal);

    std::vector<cv::Point2f> scan_line_points_1, scan_line_points_2;
    scan_line_points_1 = get_scan_line_points(vertex1, vertex_diagonal);
    scan_line_points_2 = get_scan_line_points(vertex2, vertex_diagonal);

    std::vector<cv::Point> timing_pattern_points1;
    std::vector<cv::Point> timing_pattern_points2;

    for (const auto& point : scan_line_points_1) {

        bool point_is_in_bounds = round(point.x) >= 0 && round(point.x) < edges.cols && round(point.y) >= 0 && round(point.y) < edges.rows;

        if (!point_is_in_bounds) continue;

        if(edges.at<ushort>(point) > 0){
            timing_pattern_points1.push_back(point);
        }
    }

    for (const auto& point : scan_line_points_2) {

        bool point_is_in_bounds = round(point.x) >= 0 && round(point.x) < edges.cols && round(point.y) >= 0 && round(point.y) < edges.rows;

        if (!point_is_in_bounds) continue;

        if(edges.at<ushort>(point) > 0){
            timing_pattern_points2.push_back(point);
        }
    }
    

    // std::cout << "pattern1: ";
    // for(int i = 1; i < timingPatternPoints1.size(); i++){
    //     std::cout  << distance(timingPatternPoints1[i],timingPatternPoints1[i-1]) << " ";
    // }
    // std::cout << "\n" << "pattern2: ";
    // for(int i = 1; i < timingPatternPoints2.size(); i++){
    //     std::cout  << distance(timingPatternPoints2[i],timingPatternPoints2[i-1]) << " ";
    // }
    // std::cout << "\n\n\n";

    

    if(dashed_border_is_valid(timing_pattern_points1, timing_pattern_points2)){

        // cv::Mat disp = edges.clone();
        // disp.convertTo(disp, CV_8UC1);
        // cv::cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

        int finder_pattern_size = std::max(timing_pattern_points1.size(), timing_pattern_points2.size());
        std::vector<cv::Point> ordered_vertices = order_vertices_counterclockwise({intersection, vertex1, vertex2, vertex_diagonal});
        std::vector<cv::Point> possible_detection = {ordered_vertices};

        
        // cv::circle(disp, intersection, 1, cv::Scalar(0, 0, 255), -1);
        // cv::circle(disp, vertex1, 1, cv::Scalar(0, 0, 255), -1);
        // cv::circle(disp, vertex2, 1, cv::Scalar(0, 0, 255), -1);
        // cv::circle(disp, vertex_diagonal, 1, cv::Scalar(0, 0, 255), -1);
        // draw_points(disp, timing_pattern_points1, cv::Scalar(0, 0, 255));
        // draw_points(disp, timing_pattern_points2, cv::Scalar(0, 0, 255));
        // cv::imshow("pattern", disp);
        // cv::waitKey(0);

        return possible_detection;

    }
    else{
        return {};
    }

}


// Function that calculates intersection over union for two given convex polynoms
double calculate_IoU(const std::vector<cv::Point>& poly_first, const std::vector<cv::Point>& poly_second) {
    std::vector<cv::Point> intersectionPoly;

    double intersection_area = cv::intersectConvexConvex(poly_first, poly_second, intersectionPoly);
    if (intersection_area == 0) {
        return 0.0f;
    }
    
    double area_first = cv::contourArea(poly_first), area_second = cv::contourArea(poly_second);
    double union_area = area_first + area_second - intersection_area;
    
    return intersection_area / union_area;
}


// Function that removes duplicate detections
std::vector<std::vector<cv::Point>> filter_detections(const std::vector<std::vector<cv::Point>>& detections, float iou_filter_threshold = 0.5) {

    std::vector<std::vector<cv::Point>> filtered_detections;
    std::vector<bool> to_keep(detections.size(), true);
    
    for (int i = 0; i < detections.size(); i++) {

        double ratio = std::max(distance(detections[i][1], detections[i][2]), distance(detections[i][0], detections[i][1])) / std::min(distance(detections[i][1], detections[i][2]), distance(detections[i][0], detections[i][1]));
        if (ratio >= 1.5) to_keep[i] = false;
        if (!to_keep[i]) continue;
        std::vector<cv::Point> detection = detections[i];

        for (int j = i + 1; j < detections.size(); j++) {

            if (!to_keep[j]) continue;

            float iou = calculate_IoU(detections[i], detections[j]);
            if (iou > iou_filter_threshold) {
                to_keep[j] = false;
            }
        }
    }
    
    for (int i = 0; i < detections.size(); i++) {
        if (to_keep[i]) {
            filtered_detections.push_back(detections[i]);
        }
    }
    
    return filtered_detections;
}


// Function that composes a json file with info about all found detections
void compose_detections_json(const std::vector<std::vector<cv::Point>>& detections, const std::string& filename, const fs::path& output_folder, const float& img_area){
    std::string filename_without_extension = "";

    for(int k = 0; k < filename.size(); k++){
        if(filename[k] == '.'){
            break;
        }
        else{
            filename_without_extension+=filename[k];
        } 
    }

    cv::FileStorage gt_json(output_folder.string() + filename_without_extension + "_detections.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    gt_json << "detections_" +  filename_without_extension<< "{";
    gt_json << "img_area" << img_area;

    gt_json << "regions" << "[";

    for(int i = 0; i < detections.size(); i++){
        gt_json << "{";
        gt_json << "shape_attributes" << "{";

        std::vector<double> all_points_x, all_points_y;
        for(int j = 0; j < detections[i].size(); j++){
            all_points_x.push_back(detections[i][j].x);
            all_points_y.push_back(detections[i][j].y);
        }
        gt_json << "all_points_x"  << all_points_x;
        gt_json << "all_points_y" << all_points_y;
        
        gt_json << "}";
        gt_json << "}";
    }

    gt_json <<  "]";
    gt_json << "}";
    gt_json.release();
}


// Function that extracts data from ground_truth and detections json files
std::vector<std::vector<cv::Point>> extract_data_from_json(const std::string& json_filename, float& img_area){
    std::vector<std::vector<cv::Point>> gt_polygons;
    cv::FileStorage json(json_filename, 0);
    cv::FileNode root = json.getFirstTopLevelNode();
    cv::FileNode regions = root["regions"];
    cv::FileNode area = root["img_area"];
    img_area = area.real();

    for(int i = 0; i < regions.size(); i++){
        std::vector<cv::Point> polygon;
        cv::FileNode shape_attributes = regions[i]["shape_attributes"];
        cv::FileNode all_points_x = shape_attributes["all_points_x"];
        cv::FileNode all_points_y = shape_attributes["all_points_y"];
        for(int j = 0; j < 4; j++){
            polygon.push_back(cv::Point(all_points_x[j].real(), all_points_y[j].real()));
        }
        gt_polygons.push_back(polygon);
    }
    json.release();

    return gt_polygons;
}


// Function calculates IoU's and puts them in a matrix
std::vector<std::vector<double>> calculate_ious_matrix(const std::vector<std::vector<cv::Point>>& reference, const std::vector<std::vector<cv::Point>>& detections){


    std::vector<std::vector<double>> ious(detections.size(), std::vector<double>(reference.size(), (float)-1));

    for(int i = 0; i < detections.size(); i++){
        std::vector<cv::Point> detected_polygon = detections[i];
        for(int j = 0; j < reference.size(); j++){
            std::vector<cv::Point> gt_polygon = reference[j];
            ious[i][j] = calculate_IoU(detected_polygon, gt_polygon);
        }

    }

    return ious;
}


// Function that calculates True Positive, False Positive and False Negative stat from IoU values
std::tuple<int, int, int> calculate_evaluations(const std::vector<std::vector<double>>& ious, const double& quality_threshold){

    int TP = 0, FP = 0, FN = 0;
    for(auto detection_row: ious){
        bool FalsePositive = true;
        for(auto elem: detection_row){
            if (elem > quality_threshold){
                FalsePositive = false;
                break;
            }
        }

        if(FalsePositive) FP++;
    }
    for(int j = 0; j < ious[0].size(); j++){
        bool TruePositive = false;
        for(int i = 0; i < ious.size(); i++){
            if(ious[i][j] > quality_threshold){
                TruePositive = true;
                break;
            }
        }
        if(TruePositive) TP++; else FN++;
    }
    return {TP, FP, FN};
}

std::vector<std::vector<double>> calculate_precision_recall_f1(const std::vector<cv::String>& gt_jsons, const std::vector<cv::String>& det_jsons) {
    std::vector<std::vector<double>> precision_recall_f1;
    double iou_eval_threshold = 0.01;

    for(double thresh = iou_eval_threshold; thresh < 1; thresh+=0.1){
        int TP = 0, FP = 0, FN = 0;

        for(int i = 0; i < gt_jsons.size(); i++){

            float area;
            std::vector<std::vector<cv::Point>> gt_polygons, det_polygons;

            gt_polygons = extract_data_from_json(gt_jsons[i], area);
            det_polygons = extract_data_from_json(det_jsons[i], area);


            if(det_polygons.empty()){
                FN+=gt_polygons.size();
                continue;
            }
            else{
                std::vector<std::vector<double>> ious = calculate_ious_matrix(gt_polygons, det_polygons);

                std::tuple<int, int, int> scores = calculate_evaluations(ious, thresh);

                TP += std::get<0>(scores);
                FP += std::get<1>(scores);
                FN += std::get<2>(scores);
            }
        }

            
        std::cout << "\t\tCalculating precision/recall..." << std::endl << std::endl;

        double precision = (double) TP / (TP + FP);
        double recall = (double) TP / (TP + FN);
        double f1_score = 2 * precision * recall / (precision + recall);
        if (precision == 0 && recall == 0) f1_score = 0;
        precision_recall_f1.push_back({thresh, precision, recall, f1_score});
        
    }
    return precision_recall_f1;
}

void plotMultipleGraphs(const std::vector<std::vector<std::pair<double, double>>>& dataSets, const std::string& windowName) {
    int width = 800;
    int height = 600;
    int margin = 50;
    int extendedMargin = 100; 

    cv::Mat plotImage(height, width + extendedMargin, CV_8UC3, cv::Scalar(255, 255, 255));

    int plotWidth = width - 2 * margin;
    int plotHeight = height - 2 * margin;

    cv::line(plotImage, cv::Point(margin + extendedMargin, height - margin), cv::Point(margin + extendedMargin, margin), cv::Scalar(0, 0, 0), 2);
    cv::line(plotImage, cv::Point(margin + extendedMargin, height - margin), cv::Point(width + extendedMargin - margin, height - margin), cv::Scalar(0, 0, 0), 2);

    cv::putText(plotImage, "IoU threshold", cv::Point((width + extendedMargin) / 2, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(plotImage, windowName, cv::Point(10, height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    

    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(128, 0, 128),  // Purple
        cv::Scalar(255, 255, 0),  // Cyan
    };

    for (double i = 0.0; i <= 1.0; i += 0.1) {
        int x = static_cast<int>(margin + extendedMargin + i * plotWidth);
        int y = static_cast<int>((height - margin) - i * plotHeight);

        cv::line(plotImage, cv::Point(x, height - margin - 5), cv::Point(x, height - margin + 5), cv::Scalar(0, 0, 0), 1);
        cv::putText(plotImage, std::to_string(i).substr(0, 3), cv::Point(x - 10, height - margin + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        cv::line(plotImage, cv::Point(margin + extendedMargin - 5, y), cv::Point(margin + extendedMargin + 5, y), cv::Scalar(0, 0, 0), 1);
        cv::putText(plotImage, std::to_string(i).substr(0, 3), cv::Point(margin + extendedMargin - 40, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = std::numeric_limits<double>::max();
    double yMax = std::numeric_limits<double>::lowest();

    for (const auto& dataSet : dataSets) {
        for (const auto& point : dataSet) {
            if (point.first < xMin) xMin = point.first;
            if (point.first > xMax) xMax = point.first;
            if (point.second < yMin) yMin = point.second;
            if (point.second > yMax) yMax = point.second;
        }
    }

    for (size_t i = 0; i < dataSets.size(); ++i) {
        const auto& dataSet = dataSets[i];
        for (size_t j = 0; j < dataSet.size(); ++j) {
            int x = static_cast<int>(margin + extendedMargin + ((dataSet[j].first - xMin) / (xMax - xMin)) * plotWidth);
            int y = static_cast<int>((height - margin) - ((dataSet[j].second - yMin) / (yMax - yMin)) * plotHeight);
            cv::circle(plotImage, cv::Point(x, y), 3, colors[i % colors.size()], -1);
            if (j > 0) {
                int prevX = static_cast<int>(margin + extendedMargin + ((dataSet[j - 1].first - xMin) / (xMax - xMin)) * plotWidth);
                int prevY = static_cast<int>((height - margin) - ((dataSet[j - 1].second - yMin) / (yMax - yMin)) * plotHeight);
                cv::line(plotImage, cv::Point(prevX, prevY), cv::Point(x, y), colors[i % colors.size()], 1);
            }
        }
    }

    int legendX = width - 150;
    int legendY = 20;
    int legendBoxSize = 10;
    int legendSpacing = 5;

    for (size_t i = 0; i < dataSets.size(); ++i) {
        cv::rectangle(plotImage, cv::Point(legendX, legendY + i * (legendBoxSize + legendSpacing)), 
                    cv::Point(legendX + legendBoxSize, legendY + (i + 1) * legendBoxSize + i * legendSpacing), 
                    colors[i % colors.size()], cv::FILLED);
        cv::putText(plotImage, "Group " + std::to_string(i + 1), 
                    cv::Point(legendX + legendBoxSize + legendSpacing, legendY + (i + 1) * legendBoxSize + i * legendSpacing - 2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    std::cout << "writing image..." << std::endl;

    cv::imwrite("./" + windowName + ".png", plotImage);
}

int main(int argc, char* argv[]) {

    const cv::String keys = 
    "{type t |'default'|}"
    "{@input |./img_dataset/|}"
    "{@output |./detections/|}";

    cv::CommandLineParser parser(argc, argv, keys);

    std::string operation_type = parser.get<cv::String>("type");

    operation_type = cv::toLowerCase(operation_type);

    if(operation_type == "det"){

        fs::path input_folder = parser.get<cv::String>("@input"); //"../prj.cw/img_dataset/";
        fs::path output_folder = parser.get<cv::String>("@output"); //"../prj.cw/detections/";

        if (!fs::exists(input_folder)) {
            throw std::invalid_argument("Could not find specified input directory!");
        }

        if (!fs::exists(output_folder)) {
            fs::create_directory(output_folder);
        }
        

        std::cout << "Loading images..." << std::endl << std::endl;
        std::vector<std::pair<cv::Mat, std::string>> pics = get_pictures_from_input_dir(input_folder);
        std::cout << "Images have been successfully loaded!" << std::endl << std::endl;

        
        for(auto& elem: pics){

            std::cout << "Processing image " << elem.second << std::endl << std::endl;
            std::cout << "\t\tThe image's size is: " << elem.first.rows << " x " << elem.first.cols << std::endl << std::endl;
            
            std::cout << "\t\tUsing Canny to extract edges..." << std::endl << std::endl;
            cv::Mat grayscale, edges;
            cv::cvtColor(elem.first, grayscale, cv::COLOR_RGB2GRAY);
            int canny_thresh = 70;
            edges = extract_edges(grayscale, canny_thresh);

            std::cout << "\t\tApplying morphological filtering..." << std::endl << std::endl;
            cv::Mat dst;
            dst = cv::Scalar::all(0);
            grayscale.copyTo(dst, edges);
            cv::Mat filtered = perform_morphology_and_component_filtering(dst);


            
            int pyr_counter = 0;

            while(edges.rows > size_threshold || edges.cols > size_threshold){
                cv::pyrDown(edges, edges);
                cv::pyrDown(filtered, filtered);
                pyr_counter++;
            }
            int pyr_factor = std::pow(2, pyr_counter);

            std::cout << "\t\tFinding contours..." << std::endl << std::endl;
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(filtered, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat edges_rgb = filtered.clone();
            cv::cvtColor(edges_rgb, edges_rgb, cv::COLOR_GRAY2BGR);

            std::vector<std::vector<cv::Point>> detections_to_filter;

            std::cout << "\t\tPerforming detection..." << std::endl << std::endl;

            for(int i = 0; i < contours.size(); i++){

                
                cv::RotatedRect rect = cv::minAreaRect(contours[i]);
                cv::Rect bounding_box = rect.boundingRect();
                fit_roi_into_image(bounding_box, filtered.rows, filtered.cols);
                if(bounding_box.area() > area_threshold){
                    

                    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
                    std::vector<cv::Vec4f> lines_std;
                    std::vector<std::pair<cv::Vec4f, bool>> marked_lines;

                    lsd->detect(edges(bounding_box), lines_std);

                    for(cv::Vec4f& line: lines_std){          
                        marked_lines.push_back({line, false});
                    }

                    std::vector<std::pair<cv::Vec4f, cv::Vec4f>> l_shape_pattern_candidates = find_l_shape_finder_patterns(marked_lines);
                    std::vector<std::pair<cv::Vec4f, cv::Vec4f>> l_shape_candidates;

                    
                    

                    for(int i = 0; i < l_shape_pattern_candidates.size(); i++){
                        cv::Vec4f first_line = l_shape_pattern_candidates[i].first;
                        cv::Vec4f second_line = l_shape_pattern_candidates[i].second;

                        double length_first = distance(cv::Point2f(first_line[0], first_line[1]), cv::Point2f(first_line[2], first_line[3]));
                        double length_second = distance(cv::Point2f(second_line[0], second_line[1]), cv::Point2f(second_line[2], second_line[3]));

                        double length_ratio = std::max(length_first, length_second) / std::min(length_first, length_second);
                        if(length_ratio <= 2){
                            l_shape_candidates.push_back({first_line, second_line});
                        }
                    }

                    for(int i = 0; i < l_shape_candidates.size(); i++){

                        cv::Vec4f first_line = l_shape_candidates[i].first;
                        cv::Vec4f second_line = l_shape_candidates[i].second;
                        cv::Mat display = edges.clone();
                        cv::Mat edges_16bit;
                        edges(bounding_box).convertTo(edges_16bit, CV_16U);


                        cv::Point2f intersection = get_intersection_point(first_line, second_line);

                        first_line = extend_line_to_a_point(first_line, intersection);
                        second_line = extend_line_to_a_point(second_line, intersection);
                        std::vector<cv::Vec4f> examined_lines = {first_line, second_line};
                        // lsd->drawSegments(elem.first, examined_lines);
                        // //cv::circle(display, intersection, 1, cv::Scalar(255, 0, 0), -1);

                        // cv::imshow("examined lines", elem.first);
                        // cv::waitKey(0);
                        std::vector<cv::Point> possible_detection = detect_dashed_border(first_line, second_line, intersection, edges_16bit);

                        if(!possible_detection.empty()){

                            for(auto& elem: possible_detection){
                                elem.x += bounding_box.x;
                                elem.y += bounding_box.y;
                            }
                            detections_to_filter.push_back(possible_detection);
                        }  
                    }
                    
                }

            }


            std::cout << "\t\tFiltering duplicates..." << std::endl << std::endl;

            std::vector<std::vector<cv::Point>> filtered_detections = filter_detections(detections_to_filter, iou_filter_threshold);

            std::cout << "\t\tDrawing bounds..." << std::endl << std::endl;

            for(auto& detection: filtered_detections){

                if(pyr_factor > 1){
                    for(cv::Point& point: detection){
                        point.x*=pyr_factor;
                        point.y*=pyr_factor;
                    }
                }
                cv::polylines(elem.first, detection, true, cv::Scalar(0, 0, 255), 3);
            }
            

            std::cout << "\t\tWriting the image in the output folder..." << std::endl << std::endl;
            cv::imwrite(output_folder.string() + "detection_" + elem.second, elem.first);

            std::cout << "\t\tComposing json with detection info..." << std::endl << std::endl;
            float img_area = elem.first.rows * elem.first.cols;
            compose_detections_json(filtered_detections, elem.second, output_folder, img_area);

        }

        std::cout << "All images have been processed, check the output folder for the result!" << std::endl << std::endl;

    }
    else if(operation_type == "val"){

        

        std::cout << "Loading jsons..." << std::endl << std::endl;

        fs::path input_folder_gt = parser.get<cv::String>("@input");
        fs::path input_folder_det = parser.get<cv::String>("@output");

        if (!fs::exists(input_folder_gt)) {
            std::cerr << "Could not find specified ground truth data directory!";
            throw std::invalid_argument("Could not find specified ground truth data directory!");
        }
        if (!fs::exists(input_folder_det)) {
            std::cerr << "Could not find specified detections data directory!";
            throw std::invalid_argument("Could not find specified detections data directory!");
        }

        std::vector<std::string> gt_jsons, det_jsons;
        

        for (const auto& file : fs::directory_iterator(input_folder_gt)) {
            if (!fs::is_directory(file) && fs::path(file).extension() == ".json") {
                fs::path json = fs::path(file);
                gt_jsons.push_back(json.string());
            }
        }

        for (const auto& file : fs::directory_iterator(input_folder_det)) {
            if (!fs::is_directory(file) && fs::path(file).extension() == ".json") {
                fs::path json = fs::path(file);
                det_jsons.push_back(json.string());
            }
        }

        std::sort(gt_jsons.begin(), gt_jsons.end());
        std::sort(det_jsons.begin(), det_jsons.end());

        std::cout << "Jsons have been successfully loaded!" << std::endl << std::endl;

        std::cout << "\t\tStarting evaluation..." << std::endl << std::endl;

        std::vector<std::tuple<double, cv::String, cv::String>> size_comparison_data;
        

        for(int i = 0; i < gt_jsons.size(); i++){

            float area;

            std::vector<std::vector<cv::Point>> gt_polygons, det_polygons;

            gt_polygons = extract_data_from_json(gt_jsons[i], area);
            det_polygons = extract_data_from_json(det_jsons[i], area);

            if(gt_polygons.size() == 1){
                double contour_area = cv::contourArea(gt_polygons[0]);
                double value =  contour_area / area;
                size_comparison_data.push_back(std::make_tuple(value, gt_jsons[i], det_jsons[i]));
            }

        }

        std::sort(size_comparison_data.begin(), size_comparison_data.end(), [](const std::tuple<double, cv::String, cv::String>& a, const std::tuple<double, cv::String, cv::String>& b) {
        return std::get<0>(a) < std::get<0>(b);});


        int step = std::round((double)size_comparison_data.size() / 6);

        std::vector<std::vector<std::vector<double>>> data_to_plot;

        for(int i = 0; i < size_comparison_data.size(); i+=step){
            std::vector<cv::String> gt_j, det_j;
            for(int j = i; j < i + step; j++){
                if(j >= size_comparison_data.size()) break;
                gt_j.push_back(std::get<1>(size_comparison_data[j]));
                det_j.push_back(std::get<2>(size_comparison_data[j]));
            }
            std::vector<std::vector<double>> pr_rec_f1 = calculate_precision_recall_f1(gt_j, det_j);
            data_to_plot.push_back(pr_rec_f1);

        }

        std::vector<std::vector<std::pair<double, double>>> precision_curves, recall_curves, f1_curves;

        for(auto plot_data: data_to_plot){
            std::vector<std::pair<double, double>> precision, recall, f1;
            for(auto values: plot_data){
                precision.push_back(std::make_pair(values[0], values[1]));
                recall.push_back(std::make_pair(values[0], values[2]));
                f1.push_back(std::make_pair(values[0], values[3]));
            }
            precision_curves.push_back(precision);
            recall_curves.push_back(recall);
            f1_curves.push_back(f1);

        }

        plotMultipleGraphs(precision_curves, "precision");
        plotMultipleGraphs(recall_curves, "recall");
        plotMultipleGraphs(f1_curves, "f1");

        std::vector<std::vector<double>> precision_recall_f1 = calculate_precision_recall_f1(gt_jsons, det_jsons);
        

        for(auto& data: precision_recall_f1){
            std::cout << "IoU threshold = " << std::fixed << std::setprecision(2) << data[0] << std::endl;
            std::cout << "\tPrecision: " << std::fixed << std::setprecision(2) << data[1] << std::endl;
            std::cout << "\tRecall: " << std::fixed << std::setprecision(2) << data[2] << std::endl;
            std::cout << "\tF1 score: " << std::fixed << std::setprecision(2) << data[3] << std::endl << std::endl;
        }

    }

}
