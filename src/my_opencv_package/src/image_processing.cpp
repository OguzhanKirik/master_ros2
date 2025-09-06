#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <chrono>

class ImageProcesser : public rclcpp::Node {
public:
    ImageProcesser() : Node("image_processor") {
        RCLCPP_INFO(this->get_logger(), "=== Image Processor Node Starting ===");
        
        // Initialize subscription
        RCLCPP_INFO(this->get_logger(), "Creating image subscription on topic: /image_rect");
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_rect", 10,
            std::bind(&ImageProcesser::image_callback, this, std::placeholders::_1));

        // Initialize publishers
        RCLCPP_INFO(this->get_logger(), "Creating image publishers...");
        contours_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/contours/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Contours publisher: /contours/image_rect");

        gray_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/gray/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Grayscale publisher: /gray/image_rect");
            
        blur_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/blur/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Blur publisher: /blur/image_rect");
            
        thresholded_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/thresholded/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Threshold publisher: /thresholded/image_rect");
            
        corner_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/corner/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Corner detection publisher: /corner/image_rect");
            
        keypoints_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/keypoints/image_rect", 1);
        RCLCPP_INFO(this->get_logger(), "  - Keypoints publisher: /keypoints/image_rect");

        // Initialize frame counter
        frame_count_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "=== Image Processor Node Ready ===");
        RCLCPP_INFO(this->get_logger(), "Waiting for images on /image_rect topic...");
        RCLCPP_INFO(this->get_logger(), "Available output topics:");
        RCLCPP_INFO(this->get_logger(), "  - /contours/image_rect (Contour detection)");
        RCLCPP_INFO(this->get_logger(), "  - /gray/image_rect (Grayscale conversion)");
        RCLCPP_INFO(this->get_logger(), "  - /blur/image_rect (Blurred image)");
        RCLCPP_INFO(this->get_logger(), "  - /thresholded/image_rect (Binary threshold)");
        RCLCPP_INFO(this->get_logger(), "  - /corner/image_rect (Harris corner detection)");
        RCLCPP_INFO(this->get_logger(), "  - /keypoints/image_rect (ORB feature detection)");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        frame_count_++;
        
        // Log every 30 frames to avoid spam
        if (frame_count_ % 30 == 1) {
            RCLCPP_INFO(this->get_logger(), "Processing frame #%d (Image: %dx%d, encoding: %s)", 
                       frame_count_, msg->width, msg->height, msg->encoding.c_str());
        }
        
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            
            if (frame_count_ % 30 == 1) {
                RCLCPP_DEBUG(this->get_logger(), "Successfully converted ROS image to OpenCV format");
            }
        }
        catch(cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat processed_image = cv_ptr->image;
        cv::Mat gray_image;
        cv::Mat blur_image;
        cv::Mat thresholded_image;
        cv::Mat corner_image;
        cv::Mat keypoints_img;
        cv::Mat contours_image = processed_image.clone();

        // Start processing timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // 0. Convert to grayscale
        RCLCPP_DEBUG(this->get_logger(), "Step 1/6: Converting to grayscale...");
        cv::cvtColor(processed_image, gray_image, cv::COLOR_BGR2GRAY);

        // 1. Blurring        
        RCLCPP_DEBUG(this->get_logger(), "Step 2/6: Applying blur filter...");
        cv::blur(processed_image, blur_image, cv::Size(15, 15));
            
        // 2. Thresholding        
        RCLCPP_DEBUG(this->get_logger(), "Step 3/6: Applying binary threshold...");
        cv::threshold(gray_image, thresholded_image, 128, 255, cv::THRESH_BINARY);

        // 3. Contour Detection
        RCLCPP_DEBUG(this->get_logger(), "Step 4/6: Detecting contours...");
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresholded_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(contours_image, contours, -1, cv::Scalar(0, 255, 0), 2);
        
        if (frame_count_ % 30 == 1) {
            RCLCPP_INFO(this->get_logger(), "Found %zu contours in current frame", contours.size());
        }
        
        // 4. Feature Detection and Description
        RCLCPP_DEBUG(this->get_logger(), "Step 5/6: Detecting ORB keypoints...");
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(gray_image, cv::noArray(), keypoints, descriptors);
        cv::drawKeypoints(processed_image, keypoints, keypoints_img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (frame_count_ % 30 == 1) {
            RCLCPP_INFO(this->get_logger(), "Detected %zu ORB keypoints", keypoints.size());
        }

        // 5. Corner detection
        RCLCPP_DEBUG(this->get_logger(), "Step 6/6: Detecting Harris corners...");
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(gray_image.size(), CV_32FC1);
        
        // Parameters for Harris corner detection
        const int blockSize = 2;      // Neighborhood size
        const int apertureSize = 3;    // Aperture parameter for the Sobel operator
        double k = 0.04;               // Harris detector free parameter
        
        // Detecting corners
        cv::cornerHarris(gray_image, dst, blockSize, apertureSize, k);
        // Normalize the result
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        // Convert to an 8-bit image to draw on it
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);
        corner_image = processed_image.clone();
        
        int corner_count = 0;
        for (int i = 0; i < dst_norm.rows; i++) {
            for (int j = 0; j < dst_norm.cols; j++) {
                if ((int)dst_norm.at<float>(i, j) > 200) {
                    cv::circle(corner_image, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                    corner_count++;
                }
            }
        }
        
        if (frame_count_ % 30 == 1) {
            RCLCPP_INFO(this->get_logger(), "Detected %d Harris corners", corner_count);
        }
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (frame_count_ % 30 == 1) {
            RCLCPP_INFO(this->get_logger(), "Image processing completed in %ld ms", duration.count());
        }
        
        // Publish images
        RCLCPP_DEBUG(this->get_logger(), "Publishing processed images...");
        auto header = std_msgs::msg::Header();
        header.stamp = this->get_clock()->now();
        header.frame_id = msg->header.frame_id;

        try {
            auto contours_msg = cv_bridge::CvImage(header, "bgr8", contours_image).toImageMsg();
            contours_img_pub_->publish(*contours_msg);
            
            auto gray_msg = cv_bridge::CvImage(header, "mono8", gray_image).toImageMsg();
            gray_img_pub_->publish(*gray_msg);
            
            auto blur_msg = cv_bridge::CvImage(header, "bgr8", blur_image).toImageMsg();
            blur_img_pub_->publish(*blur_msg);
            
            auto thresholded_msg = cv_bridge::CvImage(header, "mono8", thresholded_image).toImageMsg();
            thresholded_img_pub_->publish(*thresholded_msg);
            
            auto corner_msg = cv_bridge::CvImage(header, "bgr8", corner_image).toImageMsg();
            corner_img_pub_->publish(*corner_msg);
            
            auto keypoints_msg = cv_bridge::CvImage(header, "bgr8", keypoints_img).toImageMsg();
            keypoints_img_pub_->publish(*keypoints_msg);
            
            if (frame_count_ % 30 == 1) {
                RCLCPP_INFO(this->get_logger(), "Successfully published all 6 processed images");
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error publishing images: %s", e.what());
        }
        
        // Log statistics every 100 frames
        if (frame_count_ % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "=== Processing Statistics ===");
            RCLCPP_INFO(this->get_logger(), "Total frames processed: %d", frame_count_);
            RCLCPP_INFO(this->get_logger(), "Average processing time: %ld ms", duration.count());
            RCLCPP_INFO(this->get_logger(), "Last frame - Contours: %zu, Keypoints: %zu, Corners: %d", 
                       contours.size(), keypoints.size(), corner_count);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr contours_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr gray_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr blur_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr thresholded_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr corner_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr keypoints_img_pub_;
    
    int frame_count_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting Image Processing Node...");
    RCLCPP_INFO(rclcpp::get_logger("main"), "OpenCV version: %s", CV_VERSION);
    
    auto node = std::make_shared<ImageProcesser>();
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Node created successfully. Spinning...");
    RCLCPP_INFO(rclcpp::get_logger("main"), "Press Ctrl+C to stop the node");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception in main loop: %s", e.what());
    }
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Image Processing Node shutting down...");
    rclcpp::shutdown();
    return 0;
}
