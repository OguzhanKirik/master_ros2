#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <iostream>
#include <thread>

class CartesianPlanning : public rclcpp::Node
{
public:
    CartesianPlanning() : Node("cartesian_planning") {}
    void run();
    void planToGoal(double x, double y, double z);
    void getUserInput();
    
private:
    rclcpp::Node::SharedPtr _node;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    bool should_continue_ = true;
};

void CartesianPlanning::planToGoal(double target_x, double target_y, double target_z) {
    if (!move_group_) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized!");
        return;
    }

    bool tf_found = false;    
    geometry_msgs::msg::TransformStamped t;

    auto tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    std::string fromFrameRel = move_group_->getPlanningFrame();
    std::string toFrameRel = move_group_->getEndEffectorLink();

    RCLCPP_INFO(this->get_logger(), "Getting current robot pose...");
    
    // Wait for transform to be available
    while (!tf_found && rclcpp::ok()) {    
        try {
            t = tf_buffer->lookupTransform(fromFrameRel, toFrameRel, tf2::TimePointZero);
            tf_found = true;
        } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s Try again", 
                       toFrameRel.c_str(), fromFrameRel.c_str(), ex.what());
            rclcpp::sleep_for(std::chrono::seconds(1));
        }
    }

    if (!tf_found) {
        RCLCPP_ERROR(this->get_logger(), "Could not get robot pose!");
        return;
    }

    // Create current pose from transform
    geometry_msgs::msg::Pose current_pose;
    current_pose.orientation = t.transform.rotation;
    current_pose.position.x = t.transform.translation.x;
    current_pose.position.y = t.transform.translation.y;
    current_pose.position.z = t.transform.translation.z;

    RCLCPP_INFO(this->get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f", 
                current_pose.position.x, current_pose.position.y, current_pose.position.z);

    // Create target pose
    geometry_msgs::msg::Pose target_pose = current_pose;
    target_pose.position.x = target_x;
    target_pose.position.y = target_y;
    target_pose.position.z = target_z;

    RCLCPP_INFO(this->get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f", 
                target_x, target_y, target_z);

    // Create waypoints for cartesian path
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(current_pose);  // Start from current position
    waypoints.push_back(target_pose);   // Move to target

    // Plan cartesian path
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;
    
    RCLCPP_INFO(this->get_logger(), "Computing cartesian path...");
    
    double fraction = move_group_->computeCartesianPath(
        waypoints, eef_step, jump_threshold, trajectory);
    
    RCLCPP_INFO(this->get_logger(), "Cartesian path planned: %.2f%% achieved", fraction * 100.0);
    
    // Execute if path planning was successful
    if (fraction > 0.8) {
        RCLCPP_INFO(this->get_logger(), "Executing motion...");
        auto result = move_group_->execute(trajectory);
        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "Motion executed successfully!");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Motion execution failed!");
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Cartesian path planning failed. Only %.2f%% achieved", fraction * 100.0);
    }
}

void CartesianPlanning::getUserInput() {
    double x, y, z;
    std::string input;
    
    while (should_continue_ && rclcpp::ok()) {
        std::cout << "\n=== Cartesian Goal Planning ===" << std::endl;
        std::cout << "Enter target coordinates (or 'q' to quit):" << std::endl;
        
        std::cout << "X coordinate: ";
        std::getline(std::cin, input);
        if (input == "q" || input == "quit") {
            should_continue_ = false;
            break;
        }
        try {
            x = std::stod(input);
        } catch (...) {
            std::cout << "Invalid input for X. Please try again." << std::endl;
            continue;
        }
        
        std::cout << "Y coordinate: ";
        std::getline(std::cin, input);
        if (input == "q" || input == "quit") {
            should_continue_ = false;
            break;
        }
        try {
            y = std::stod(input);
        } catch (...) {
            std::cout << "Invalid input for Y. Please try again." << std::endl;
            continue;
        }
        
        std::cout << "Z coordinate: ";
        std::getline(std::cin, input);
        if (input == "q" || input == "quit") {
            should_continue_ = false;
            break;
        }
        try {
            z = std::stod(input);
        } catch (...) {
            std::cout << "Invalid input for Z. Please try again." << std::endl;
            continue;
        }
        
        std::cout << "Planning to: x=" << x << ", y=" << y << ", z=" << z << std::endl;
        
        // Execute the planning
        planToGoal(x, y, z);
        
        std::cout << "Press Enter to continue or 'q' to quit: ";
        std::getline(std::cin, input);
        if (input == "q" || input == "quit") {
            should_continue_ = false;
        }
    }
    
    std::cout << "Exiting..." << std::endl;
    rclcpp::shutdown();
}

void CartesianPlanning::run() {
    _node = rclcpp::Node::make_shared("CartesianPlan");
    
    try {
        // Initialize MoveGroup
        RCLCPP_INFO(this->get_logger(), "Initializing MoveGroup with 'arm' group...");
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(_node, "arm");
        RCLCPP_INFO(this->get_logger(), "MoveGroup initialized successfully!");
        
        // Wait for MoveIt to be ready
        RCLCPP_INFO(this->get_logger(), "Waiting for MoveIt to be ready...");
        rclcpp::sleep_for(std::chrono::seconds(2));
        
        // Go directly to interactive mode
        RCLCPP_INFO(this->get_logger(), "Starting interactive mode...");
        
        // Start user input in a separate thread
        std::thread input_thread(&CartesianPlanning::getUserInput, this);
        
        // Spin the node
        rclcpp::spin(_node);
        
        // Wait for input thread to finish
        if (input_thread.joinable()) {
            input_thread.join();
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error initializing MoveGroup: %s", e.what());
        should_continue_ = false;
    }
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    std::cout << "Starting Cartesian Goal Planning..." << std::endl;
    std::cout << "This program will ask you for target coordinates and move the robot." << std::endl;
    
    CartesianPlanning cp;
    cp.run();
    
    return 0;
}



