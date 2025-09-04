#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "panda_moveit_control/srv/cartesian_goal.hpp"

class CartesianPlanningService : public rclcpp::Node
{
public:
    CartesianPlanningService() : Node("cartesian_planning_service") 
    {
        // Initialize TF buffer and listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        RCLCPP_INFO(this->get_logger(), "Cartesian Planning Service node created");
        
        // Don't initialize MoveGroup here - do it in a timer callback
        // to avoid bad_weak_ptr error
        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&CartesianPlanningService::initialize_moveit, this));
    }

private:
    void initialize_moveit()
    {
        // Cancel the timer - we only need to run this once
        init_timer_->cancel();
        
        try {
            RCLCPP_INFO(this->get_logger(), "Initializing MoveGroup with 'arm' group...");
            
            // Now shared_from_this() works because object is fully constructed
            move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "arm");
                
            RCLCPP_INFO(this->get_logger(), "MoveGroup initialized successfully!");
            
            // Now create the service
            service_ = this->create_service<panda_moveit_control::srv::CartesianGoal>(
                "cartesian_goal", 
                std::bind(&CartesianPlanningService::handle_cartesian_goal, this, 
                         std::placeholders::_1, std::placeholders::_2));
            
            RCLCPP_INFO(this->get_logger(), "Cartesian Goal Service ready at /cartesian_goal");
            RCLCPP_INFO(this->get_logger(), "Call with: ros2 service call /cartesian_goal panda_moveit_control/srv/CartesianGoal \"{target_x: 0.3, target_y: 0.1, target_z: 0.5}\"");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize MoveGroup: %s", e.what());
            rclcpp::shutdown();
        }
    }

    geometry_msgs::msg::Pose getCurrentPose()
    {
        geometry_msgs::msg::Pose current_pose;
        
        if (!move_group_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized!");
            return current_pose;
        }

        bool tf_found = false;    
        geometry_msgs::msg::TransformStamped t;

        std::string fromFrameRel = move_group_->getPlanningFrame();
        std::string toFrameRel = move_group_->getEndEffectorLink();

        RCLCPP_INFO(this->get_logger(), "Getting current robot pose...");
        
        // Wait for transform to be available
        int attempts = 0;
        while (!tf_found && rclcpp::ok() && attempts < 5) {    
            try {
                t = tf_buffer_->lookupTransform(fromFrameRel, toFrameRel, tf2::TimePointZero);
                tf_found = true;
            } catch (const tf2::TransformException & ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform %s to %s: %s (attempt %d/5)", 
                           toFrameRel.c_str(), fromFrameRel.c_str(), ex.what(), attempts + 1);
                rclcpp::sleep_for(std::chrono::seconds(1));
                attempts++;
            }
        }

        if (tf_found) {
            // Create current pose from transform
            current_pose.orientation = t.transform.rotation;
            current_pose.position.x = t.transform.translation.x;
            current_pose.position.y = t.transform.translation.y;
            current_pose.position.z = t.transform.translation.z;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Could not get robot pose after 5 attempts!");
        }
        
        return current_pose;
    }

    void handle_cartesian_goal(
        const std::shared_ptr<panda_moveit_control::srv::CartesianGoal::Request> request,
        std::shared_ptr<panda_moveit_control::srv::CartesianGoal::Response> response)
    {
        if (!move_group_) {
            response->success = false;
            response->message = "MoveGroup not initialized yet. Please wait and try again.";
            response->fraction_achieved = 0.0;
            RCLCPP_ERROR(this->get_logger(), "Service called before MoveGroup initialization!");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Received cartesian goal request: x=%.3f, y=%.3f, z=%.3f", 
                   request->target_x, request->target_y, request->target_z);
        
        try {
            // Get current pose
            geometry_msgs::msg::Pose current_pose = getCurrentPose();
            
            // Set current pose in response
            response->current_x = current_pose.position.x;
            response->current_y = current_pose.position.y;
            response->current_z = current_pose.position.z;
            
            RCLCPP_INFO(this->get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f", 
                       current_pose.position.x, current_pose.position.y, current_pose.position.z);
            
            // Create target pose
            geometry_msgs::msg::Pose target_pose = current_pose;
            target_pose.position.x = request->target_x;
            target_pose.position.y = request->target_y;
            target_pose.position.z = request->target_z;

            RCLCPP_INFO(this->get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f", 
                       request->target_x, request->target_y, request->target_z);

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
            
            response->fraction_achieved = fraction;
            
            RCLCPP_INFO(this->get_logger(), "Cartesian path planned: %.2f%% achieved", fraction * 100.0);
            
            // Execute if path planning was successful
            if (fraction > 0.8) {
                RCLCPP_INFO(this->get_logger(), "Executing motion...");
                
                // Set safe execution parameters
                move_group_->setMaxVelocityScalingFactor(0.3);
                move_group_->setMaxAccelerationScalingFactor(0.3);
                
                auto result = move_group_->execute(trajectory);
                if (result == moveit::core::MoveItErrorCode::SUCCESS) {
                    response->success = true;
                    response->message = "Motion executed successfully! Fraction: " + 
                                      std::to_string(fraction * 100.0) + "%";
                    RCLCPP_INFO(this->get_logger(), "Motion executed successfully!");
                } else {
                    response->success = false;
                    response->message = "Motion execution failed with error code: " + 
                                      std::to_string(result.val);
                    RCLCPP_ERROR(this->get_logger(), "Motion execution failed with code: %d", result.val);
                }
            } else {
                response->success = false;
                response->message = "Cartesian path planning failed. Only " + 
                                  std::to_string(fraction * 100.0) + "% achieved";
                RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
            }
            
        } catch (const std::exception& e) {
            response->success = false;
            response->message = "Exception occurred: " + std::string(e.what());
            response->fraction_achieved = 0.0;
            response->current_x = 0.0;
            response->current_y = 0.0;
            response->current_z = 0.0;
            RCLCPP_ERROR(this->get_logger(), "Exception in service handler: %s", e.what());
        }
        
        RCLCPP_INFO(this->get_logger(), "Service call completed. Success: %s, Message: %s", 
                   response->success ? "true" : "false", response->message.c_str());
    }

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Service<panda_moveit_control::srv::CartesianGoal>::SharedPtr service_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr init_timer_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting Cartesian Planning Service...");
    
    auto node = std::make_shared<CartesianPlanningService>();
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}