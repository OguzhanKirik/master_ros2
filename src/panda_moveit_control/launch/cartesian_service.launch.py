from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import launch_ros.actions

def generate_launch_description():
    # MoveIt configuration
    moveit_config = MoveItConfigsBuilder("panda").to_moveit_configs()
    
    # Set simulation time globally
    launch_ros.actions.SetParameter(name='use_sim_time', value=True)
    
    # Cartesian planning service node
    cartesian_service_node = Node(
        package="panda_moveit_control",
        executable="cartesian_goal_service",
        name="cartesian_planning_service",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ]
    )

    return LaunchDescription([cartesian_service_node])