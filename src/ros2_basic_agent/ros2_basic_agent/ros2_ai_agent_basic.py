#!/usr/bin/env python3

"""
This script defines a ROS 2 node that integrates with OpenAI's language model to interact with a ROS 2 system.
The node subscribes to a topic and processes incoming messages using an AI agent equipped with predefined tools.

Classes:
    ROS2AIAgent(Node): A ROS 2 node that uses an AI agent to process messages received on a subscribed topic.

Key Methods:
    - prompt_callback(msg: String): Handles incoming messages and processes them using the AI agent.
    - get_ros_distro() -> str: Retrieves the current ROS distribution name.
    - get_domain_id() -> str: Retrieves the current ROS domain ID.

Main Functionality:
    - Subscribes to a topic named 'prompt' to receive user queries.
    - Uses an AI agent with tools to provide system information about ROS 2, such as the ROS distribution and domain ID.

Dependencies:
    - ROS 2 libraries: rclpy, std_msgs.msg, ament_index_python.packages
    - OpenAI integration: langchain, langchain_openai
    - Environment management: dotenv
    - Utility libraries: os, pathlib, subprocess, typing
"""
import os
import math
import subprocess
from pathlib import Path
from typing import List

from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Langchain imports - handle missing dependencies gracefully
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_openai import ChatOpenAI
    from langchain.tools import BaseTool, StructuredTool, tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from dotenv import load_dotenv
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Langchain dependencies not available: {e}")
    LANGCHAIN_AVAILABLE = False


class ROS2AIAgent(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent')
        self.get_logger().info('ROS2 AI Agent has been started')

        if not LANGCHAIN_AVAILABLE:
            self.get_logger().error("Langchain dependencies not available. Please install them:")
            self.get_logger().error("pip install langchain langchain-openai python-dotenv")
            return

        # Create tools
        self.get_ros_distro_tool = tool(self.get_ros_distro)
        self.get_domain_id_tool = tool(self.get_domain_id)
        self.get_topic_list_tool = tool(self.get_topic_list)
        self.get_node_list_tool = tool(self.get_node_list)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a ROS 2 system information assistant.
            You can check ROS 2 system status using these commands:
            - get_ros_distro(): Get the current ROS distribution name
            - get_domain_id(): Get the current ROS_DOMAIN_ID
            - get_topic_list(): Get list of active ROS2 topics
            - get_node_list(): Get list of active ROS2 nodes
            
            Return only the necessary information and results. Be concise and helpful.
            
            Examples:
            Human: What ROS distribution am I using?
            AI: Current ROS distribution: humble
            
            Human: What is my ROS domain ID?
            AI: Current ROS domain ID: 0
            
            Human: What topics are available?
            AI: [List of topics will be shown]
            """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Load environment variables for OpenAI API
        try:
            share_dir = get_package_share_directory('ros2_basic_agent')
            config_dir = os.path.join(share_dir, 'config', 'openai.env')
            
            if os.path.exists(config_dir):
                load_dotenv(Path(config_dir))
                self.get_logger().info(f"Loaded config from: {config_dir}")
            else:
                # Try to load from default locations
                load_dotenv()  # Load from current directory or environment
                self.get_logger().warn(f"Config file not found at {config_dir}, using environment variables")
                
        except Exception as e:
            self.get_logger().warn(f"Could not load config file: {e}, using environment variables")
            load_dotenv()

        # Setup the toolkit with all tools
        self.toolkit = [
            self.get_ros_distro_tool, 
            self.get_domain_id_tool,
            self.get_topic_list_tool,
            self.get_node_list_tool
        ]

        # Choose the LLM that will drive the agent
        try:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.get_logger().info("OpenAI LLM initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize OpenAI LLM: {e}")
            self.get_logger().error("Make sure OPENAI_API_KEY environment variable is set")
            return

        # Construct the OpenAI Tools agent
        self.agent = create_openai_tools_agent(self.llm, self.toolkit, self.prompt)

        # Create an agent executor
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)

        # Create the subscriber for prompts
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            10
        )
        
        self.get_logger().info("Subscribed to 'prompt' topic")
        self.get_logger().info("Send messages to /prompt topic to interact with the AI agent")

    def get_ros_distro(self) -> str:
        """Get the current ROS distribution name."""
        try:
            ros_distro = os.environ.get('ROS_DISTRO')
            if ros_distro:
                return f"Current ROS distribution: {ros_distro}"
            else:
                return "ROS distribution environment variable (ROS_DISTRO) not set"
        except Exception as e:
            return f"Error getting ROS distribution: {str(e)}"

    def get_domain_id(self) -> str:
        """Get the current ROS domain ID."""
        try:
            domain_id = os.environ.get('ROS_DOMAIN_ID', '0')  # Default is 0 if not set
            return f"Current ROS domain ID: {domain_id}"
        except Exception as e:
            return f"Error getting ROS domain ID: {str(e)}"

    def get_topic_list(self) -> str:
        """Get list of active ROS2 topics."""
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                return f"Active ROS2 topics ({len(topics)} total):\n" + '\n'.join(topics)
            else:
                return f"Error getting topic list: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout getting topic list"
        except Exception as e:
            return f"Error getting topic list: {str(e)}"

    def get_node_list(self) -> str:
        """Get list of active ROS2 nodes."""
        try:
            result = subprocess.run(['ros2', 'node', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                return f"Active ROS2 nodes ({len(nodes)} total):\n" + '\n'.join(nodes)
            else:
                return f"Error getting node list: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout getting node list"
        except Exception as e:
            return f"Error getting node list: {str(e)}"

    def prompt_callback(self, msg):
        """Handle incoming prompt messages and process them with the AI agent"""
        try:
            self.get_logger().info(f"Received prompt: {msg.data}")
            result = self.agent_executor.invoke({"input": msg.data})
            self.get_logger().info(f"AI Response: {result['output']}")
        except Exception as e:
            self.get_logger().error(f'Error processing prompt: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ROS2AIAgent()
        
        if not LANGCHAIN_AVAILABLE:
            print("Cannot start AI agent without required dependencies.")
            return
        
        # Check if OpenAI API key is available
        if not os.environ.get('OPENAI_API_KEY'):
            node.get_logger().error("OPENAI_API_KEY environment variable not set!")
            node.get_logger().error("Please set it with: export OPENAI_API_KEY='your-api-key'")
            return
        
        node.get_logger().info("ROS2 AI Agent is ready. Send messages to /prompt topic.")
        node.get_logger().info("Example commands:")
        node.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"What ROS distribution am I using?\"}'")
        node.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"List all active topics\"}'")
        node.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"Show me all running nodes\"}'")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down ROS2 AI Agent...")
    except Exception as e:
        print(f"Error starting ROS2 AI Agent: {e}")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
