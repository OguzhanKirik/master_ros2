#!/usr/bin/env python3

"""
This script defines a ROS2 node that integrates with OpenAI's language model to interact with a ROS 2 system.
The node subscribes to a topic and processes incoming messages using an AI agent with predefined tools.
Falls back to basic mode if OpenAI API has issues.

Classes:
    ROS2AIAgent(Node): A ROS2 node that subscribes to a topic and uses an AI agent to process messages.

Methods:
    prompt_callback(msg: String): Callback function to process incoming messages.
    list_topics() -> str: Lists all available ROS 2 topics.
    list_nodes() -> str: Lists all running ROS 2 nodes.
    list_services() -> str: Lists all available ROS 2 services.
    list_actions() -> str: Lists all available ROS 2 actions.
    main(args=None): Initializes and spins the ROS2 node.
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

# Handle Langchain imports gracefully
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
        super().__init__('ros2_ai_agent_tools')
        self.get_logger().info('ROS2 AI Agent Tools has been started')
        
        # Load configuration
        self.load_config()
        
        # Check what mode we can run in
        self.openai_available = bool(os.environ.get('OPENAI_API_KEY'))
        self.ai_mode_active = False
        
        if LANGCHAIN_AVAILABLE and self.openai_available:
            success = self.setup_ai_mode()
            if not success:
                self.setup_basic_mode()
        else:
            self.setup_basic_mode()

        # Create the subscriber for prompts
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            10
        )
        
        self.get_logger().info("Subscribed to 'prompt' topic")
        self.log_usage_info()

    def load_config(self):
        """Load configuration from file or environment"""
        try:
            share_dir = get_package_share_directory('ros2_basic_agent')
            config_dir = os.path.join(share_dir, 'config', 'openai.env')
            
            if os.path.exists(config_dir):
                load_dotenv(Path(config_dir))
                self.get_logger().info(f"Loaded config from: {config_dir}")
            else:
                load_dotenv()
                self.get_logger().warn(f"Config file not found at {config_dir}, using environment variables")
                
        except Exception as e:
            self.get_logger().warn(f"Could not load config file: {e}, using environment variables")
            if LANGCHAIN_AVAILABLE:
                load_dotenv()

    def setup_ai_mode(self):
        """Setup AI mode with OpenAI - returns True if successful"""
        try:
            self.get_logger().info("Setting up AI mode with OpenAI...")
            
            # Create tools using class methods
            self.list_topics_tool = tool(self.list_topics)
            self.list_nodes_tool = tool(self.list_nodes)
            self.list_services_tool = tool(self.list_services)
            self.list_actions_tool = tool(self.list_actions)

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a ROS 2 system information assistant.
                You can check ROS 2 system status using these commands:
                - list_topics(): List all available ROS 2 topics
                - list_nodes(): List all running ROS 2 nodes
                - list_services(): List all available ROS 2 services
                - list_actions(): List all available ROS 2 actions
                
                Return only the necessary information and results. Be concise and helpful.
                Example:
                Human: Show me all running nodes
                AI: Here are the running ROS 2 nodes: [node list]
                """),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])

            # Setup the toolkit with the decorated class methods
            self.toolkit = [
                self.list_topics_tool,
                self.list_nodes_tool,
                self.list_services_tool,
                self.list_actions_tool
            ]

            # Choose the LLM that will drive the agent
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # Test the connection
            test_response = self.llm.invoke("test")
            self.get_logger().info("OpenAI LLM initialized and tested successfully")

            # Construct the OpenAI Tools agent
            self.agent = create_openai_tools_agent(self.llm, self.toolkit, self.prompt)
            self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)
            
            self.mode = "AI"
            self.ai_mode_active = True
            self.get_logger().info("AI mode setup complete")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "429" in error_msg:
                self.get_logger().error("OpenAI API quota exceeded! You need to add credits to your OpenAI account.")
                self.get_logger().error("Visit: https://platform.openai.com/account/billing")
            elif "401" in error_msg or "unauthorized" in error_msg:
                self.get_logger().error("OpenAI API key is invalid or expired.")
            else:
                self.get_logger().error(f"Failed to initialize AI mode: {e}")
            
            self.get_logger().warn("Falling back to basic mode...")
            return False

    def setup_basic_mode(self):
        """Setup basic mode without AI"""
        self.get_logger().info("Setting up basic mode (no AI required)")
        self.agent_executor = None
        self.mode = "Basic"
        self.ai_mode_active = False
        
        if not LANGCHAIN_AVAILABLE:
            self.get_logger().warn("Langchain not available")
        if not self.openai_available:
            self.get_logger().warn("OpenAI API key not set")

    def list_topics(self) -> str:
        """List all available ROS 2 topics."""
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                topics = [t for t in topics if t.strip()]
                return f"Available ROS 2 topics ({len(topics)} total):\n" + '\n'.join(topics)
            else:
                return f"Error listing topics: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout listing topics"
        except Exception as e:
            return f"Error listing topics: {str(e)}"

    def list_nodes(self) -> str:
        """List all running ROS 2 nodes."""
        try:
            result = subprocess.run(['ros2', 'node', 'list'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                nodes = [n for n in nodes if n.strip()]
                return f"Running ROS 2 nodes ({len(nodes)} total):\n" + '\n'.join(nodes)
            else:
                return f"Error listing nodes: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout listing nodes"
        except Exception as e:
            return f"Error listing nodes: {str(e)}"

    def list_services(self) -> str:
        """List all available ROS 2 services."""
        try:
            result = subprocess.run(['ros2', 'service', 'list'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                services = result.stdout.strip().split('\n')
                services = [s for s in services if s.strip()]
                return f"Available ROS 2 services ({len(services)} total):\n" + '\n'.join(services)
            else:
                return f"Error listing services: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout listing services"
        except Exception as e:
            return f"Error listing services: {str(e)}"

    def list_actions(self) -> str:
        """List all available ROS 2 actions."""
        try:
            result = subprocess.run(['ros2', 'action', 'list'], 
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                actions = result.stdout.strip().split('\n')
                actions = [a for a in actions if a.strip()]
                return f"Available ROS 2 actions ({len(actions)} total):\n" + '\n'.join(actions)
            else:
                return f"Error listing actions: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Timeout listing actions"
        except Exception as e:
            return f"Error listing actions: {str(e)}"

    def process_basic_command(self, command: str) -> str:
        """Process commands in basic mode"""
        command_lower = command.lower().strip()
        
        if "topic" in command_lower:
            return self.list_topics()
        elif "node" in command_lower:
            return self.list_nodes()
        elif "service" in command_lower:
            return self.list_services()
        elif "action" in command_lower:
            return self.list_actions()
        elif "help" in command_lower:
            return """Available commands in basic mode:
            - "topics" - List all topics
            - "nodes" - List all nodes
            - "services" - List all services
            - "actions" - List all actions
            - "help" - Show this help
            
            Note: AI mode unavailable due to OpenAI API issues."""
        else:
            return f"Basic mode: Command '{command}' not recognized. Available: topics, nodes, services, actions, help"

    def prompt_callback(self, msg):
        """Handle incoming prompt messages"""
        try:
            self.get_logger().info(f"Received prompt: {msg.data}")
            
            if self.ai_mode_active and self.agent_executor:
                # Use AI agent
                try:
                    result = self.agent_executor.invoke({"input": msg.data})
                    response = result['output']
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "429" in error_msg:
                        self.get_logger().error("OpenAI quota exceeded, falling back to basic mode")
                        response = self.process_basic_command(msg.data)
                    else:
                        raise e
            else:
                # Use basic command processing
                response = self.process_basic_command(msg.data)
            
            self.get_logger().info(f"Response: {response}")
            
        except Exception as e:
            self.get_logger().error(f'Error processing prompt: {str(e)}')

    def log_usage_info(self):
        """Log usage information"""
        self.get_logger().info(f"Running in {self.mode} mode")
        
        if self.mode == "AI":
            self.get_logger().info("AI Agent ready! Natural language queries supported.")
            self.get_logger().info("Examples:")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"Show me all topics\"}'")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"List running nodes\"}'")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"What services are available?\"}'")
        else:
            self.get_logger().info("Basic mode active. Simple commands only.")
            self.get_logger().info("Examples:")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"topics\"}'")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"nodes\"}'")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"services\"}'")
            self.get_logger().info("  ros2 topic pub /prompt std_msgs/String '{data: \"actions\"}'")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ROS2AIAgent()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down ROS2 AI Agent Tools...")
    except Exception as e:
        print(f"Error starting ROS2 AI Agent Tools: {e}")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
