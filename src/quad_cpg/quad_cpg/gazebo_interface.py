import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetModelConfiguration
from quad_cpg.env.configs_go1 import URDF_PATH

class GazeboInterface(Node):
    def __init__(self):
        super().__init__('gazebo_interface')
        self.cli = self.create_client(SetModelConfiguration, '/gazebo/set_model_configuration')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_model_configuration service...')
        self.get_logger().info('GazeboInterface node started.')

    def set_model_config(self, model_name, joint_names, joint_positions):
        req = SetModelConfiguration.Request()
        req.model_name = model_name
        req.urdf_param_name = URDF_PATH
        req.joint_names = joint_names
        req.joint_positions = joint_positions
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    node = GazeboInterface()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()