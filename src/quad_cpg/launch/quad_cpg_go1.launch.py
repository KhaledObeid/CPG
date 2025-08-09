from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Gazebo launch file (use empty world or custom GO1 world if available)
    gazebo_pkg = get_package_share_directory('gazebo_ros')
    go1_world = os.path.join(get_package_share_directory('go1_gazebo'), 'worlds', 'go1.world')
    world_arg = go1_world if os.path.exists(go1_world) else 'empty.world'

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_arg}.items()
    )

    # CPG nodes
    gazebo_interface_node = Node(
        package='quad_cpg',
        executable='gazebo_interface',
        name='gazebo_interface',
        output='screen',
        parameters=[{'omega': 2.0, 'coupling': 0.1}]
    )

    cpg_runner_node = Node(
        package='quad_cpg',
        executable='cpg_runner',
        name='cpg_runner',
        output='screen',
        parameters=[{'omega': 2.0, 'coupling': 0.1}]
    )

    return LaunchDescription([
        gazebo_interface_node,
        cpg_runner_node,
    ])