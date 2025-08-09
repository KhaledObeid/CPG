from ament_index_python.packages import get_package_share_directory
import os

URDF_PATH = os.path.join(get_package_share_directory('go1_description'), 'urdf', 'go1.urdf')