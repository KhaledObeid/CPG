from setuptools import setup, find_packages

setup(
    name='quad_cpg',
    version='0.1.0',
    packages=['quad_cpg', 'quad_cpg.env'],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/quad_cpg']),
        ('share/quad_cpg/launch', ['launch/quad_cpg_go1.launch.py']),
    ],
    install_requires=[
        'rclpy',
        'std_msgs',
        'sensor_msgs',
        'gazebo_msgs',
        'tf_transformations',
        'tf2_ros',
    ],
    entry_points={
        'console_scripts': [
            'gazebo_interface = quad_cpg.gazebo_interface:main',
            'cpg_runner = quad_cpg.cpg_runner:main',
        ],
    },
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Hopf oscillator CPG for quadruped control in Gazebo',
    license='MIT',
    tests_require=['pytest'],
)