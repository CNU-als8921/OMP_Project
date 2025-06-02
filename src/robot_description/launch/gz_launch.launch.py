from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='thruster_bridge',
            arguments=[
                '/box_usv/thruster_L@std_msgs/msg/Float64@ignition.msgs.Double',
                '/box_usv/thruster_R@std_msgs/msg/Float64@ignition.msgs.Double',
                '/world/my_world/model/box_usv/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@ignition.msgs.Image'

            ],
            output='screen'
        ),
    ])