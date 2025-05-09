from setuptools import setup

package_name = 'obstacle_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='you@example.com',
    description='ROS2 node for obstacle detection demo',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'obstacle_detect_node = obstacle_detection.obstacle_detect_node:main',
        ],
    },
)
