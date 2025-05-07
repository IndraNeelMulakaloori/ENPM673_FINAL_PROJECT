from setuptools import setup

package_name = 'paper_direction_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Detect paper orientation and center',
    license='MIT',
    entry_points={
        'console_scripts': [
            'paper_direction_node = paper_direction_pkg.paper_direction_node:main',
            'paper_follower_node = paper_direction_pkg.paper_follower_node:main',
        ],
    },
)

