from setuptools import find_packages, setup

package_name = 'nav2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ram',
    maintainer_email='ram@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ["navig.py = nav2.navig:main",
                            "camera_viewer.py = nav2.camera_viewer:main",
                            "task1c.py = nav2.task1c:main",
                            "camera_viewer_3A.py = nav2.camera_viewer_incorrect_tfs:main",
                            "servoing2.py = nav2.servoing2:main",
                            "lidar.py = nav2.lidar:main",
                            "task2a.py = nav2.task2a:main",

        
        ],
    },
)
