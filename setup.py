from setuptools import setup

package_name = 'py_undist'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mathias',
    maintainer_email='mathiatp@stud.ntnu.no',
    description='Reads images and undistorts them',
    license='My license',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'undistorter = py_undist.undistort_function:main',
        'BEW_maker = py_undist.BEW_function:main',
        'mock_camera = py_undist.camera_function:main'
        ],
    },
)
