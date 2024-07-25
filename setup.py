from setuptools import setup, find_packages

setup(
    name="rgbdslam",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "Pillow",
        "click",
        "matplotlib",
        "tqdm",
        "pandas",
        "open3d",
        "opencv-python",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "single-image-pair=rgbdslam:cli_single_image_pair",
            "run-on-directory=rgbdslam:cli",
        ],
    },
    author="Hugo Hadfield",
    author_email="hadfield.hugo@gmail.com",
    description="A Python module for RGB-D SLAM using OpenCV, Open3D, and SciPy",
    url="https://github.com/hugohadfield/rgbdslam",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
