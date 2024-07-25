# RGBD SLAM Module

This module provides a comprehensive solution for RGB-D SLAM (Simultaneous Localization and Mapping) using Python. The code integrates several key libraries, such as OpenCV, Open3D, and SciPy, to process RGB-D images, perform visual odometry, and visualize results.

## Features

- **Loading Camera Intrinsics**: Load camera intrinsics from a file.
- **Combining RGB and Depth Images**: Combine RGB and depth images for processing.
- **Visualizing RGB-D Images**: Visualize RGB-D images using Matplotlib and Open3D.
- **Performing RGB-D Odometry**: Perform visual odometry using Open3D.
- **Sequence Processing**: Process sequences of RGB-D images.
- **Hybrid Visual Odometry**: Combine color and depth information for odometry using feature tracking and 3D points.

## Dependencies

Ensure you have the following Python libraries installed:
- numpy
- Pillow
- click
- matplotlib
- tqdm
- pandas
- open3d
- opencv-python
- scipy

You can install these dependencies using pip:
```sh
pip install numpy Pillow click matplotlib tqdm pandas open3d opencv-python scipy
```

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/rgbd-slam.git
    cd rgbd-slam
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### CLI Commands

The module provides a command-line interface (CLI) to process images.

#### Single Image Pair

Process a single pair of RGB and depth images:
```sh
python rgbdslam.py single-image-pair <rgb_path_a> <depth_path_a> <rgb_path_b> <depth_path_b>
```

#### Directory of Images

Process a directory of RGB and depth images:
```sh
python rgbdslam.py run-on-directory <rgb_path> <depth_path> <output_csv>
```

### Example

To run the SLAM algorithm on a directory of images:
```sh
python rgbdslam.py run-on-directory rgb/ depth/ output.csv
```

This command processes all RGB and depth images in the `rgb/` and `depth/` directories and saves the results in `output.csv`.

### Functions

#### `load_intrinsics(intrinsics_path: str) -> np.ndarray`

Load camera intrinsics from a file.

#### `combine_rgb_depth(rgb_path: str, depth_path: str) -> np.ndarray`

Combine RGB and depth images into a single image for processing.

#### `visualise_rgbd_matplotlib(rgbd_image: np.ndarray)`

Visualize an RGBD image using Matplotlib.

#### `visualise_rgbd_o3d(rgbd_image: np.ndarray)`

Visualize an RGBD image using Open3D.

#### `extract_translation_from_4x4_matrix(matrix: np.ndarray) -> np.ndarray`

Extract translation information from a 4x4 transformation matrix.

#### `o3d_rgbd_vo(rgdb_a: np.ndarray, rgdb_b: np.ndarray, intrinsic_matrix: np.ndarray)`

Perform RGBD Odometry using Open3D.

#### `o3d_rgbd_sequence(rgbd_image_list: List[np.ndarray], intrinsic_matrix: np.ndarray, use_tqdm: bool = True)`

Process a sequence of RGB-D images.

#### `odom_extract_scipy_minimize(points_3d, points_2d, intrinsic_matrix)`

Extract yaw and distance using SciPy's minimize function.

#### `odom_extract_opencv_pnpransac(points_3d, points_2d, intrinsic_matrix)`

Extract yaw and distance using OpenCV's solvePnPRansac.

#### `hybrid_rgbd_odometry(rgbd_image_list: List[np.ndarray], intrinsic_matrix: np.ndarray, use_tqdm: bool = True, backward: bool = False)`

Perform hybrid visual odometry using both color and depth information.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please get in contact with Hugo Hadfield, contact details can be found on [his website](https://hh409.user.srcf.net).


