import os
from pathlib import Path

from typing import List, Tuple
import numpy as np
from PIL import Image
import click
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


def load_intrinsics(intrinsics_path: str) -> np.ndarray:
    """
    Load camera intrinsics from a file.
    """
    intrinsics = np.load(intrinsics_path)
    return intrinsics


def combine_rgb_depth(rgb_path: str, depth_path: str) -> np.ndarray:
    """
    Combine RGB and Depth images into a single image for processing.
    They may be different sizes, so the depth image is resized to match the RGB image.
    """
    img = Image.open(rgb_path)
    img = img.resize((img.width//2, img.height//2), Image.BILINEAR)
    rgb = np.array(img)
    
    # Depth is a floating point numpy array with values in meters
    # and is saved like that too, so use np.load(depth_path)
    depth = np.load(depth_path)

    # Resize the depth matrix to match the RGB image
    depth = np.array(Image.fromarray(depth).resize(rgb.shape[:2][::-1], Image.NEAREST))
    # Add a channel dimension to the depth image
    depth = np.expand_dims(depth, axis=2)

    # Stack the RGB and Depth images along the channels axis
    combined = np.concatenate([rgb, depth], axis=2)

    return combined


def visualise_rgbd_matplotlib(rgbd_image: np.ndarray):
    """
    Visualise an RGBD image.
    """
    rgb, depth = np.split(rgbd_image, [3], axis=2)
    depth = depth.squeeze()

    # Visualise the RGB and Depth images, side by side in the same figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb/255.0)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    # Display the depth image with a red colormap
    depth = np.clip(depth, 0, 30)
    depth_image = axes[1].imshow(depth, cmap="Reds")
    plt.colorbar(depth_image, ax=axes[1], orientation="horizontal")
    axes[1].set_title("Depth")
    axes[1].axis("off")
    plt.show()


def visualise_rgbd_o3d(rgbd_image: np.ndarray):
    """
    Visualise an RGBD image using Open3D.
    """
    rgb = rgbd_image[:, :, :3].astype(np.uint8)
    depth = rgbd_image[:, :, 3]
    max_depth = 60
    depth = np.clip(depth, 0, max_depth)
    depth = (depth /float(max_depth) * 255).astype(np.uint8)
    depth = o3d.geometry.Image(depth)
    rgb = o3d.geometry.Image(rgb)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )

    # Flip it, otherwise the point cloud will be upside down
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Visualise the point cloud
    o3d.visualization.draw_geometries([pcd])


def extract_translation_from_4x4_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:3, 3]


def o3d_rgbd_vo(
    rgdb_a: np.ndarray,
    rgdb_b: np.ndarray,
    intrinsic_matrix: np.ndarray,
):
    """
    Perform RGBD Odometry using Open3D.

    Arguments:
    rgdb_a: np.ndarray -- The first RGBD image as a numpy array.
    rgdb_b: np.ndarray -- The second RGBD image as a numpy array.
    intrinsic_matrix: np.ndarray -- The intrinsic matrix for the camera.
    """
    rgb_a = rgdb_a[:, :, :3].astype(np.uint8)
    depth_a = rgdb_a[:, :, 3]
    rgb_b = rgdb_b[:, :, :3].astype(np.uint8)
    depth_b = rgdb_b[:, :, 3]

    depth_trunc = 100.0

    depth_a = o3d.geometry.Image(np.array(depth_a))
    rgb_a = o3d.geometry.Image(rgb_a)
    rgbd_a = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_a, depth_a, depth_scale=1.0, depth_trunc=depth_trunc)

    depth_b = o3d.geometry.Image(np.array(depth_b))
    rgb_b = o3d.geometry.Image(rgb_b)
    rgbd_b = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_b, depth_b, depth_scale=1.0, depth_trunc=depth_trunc)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = intrinsic_matrix

    # These options go to the SLAM algorithm
    min_depth_meters = 1.0
    max_depth_meters = depth_trunc-1
    max_depth_diff_meters = 10.0
    option = o3d.pipelines.odometry.OdometryOption(
        min_depth=min_depth_meters,
        max_depth=max_depth_meters,
        max_depth_diff=max_depth_diff_meters,
    )
    odo_init = np.identity(4)

    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_a, rgbd_b, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        # o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    return success, trans


def o3d_rgbd_sequence(
    rgbd_image_list: List[np.ndarray],
    intrinsic_matrix: np.ndarray,
    use_tqdm: bool = True
):
    """
    The output is one shorter than the input
    """
    yaw_list = []
    distance_list = []
    success_list = []
    range_generator = tqdm.tqdm(range(len(rgbd_image_list) - 1)) if use_tqdm else range(len(rgbd_image_list) - 1)
    for i in range_generator:
        success, trans = o3d_rgbd_vo(rgbd_image_list[i], rgbd_image_list[i+1], intrinsic_matrix)
        if success:
            yaw, distance = extract_info_from_matrix(trans)
        else:
            yaw, distance = 0, 0
        print(f"Frame {i}: Success: {success}, Yaw: {np.degrees(yaw):.2f} degrees, Distance: {distance:.2f} meters")
        success_list.append(success)
        yaw_list.append(yaw)
        distance_list.append(distance)
    return success_list, yaw_list, distance_list        
    

def odom_extract_scipy_minimize(points_3d, points_2d, intrinsic_matrix):
    """
    Extracts the yaw and distance from a sequence of 3D points and 2D points using scipy minimize.
    """

    def reprojection_error(rvec, tvec, points_3d, points_2d):
        # Project the 3D points to 2D
        projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
        residual = points_2d - projected_points
        return np.linalg.norm(residual)

    def solve_pnp(points_3d, points_2d):
        # Initial guess
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        x0 = np.concatenate([rvec, tvec])
        # Optimize
        def cost_function(x):
            rvec = x[:3].copy()
            tvec = x[3:].copy()
            return reprojection_error(rvec, tvec, points_3d, points_2d)
        res = minimize(cost_function, x0,
                            method='l-bfgs-b',
                            options={'xatol': 1e-8, 'disp': False})
        return res
    
    res = solve_pnp(points_3d, points_2d)
    success = res.success
    rvec = res.x[:3]
    tvec = res.x[3:]
    # Calculate the transformation matrix
    frame_matrix = np.eye(4)
    frame_matrix[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    frame_matrix[:3, 3] = tvec.flatten()
    yaw, distance = extract_info_from_matrix(frame_matrix)
    return success, yaw, distance


def odom_extract_opencv_pnpransac(points_3d, points_2d, intrinsic_matrix):
    """
    Extracts the yaw and distance from a sequence of 3D points and 2D points using OpenCV solvePnPRansac.
    """
    # Convert the points to the correct format
    points_3d = points_3d.astype(np.float32)
    points_2d = points_2d.astype(np.float32)
    # Calculate the camera pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, intrinsic_matrix, None, flags=cv2.SOLVEPNP_EPNP)
    # Calculate the transformation matrix
    frame_matrix = np.eye(4)
    frame_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    frame_matrix[:3, 3] = tvec.flatten()
    yaw, distance = extract_info_from_matrix(frame_matrix)
    return success, yaw, distance


def hybrid_rgbd_odometry(
    rgbd_image_list: List[np.ndarray],
    intrinsic_matrix: np.ndarray,
    use_tqdm: bool = True,
    backward: bool = False
):
    """
    This function does a hybrid VO solution using the color and depth information,
    specifically:
    1. Use opencv good features to track in first frame
    2. Use the depth image to calculate the 3D points
    3. Find the good features in the subsequent frames
    4. Use opencv to calculate the 3D camera position of each frame relative to the 3D points
    """
    if backward:
        success_list, yaw_list, distance_list = hybrid_rgbd_odometry(rgbd_image_list[::-1], intrinsic_matrix, use_tqdm=use_tqdm, backward=False)
        return success_list[::-1], [-y for y in yaw_list[::-1]], distance_list[::-1]
    ## 1. Use opencv good features to track in first frame
    # Get the image
    rgb_0  = rgbd_image_list[0][:, :, :3].astype(np.uint8)
    # Convert to grayscale
    gray_0 = cv2.cvtColor(rgb_0, cv2.COLOR_BGR2GRAY)
    # Find the good features
    p0 = cv2.goodFeaturesToTrack(gray_0, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    ## 2. Use the depth image and the intrinsic matrix to calculate the 3D points at each good feature to track

    # Inverse of the intrinsic matrix
    K_inv = np.linalg.inv(intrinsic_matrix)

    ## 2. Use the depth image and the intrinsic matrix to calculate the 3D points at each good feature to track
    min_depth = 1
    max_depth = 250
    depth_0 = rgbd_image_list[0][:, :, 3]
    # Get the 3D points
    points_3d = []
    for i in range(p0.shape[0]):
        u, v = p0[i, 0]
        z = depth_0[int(v), int(u)]
        if z > min_depth:  # To avoid invalid depth values
            # Transform pixel coordinates to normalized camera coordinates
            pixel_coords = np.array([u, v, 1])
            normalized_coords = K_inv.dot(pixel_coords) * z
            x = normalized_coords[0]
            y = normalized_coords[1]
            points_3d.append([x, y, z])
    points_3d = np.array(points_3d)
    # Remove any points with a depth less than min and more than max depth
    valid_points_mask = (points_3d[:, 2] > min_depth) & (points_3d[:, 2] < max_depth)
    points_3d = points_3d[valid_points_mask]
    p0 = p0[valid_points_mask]

    ## 3. Find the good features in the subsequent frames
    success_list = []
    yaw_list = []
    distance_list = []
    range_generator = tqdm.tqdm(range(1, len(rgbd_image_list))) if use_tqdm else range(1, len(rgbd_image_list))
    distance_prev = 0.0
    yaw_prev = 0.0
    for i in range_generator:
        # Get the image
        rgb = rgbd_image_list[i][:, :, :3].astype(np.uint8)
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # Find the good features in this frame
        p_frame, st, err = cv2.calcOpticalFlowPyrLK(gray_0, gray, p0, None)
        st_flat = st.flatten()
        ## 4. Use opencv to calculate the 3D camera position of each frame relative to the 3D points
        # Calculate the camera position using solvePnP
        try:
            # success, yaw_from_start, distance_from_start = odom_extract_scipy_minimize(points_3d, p_frame, intrinsic_matrix)
            # Select good points 
            good_new = p_frame[st_flat == 1] 
            good_3d = points_3d[st_flat == 1]
            success, yaw_from_start, distance_from_start = odom_extract_opencv_pnpransac(good_3d, good_new, intrinsic_matrix)

            distance = distance_from_start - distance_prev
            distance_prev = distance_from_start 
            yaw = yaw_from_start - yaw_prev
            yaw_prev = yaw_from_start
        except Exception as e:
            print(f"Failed to calculate odometry: {e}")
            success, yaw, distance = False, 0, 0

        print(f"Frame {i}: Success: {success}, Yaw: {np.degrees(yaw):.2f} degrees, Distance: {distance:.2f} meters")
        success_list.append(success)
        yaw_list.append(yaw)
        distance_list.append(distance)
        
    return success_list, yaw_list, distance_list  


def extract_info_from_matrix(matrix: np.ndarray):
    """
    Extract translation and rotation information from a 4x4 transformation matrix.
    """
    # The coordinate system is right handed and y is up
    translation = extract_translation_from_4x4_matrix(matrix)
    # 1D translation distance directly forward
    delta_distance_m = np.abs(translation[2])
    rotation = Rotation.from_matrix(
        matrix[:3, :3]
    )
    # Apply the rotation to a vector that points in the direction of the camera
    # and calculate the change in yaw angle
    direction = rotation.apply([0, 0, 1])
    yaw_rads = np.arctan2(direction[0], direction[2])
    return yaw_rads, delta_distance_m


@click.command()
@click.argument('rgb_path_a', type=click.Path(exists=True))
@click.argument('depth_path_a', type=click.Path(exists=True))
@click.argument('rgb_path_b', type=click.Path(exists=True))
@click.argument('depth_path_b', type=click.Path(exists=True))
def single_image_pair(rgb_path_a, depth_path_a, rgb_path_b, depth_path_b):
    """
    Combine RGB and Depth images into a single image for processing.
    """
    rgbd_image_a = combine_rgb_depth(rgb_path_a, depth_path_a)
    rgbd_image_b = combine_rgb_depth(rgb_path_b, depth_path_b)
    # visualise_rgbd_o3d(rgbd_image)
    intrinsics = load_intrinsics("demo/1500_intrinsics.npy")
    success, trans = o3d_rgbd_vo(rgbd_image_a, rgbd_image_b, intrinsics)
    if success:
        yaw, distance = extract_info_from_matrix(trans)
        print(f"Yaw: {np.degrees(yaw):.2f} degrees, Distance: {distance:.2f} meters")
    else:
        print("Failed to calculate odometry")


def load_depth_names(depth_path: str, run_hybrid: bool, batch_size: int) -> List[np.ndarray]:
    """
    Load a list of depth images from a directory.
    """
    # Get the list of depth images first
    depth_names = []
    for file in os.listdir(depth_path):
        if file.endswith(".npy") and not file.endswith("intrinsics.npy"):
            depth_names.append(os.path.join(depth_path, file))
    # Sort depth names by number
    depth_names = sorted(depth_names, key=lambda x: int(os.path.basename(x).split(".")[0]))
    # If running hybrid, take only depth names that are spaced by batch size
    final_depth_names = []
    if run_hybrid:
        for d in depth_names:
            number = int(os.path.basename(d).split(".")[0])
            if number % batch_size == 0:
                final_depth_names.append(d)
    print(f"Loaded {len(final_depth_names)} depth names")
    return final_depth_names


def load_rgb_names(rgb_path: str) -> List[np.ndarray]:
    """
    Load a list of RGB images from a directory.
    """
    # Get the list of RGB images first
    rgb_names = []
    for file in os.listdir(rgb_path):
        if file.endswith(".jpg"):
            rgb_names.append(os.path.join(rgb_path, file))
    # Sort RGB names by number
    rgb_names = sorted(rgb_names, key=lambda x: int(os.path.basename(x).split(".")[0]))
    print(f"Loaded {len(rgb_names)} RGB images")
    return rgb_names


def filter_rgb_to_depth(rgb_names: List[str], depth_names: List[str], batch_size: int) -> Tuple[List[str], List[str]]:
    """
    Filter the RGB images to only include those that have corresponding depth images.
    """
    # Get depth numbers
    depth_numbers = [int(os.path.basename(x).split(".")[0]) for x in depth_names]
    depth_numbers = sorted(depth_numbers)
    new_depth_numbers = []
    # Extend the depth numbers to include the batch size
    for d in depth_numbers:
        for i in range(batch_size):
            if d+i not in depth_numbers:
                new_depth_numbers.append(d+i)
    depth_numbers.extend(new_depth_numbers)
    depth_numbers = sorted(depth_numbers)
    # Filter RGB images to only include those that have corresponding depth images
    rgb_names_filtered = [x for x in rgb_names if int(os.path.basename(x).split(".")[0]) in depth_numbers]
    assert len(rgb_names_filtered) > 0 , "No RGB images found that have corresponding depth images"

    rgb_names_filtered = sorted(rgb_names_filtered, key=lambda x: int(os.path.basename(x).split(".")[0]))
    return rgb_names_filtered


def load_batch(
    rgb_names: List[str],
    depth_names: List[str]
):
    # Load the depth and RGB images
    rgbd_image_list = []
    for image_name, depth_name in zip(rgb_names, depth_names):
        rgbd_image = combine_rgb_depth(image_name, depth_name)
        rgbd_image_list.append(rgbd_image)
    return rgbd_image_list
    

def load_hybrid_batch(
    rgb_names: List[str],
    depth_name: str
):
    # Load the depth and RGB images
    rgbd_image_list = []
    for image_name in rgb_names:
        rgbd_image = combine_rgb_depth(image_name, depth_name)
        rgbd_image_list.append(rgbd_image)
    return rgbd_image_list


def run_batch(
    rgbd_image_list: List[np.ndarray],
    delta_times_s: List[float], 
    intrinsics: np.ndarray,
):
    """
    Load the depth and RGB images and run the RGBD SLAM algorithm on a batch of images.
    """

    # Run slam
    # success_list, yaw_list, distance_list = o3d_rgbd_sequence(rgbd_image_list, intrinsics, use_tqdm=False)
    success_list, yaw_list, distance_list = hybrid_rgbd_odometry(rgbd_image_list, intrinsics, use_tqdm=False, backward=False)
    # Calculate the yaw rates and speeds
    yaw_rates = [yaw / delta_time for yaw, delta_time in zip(yaw_list, delta_times_s)]
    speed_mps = [distance / delta_time for distance, delta_time in zip(distance_list, delta_times_s)]

    # Create a pandas dataframe to store the results
    df = pd.DataFrame({
        "success": success_list,
        "yaw_rate": yaw_rates,
        "speed_mps": speed_mps
    })
    return df


@click.command()
@click.argument('rgb_path', type=click.Path(exists=True))
@click.argument('depth_path', type=click.Path(exists=True))
@click.argument('output_csv')
def run_on_directory(rgb_path, depth_path, output_csv: str):
    """
    Run the RGBD SLAM algorithm on a directory of images with o3d_rgbd_sequence
    """
    run_hybrid = True
    batch_size = 10

    # Load what is available
    assert os.path.exists(rgb_path), f"RGB path {rgb_path} does not exist"
    assert os.path.exists(depth_path), f"Depth path {depth_path} does not exist"
    depth_names = load_depth_names(depth_path, run_hybrid, batch_size)
    rgb_names = load_rgb_names(rgb_path)

    # Filter the RGB images to only include those that have corresponding depth images
    rgb_names = filter_rgb_to_depth(rgb_names, depth_names, batch_size)
   
    # Only take images up to max image number
    max_image_number = np.inf
    rgb_names = [x for x in rgb_names if int(os.path.basename(x).split(".")[0]) <= max_image_number]
    image_numbers = [int(os.path.basename(x).split(".")[0]) for x in rgb_names]

    # Calculate the times relative to the first image
    video_frame_rate = 30.0
    times_s = [(x - image_numbers[0]) / video_frame_rate for x in image_numbers]
    delta_times_s = [times_s[i] - times_s[i-1] for i in range(1, len(times_s))]
    delta_times_s.append(delta_times_s[-1])

    # Run batches
    yaw_rates = []
    speed_mps = []
    success_list = []
    if run_hybrid:
        range_generator = tqdm.tqdm(range(0, len(depth_names)))
    else:
        range_generator = tqdm.tqdm(range(0, len(depth_names)))
    for bind in range_generator:
        if run_hybrid:
            rgbd_image_list = load_hybrid_batch(
                rgb_names[bind*batch_size:bind*batch_size+batch_size+1],
                depth_names[bind]
            )
            # Load the intrinsics
            int_name = depth_names[bind].replace(".npy", "_intrinsics.npy")
            intrinsics = load_intrinsics(int_name)
        else:
            rgbd_image_list = load_batch(
                rgb_names[bind:bind+batch_size+1],
                depth_names[bind:bind+batch_size+1]
            )
            # Load the intrinsics
            int_name = depth_names[bind].replace(".npy", "_intrinsics.npy")
            intrinsics = load_intrinsics(int_name)
        batch_df = run_batch(
            rgbd_image_list,
            delta_times_s[bind:bind+batch_size+1],
            intrinsics
        )
        success_list.extend(batch_df["success"])
        yaw_rates.extend(batch_df["yaw_rate"])
        speed_mps.extend(batch_df["speed_mps"])

    # Save
    print(len(image_numbers), len(yaw_rates), len(speed_mps))
    df = pd.DataFrame({
        "frame": image_numbers[1:],
        "yaw_rate": yaw_rates,
        "speed_mps": speed_mps,
        "success": success_list
    })
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    df = pd.read_csv(output_csv)
    image_numbers = df["frame"].values
    yaw_rates = df["yaw_rate"].values
    speed_mps = df["speed_mps"].values
    success_list = df["success"].values

    plt.figure()
    plt.plot(image_numbers, yaw_rates, label="Yaw rate")
    # Plot a low pass filter
    plt.plot(image_numbers, pd.Series(yaw_rates).rolling(window=11).median(), label="Low pass filter")
    plt.legend()
    plt.title("Yaw rate")
    plt.xlabel("Frame")
    plt.ylabel("Degrees per second")

    plt.figure()
    plt.plot(image_numbers, speed_mps, label="Speed")
    plt.plot(image_numbers, pd.Series(speed_mps).rolling(window=11).median(), label="Low pass filter")
    plt.legend()
    plt.title("Distance")
    plt.xlabel("Frame")
    plt.ylabel("Meters per second")
    plt.show()
    


if __name__ == "__main__":
    run_on_directory()
    # Example usage: python3 rgbdslam.py rgb/ depth/ output.csv

    # single_image_pair()
    # Example usage: python3 rgbdslam.py demo/1500.jpg demo/1500.npy demo/1509.jpg demo/1509.npy