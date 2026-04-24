import copy
import open3d as o3d
import numpy as np


def crop_geometry(pcd):
    """
    Crop a point cloud manually using Open3D visualization.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud to crop.
    """
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with the negative direction of the y-axis")
    print("2) Press 'K' to lock the screen and switch to selection mode")
    print("3) Drag for rectangle selection or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")

    o3d.visualization.draw_geometries_with_editing([pcd])

def draw_registration_result(source, target, transformation, save_pc=True):
    """
    Visualize the registration result of two point clouds.
    Args:
        :param source: (open3d.geometry.PointCloud): Source point cloud.
        :param target: (open3d.geometry.PointCloud): Target point cloud.
        :param transformation: 4x4 transformation matrix.
        :param save_pc: indicate if save point cloud or not.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if save_pc:
        pc_combined = source_temp+target_temp
        o3d.io.write_point_cloud("registration_example.ply", pc_combined)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    """
    Allow user to pick points from a point cloud in a visualizer.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.

    Returns:
        list: Indices of the picked points.
    """
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

    return vis.get_picked_points()

def register_via_correspondences(source, target, source_points, target_points):
    """
    Perform point cloud registration using manually picked correspondences.

    Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.
        source_points (list): Indices of picked points in the source point cloud.
        target_points (list): Indices of picked points in the target point cloud.
    """
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points

    # Estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    # Point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    draw_registration_result(source, target, reg_p2p.transformation)



def manual_registration(source, target):
    """
    Perform manual ICP registration on two given point clouds.

    Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.
    """
    print("Manual ICP registration")
    # Pick points from two point clouds to build correspondences
    print("Pick points from the source point cloud")
    source_points = pick_points(source)
    print("Pick points from the target point cloud")
    target_points = pick_points(target)

    assert len(source_points) >= 3 and len(target_points) >= 3, "At least three points must be selected in each point cloud."
    assert len(source_points) == len(target_points), "The number of source and target points must match."

    register_via_correspondences(source, target, source_points, target_points)



def load_ply_point_cloud(filepath,
                         remove_outliers=False,
                         nb_neighbors=20,
                         std_ratio=2.0,
                         show=True):
    """
    Loads a point cloud from a .ply file and optionally removes statistical outliers.

    Args:
        filepath (str): Path to the .ply file.
        remove_outliers (bool): Whether to apply statistical outlier removal.
        nb_neighbors (int): Number of neighbors to analyze for each point (used in outlier removal).
        std_ratio (float): Standard deviation ratio threshold for outlier removal.
        show (bool): Whether to visualize the loaded point cloud.

    Returns:
        open3d.geometry.PointCloud: The loaded (and possibly filtered) point cloud.
    """
    pcd = o3d.io.read_point_cloud(filepath)

    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if show:
        o3d.visualization.draw_geometries([pcd])

    return pcd
