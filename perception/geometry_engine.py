"""
perception/geometry_engine.py
===============================
Scalable Geometry Engine — beyond boxes and cylinders.

Instead of approximating every object as a box, this module:
  1. Extracts 3D point clouds from SAM3 masks + depth maps
  2. Computes tight-fitting Convex Hulls around the points
  3. Optionally decomposes concave shapes into unions of convex hulls
  4. Exports MuJoCo-compatible mesh colliders
  5. Provides level-of-detail selection (primitive for far, hull for near)

Usage:
    engine = GeometryEngine()
    
    # From a SAM3 mask and depth map:
    hull = engine.extract_convex_hull(mask, depth, intrinsics, extrinsics)
    print(hull.vertices.shape, hull.volume_m3)
    
    # Generate MuJoCo mesh XML:
    xml = engine.to_mujoco_mesh(hull, "red_mug")
    
    # Level-of-detail:
    geom = engine.get_best_representation(hull, distance_to_robot=0.3)
    # Returns convex hull (close) or primitive box (far)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PointCloud:
    """3D point cloud extracted from a depth map with a mask."""
    points: np.ndarray          # (N, 3) world-frame coordinates
    colors: np.ndarray = None   # (N, 3) RGB colors if available
    normals: np.ndarray = None  # (N, 3) estimated surface normals
    num_points: int = 0
    source_camera: str = ""


@dataclass
class ConvexHull:
    """A convex hull wrapping a set of 3D points."""
    vertices: np.ndarray        # (V, 3) hull vertices
    faces: np.ndarray           # (F, 3) triangle indices
    centroid: np.ndarray        # (3,) center of mass
    volume_m3: float            # enclosed volume
    surface_area_m2: float
    bounding_box: Dict[str, float]  # width, height, depth
    num_vertices: int
    num_faces: int
    
    # Fitted primitive (for level-of-detail fallback)
    best_primitive: str = "box"  # box, cylinder, sphere
    primitive_fit_error: float = 0.0  # how well the primitive fits


@dataclass
class MeshCollider:
    """A MuJoCo-compatible mesh collider."""
    vertices: np.ndarray        # (V, 3)
    faces: np.ndarray           # (F, 3)
    name: str
    xml_asset: str              # <mesh> XML for MuJoCo
    xml_geom: str               # <geom type="mesh"> XML


class GeometryEngine:
    """
    Extracts precise 3D geometry from perception data.
    
    The key insight is that depth maps + segmentation masks contain
    enough information to reconstruct the visible surface of any object.
    Multi-view fusion then fills in the occluded surfaces.
    """
    
    def __init__(self, voxel_size: float = 0.005,
                 min_points: int = 50,
                 hull_simplification: float = 0.02):
        """
        Args:
            voxel_size: downsampling grid size for point clouds
            min_points: minimum points needed to compute a hull
            hull_simplification: max deviation from true hull (metres)
        """
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.hull_simplification = hull_simplification
    
    def extract_point_cloud(self, mask: np.ndarray,
                            depth: np.ndarray,
                            intrinsics: Dict[str, float],
                            extrinsics: np.ndarray = None,
                            rgb: np.ndarray = None,
                            camera_name: str = "") -> Optional[PointCloud]:
        """
        Extract a 3D point cloud for a segmented object.
        
        Args:
            mask: (H, W) bool — SAM3 segmentation mask
            depth: (H, W) float — depth map in metres
            intrinsics: {fx, fy, cx, cy}
            extrinsics: optional (4, 4) camera-to-world transform
            rgb: optional (H, W, 3) for colored point cloud
            camera_name: source camera identifier
            
        Returns:
            PointCloud in world frame, or None if insufficient points
        """
        # Get masked depth pixels
        ys, xs = np.where(mask)
        if len(ys) < self.min_points:
            return None
        
        depths = depth[ys, xs]
        valid = depths > 0
        ys, xs, depths = ys[valid], xs[valid], depths[valid]
        
        if len(ys) < self.min_points:
            return None
        
        # Unproject to 3D (camera frame)
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        
        x3d = (xs.astype(float) - cx) * depths / fx
        y3d = (ys.astype(float) - cy) * depths / fy
        z3d = depths
        
        points_camera = np.stack([x3d, y3d, z3d], axis=1)  # (N, 3)
        
        # Transform to world frame
        if extrinsics is not None:
            ones = np.ones((len(points_camera), 1))
            points_h = np.hstack([points_camera, ones])  # (N, 4)
            points_world = (extrinsics @ points_h.T).T[:, :3]  # (N, 3)
        else:
            points_world = points_camera
        
        # Downsample using voxel grid
        points_world = self._voxel_downsample(points_world)
        
        # Extract colors if available
        colors = None
        if rgb is not None:
            colors_raw = rgb[ys[valid], xs[valid]]
            # Downsample colors would require matching — skip for now
            colors = None
        
        # Estimate normals
        normals = self._estimate_normals(points_world)
        
        return PointCloud(
            points=points_world,
            colors=colors,
            normals=normals,
            num_points=len(points_world),
            source_camera=camera_name,
        )
    
    def compute_convex_hull(self, point_cloud: PointCloud) -> Optional[ConvexHull]:
        """
        Compute the convex hull of a 3D point cloud.
        
        Uses scipy's ConvexHull for robust computation.
        Falls back to bounding box if scipy is unavailable.
        """
        points = point_cloud.points
        
        if len(points) < 4:
            return None
        
        try:
            from scipy.spatial import ConvexHull as ScipyHull
            
            hull = ScipyHull(points)
            
            vertices = points[hull.vertices]
            # Convert simplices to triangular faces
            faces = hull.simplices
            
            centroid = np.mean(vertices, axis=0)
            volume = hull.volume
            area = hull.area
            
            # Bounding box
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_dims = bbox_max - bbox_min
            
            # Determine best-fitting primitive
            best_prim, fit_err = self._fit_primitive(vertices, bbox_dims)
            
            return ConvexHull(
                vertices=vertices,
                faces=faces,
                centroid=centroid,
                volume_m3=float(volume),
                surface_area_m2=float(area),
                bounding_box={
                    "width": float(bbox_dims[0]),
                    "height": float(bbox_dims[1]),
                    "depth": float(bbox_dims[2]),
                },
                num_vertices=len(vertices),
                num_faces=len(faces),
                best_primitive=best_prim,
                primitive_fit_error=fit_err,
            )
            
        except ImportError:
            # Fallback: just use bounding box
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            bbox_dims = bbox_max - bbox_min
            centroid = (bbox_min + bbox_max) / 2
            
            # Create box vertices
            corners = np.array([
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
            ])
            
            return ConvexHull(
                vertices=corners,
                faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                                [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                                [1,2,6],[1,6,5],[0,3,7],[0,7,4]]),
                centroid=centroid,
                volume_m3=float(np.prod(bbox_dims)),
                surface_area_m2=float(2 * (bbox_dims[0]*bbox_dims[1] + 
                                           bbox_dims[1]*bbox_dims[2] + 
                                           bbox_dims[0]*bbox_dims[2])),
                bounding_box={
                    "width": float(bbox_dims[0]),
                    "height": float(bbox_dims[1]),
                    "depth": float(bbox_dims[2]),
                },
                num_vertices=8,
                num_faces=12,
                best_primitive="box",
                primitive_fit_error=0.0,
            )
    
    def merge_point_clouds(self, clouds: List[PointCloud]) -> PointCloud:
        """
        Merge point clouds from multiple cameras into one.
        
        This is how multi-view fusion achieves complete 3D coverage.
        Each camera sees one side; merging sees all sides.
        """
        all_points = []
        for cloud in clouds:
            if cloud is not None:
                all_points.append(cloud.points)
        
        if not all_points:
            return PointCloud(points=np.zeros((0, 3)), num_points=0)
        
        merged = np.vstack(all_points)
        merged = self._voxel_downsample(merged)
        
        return PointCloud(
            points=merged,
            num_points=len(merged),
            source_camera="merged",
        )
    
    def to_mujoco_mesh(self, hull: ConvexHull, name: str) -> MeshCollider:
        """
        Convert a convex hull to MuJoCo mesh XML.
        
        Returns XML strings ready to paste into a MuJoCo model.
        """
        # Flatten vertices for MuJoCo
        vert_str = " ".join(f"{v:.5f}" for v in hull.vertices.flatten())
        face_str = " ".join(str(int(f)) for f in hull.faces.flatten())
        
        xml_asset = f'<mesh name="{name}_mesh" vertex="{vert_str}" face="{face_str}"/>'
        xml_geom = f'<geom type="mesh" mesh="{name}_mesh"/>'
        
        return MeshCollider(
            vertices=hull.vertices,
            faces=hull.faces,
            name=name,
            xml_asset=xml_asset,
            xml_geom=xml_geom,
        )
    
    def get_best_representation(self, hull: ConvexHull,
                                 distance_to_robot: float) -> dict:
        """
        Level-of-detail selection.
        
        Close objects get convex hull colliders (accurate).
        Far objects get primitive colliders (fast).
        
        Args:
            hull: the computed convex hull
            distance_to_robot: distance from robot to object in metres
            
        Returns:
            dict with 'type' ('mesh' or 'primitive'), 'geom_xml', 'vertices'
        """
        # Within 50cm: use full hull
        if distance_to_robot < 0.5:
            mesh = self.to_mujoco_mesh(hull, "obj")
            return {
                "type": "mesh",
                "geom_xml": mesh.xml_geom,
                "asset_xml": mesh.xml_asset,
                "detail": "convex_hull",
            }
        
        # Beyond 50cm: use best-fitting primitive
        dims = hull.bounding_box
        w, h, d = dims["width"], dims["height"], dims["depth"]
        shape = hull.best_primitive
        
        if shape == "sphere":
            r = max(w, h, d) / 2
            geom = f'<geom type="sphere" size="{r:.4f}"/>'
        elif shape == "cylinder":
            geom = f'<geom type="cylinder" size="{max(w,d)/2:.4f} {h/2:.4f}"/>'
        else:
            geom = f'<geom type="box" size="{w/2:.4f} {h/2:.4f} {d/2:.4f}"/>'
        
        return {
            "type": "primitive",
            "geom_xml": geom,
            "detail": hull.best_primitive,
        }
    
    # ─── Internal Methods ───
    
    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Downsample point cloud using voxel grid."""
        if len(points) == 0:
            return points
        
        # Quantize to voxel grid
        quantized = np.round(points / self.voxel_size).astype(int)
        
        # Get unique voxels
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        
        return points[unique_indices]
    
    def _estimate_normals(self, points: np.ndarray,
                          k: int = 10) -> Optional[np.ndarray]:
        """Estimate surface normals using local PCA."""
        if len(points) < k:
            return None
        
        normals = np.zeros_like(points)
        
        for i in range(len(points)):
            # Find k nearest neighbors (brute force for now)
            dists = np.linalg.norm(points - points[i], axis=1)
            nn_idx = np.argsort(dists)[:k]
            neighbors = points[nn_idx]
            
            # PCA: smallest eigenvector = normal direction
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # smallest eigenvalue direction
        
        return normals
    
    def _fit_primitive(self, vertices: np.ndarray,
                        bbox_dims: np.ndarray) -> Tuple[str, float]:
        """
        Determine which primitive shape best fits the hull.
        
        Returns (shape_name, fit_error).
        """
        w, h, d = bbox_dims
        
        # Aspect ratios
        max_dim = max(w, h, d)
        min_dim = max(min(w, h, d), 0.001)
        aspect = max_dim / min_dim
        
        # Isotropy (how cube-like)
        isotropy = min_dim / max_dim
        
        if isotropy > 0.7:
            # Nearly equal in all dimensions
            return "sphere", 1.0 - isotropy
        elif aspect > 2.0:
            # One dimension much longer → cylinder
            return "cylinder", aspect / 10.0
        else:
            # Default to box
            return "box", 0.0
    
    def print_hull_info(self, hull: ConvexHull, name: str = ""):
        """Print hull information."""
        print(f"\n[Geometry] {'Hull: ' + name if name else 'Convex Hull'}")
        print(f"  Vertices: {hull.num_vertices} | Faces: {hull.num_faces}")
        print(f"  Volume: {hull.volume_m3*1e6:.1f} cm³")
        print(f"  BBox: {hull.bounding_box['width']*100:.1f} × "
              f"{hull.bounding_box['height']*100:.1f} × "
              f"{hull.bounding_box['depth']*100:.1f} cm")
        print(f"  Best primitive: {hull.best_primitive}")
        print(f"  Centroid: ({hull.centroid[0]:.3f}, {hull.centroid[1]:.3f}, {hull.centroid[2]:.3f})")

