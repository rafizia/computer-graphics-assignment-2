#!/usr/bin/env python3
"""
Scene management untuk multiple objects
"""

import numpy as np
from typing import List, Optional, Tuple
from halfedge_mesh import HalfedgeMesh
import uuid

class Transform:
    """Transform untuk object dalam scene"""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # Euler angles dalam radians
        self.scale = np.array([1.0, 1.0, 1.0])

    def get_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        # Translation matrix
        T = np.eye(4)
        T[0:3, 3] = self.position

        # Rotation matrices (Z-Y-X order)
        rx, ry, rz = self.rotation

        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(rx), -np.sin(rx), 0],
                       [0, np.sin(rx), np.cos(rx), 0],
                       [0, 0, 0, 1]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                       [0, 1, 0, 0],
                       [-np.sin(ry), 0, np.cos(ry), 0],
                       [0, 0, 0, 1]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                       [np.sin(rz), np.cos(rz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Scale matrix
        S = np.eye(4)
        S[0, 0] = self.scale[0]
        S[1, 1] = self.scale[1]
        S[2, 2] = self.scale[2]

        # Combine: T * Rz * Ry * Rx * S
        return T @ Rz @ Ry @ Rx @ S

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point"""
        p = np.ones(4)
        p[:3] = point
        transformed = self.get_matrix() @ p
        return transformed[:3]


class SceneObject:
    """Object dalam scene"""

    def __init__(self, mesh: HalfedgeMesh, name: str = None):
        self.id = str(uuid.uuid4())
        self.name = name or f"Object_{self.id[:8]}"
        self.mesh = mesh
        self.transform = Transform()
        self.visible = True
        self.selected = False
        self.wireframe = False
        self.color = [0.7, 0.7, 0.7]  # Default gray

    def get_transformed_mesh(self) -> HalfedgeMesh:
        """Get mesh dengan transformasi applied"""
        # Create copy dari mesh
        transformed = HalfedgeMesh()

        # Copy vertices dengan transformasi
        vertex_map = {}
        for v in self.mesh.vertices:
            if v.deleted:
                continue
            new_pos = self.transform.transform_point(v.position)
            new_v = transformed.add_vertex(new_pos)
            vertex_map[v.id] = new_v

        # Copy faces
        from mesh_loader import MeshLoader
        faces_vertices = []
        for f in self.mesh.faces:
            if f.deleted or f.is_boundary:
                continue
            face_verts = []
            for v in f.vertices():
                if v.id in vertex_map:
                    face_verts.append(vertex_map[v.id])
            if len(face_verts) >= 3:
                faces_vertices.append(face_verts)

        if faces_vertices:
            MeshLoader._build_halfedge_connectivity(transformed, faces_vertices)

        return transformed

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box dari transformed object"""
        min_pt = np.array([float('inf')] * 3)
        max_pt = np.array([float('-inf')] * 3)

        for v in self.mesh.vertices:
            if v.deleted:
                continue
            pos = self.transform.transform_point(v.position)
            min_pt = np.minimum(min_pt, pos)
            max_pt = np.maximum(max_pt, pos)

        return min_pt, max_pt


class Scene:
    """Scene manager untuk multiple objects"""

    def __init__(self):
        self.objects: List[SceneObject] = []
        self.selected_object: Optional[SceneObject] = None
        self.show_grid = True
        self.show_axes = True
        self.grid_size = 20
        self.grid_spacing = 1.0

    def add_object(self, mesh: HalfedgeMesh, name: str = None) -> SceneObject:
        """Tambah object ke scene"""
        obj = SceneObject(mesh, name)
        self.objects.append(obj)
        return obj

    def remove_object(self, obj: SceneObject):
        """Remove object dari scene"""
        if obj in self.objects:
            self.objects.remove(obj)
            if self.selected_object == obj:
                self.selected_object = None

    def select_object(self, obj: SceneObject):
        """Select object"""
        # Deselect all
        for o in self.objects:
            o.selected = False

        # Select new object
        if obj:
            obj.selected = True
            self.selected_object = obj
        else:
            self.selected_object = None

    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get overall bounding box dari semua objects"""
        if not self.objects:
            return np.array([-5, -5, -5]), np.array([5, 5, 5])

        min_pt = np.array([float('inf')] * 3)
        max_pt = np.array([float('-inf')] * 3)

        for obj in self.objects:
            if not obj.visible:
                continue
            obj_min, obj_max = obj.get_bounding_box()
            min_pt = np.minimum(min_pt, obj_min)
            max_pt = np.maximum(max_pt, obj_max)

        # Ensure minimum size
        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt
        min_size = 10.0
        if np.max(size) < min_size:
            half_size = min_size / 2
            min_pt = center - half_size
            max_pt = center + half_size

        return min_pt, max_pt

    def clear(self):
        """Clear semua objects dari scene"""
        self.objects.clear()
        self.selected_object = None

    def duplicate_selected(self) -> Optional[SceneObject]:
        """Duplicate selected object"""
        if not self.selected_object:
            return None

        # Create copy dari mesh
        new_mesh = HalfedgeMesh()

        # Copy vertices
        vertex_map = {}
        for v in self.selected_object.mesh.vertices:
            if v.deleted:
                continue
            new_v = new_mesh.add_vertex(v.position.copy())
            vertex_map[v.id] = new_v

        # Copy faces
        from mesh_loader import MeshLoader
        faces_vertices = []
        for f in self.selected_object.mesh.faces:
            if f.deleted or f.is_boundary:
                continue
            face_verts = []
            for v in f.vertices():
                if v.id in vertex_map:
                    face_verts.append(vertex_map[v.id])
            if len(face_verts) >= 3:
                faces_vertices.append(face_verts)

        if faces_vertices:
            MeshLoader._build_halfedge_connectivity(new_mesh, faces_vertices)

        # Create new object
        new_obj = self.add_object(new_mesh, f"{self.selected_object.name}_copy")

        # Copy transform dengan slight offset
        new_obj.transform.position = self.selected_object.transform.position + np.array([2, 0, 0])
        new_obj.transform.rotation = self.selected_object.transform.rotation.copy()
        new_obj.transform.scale = self.selected_object.transform.scale.copy()
        new_obj.color = self.selected_object.color.copy()

        return new_obj

    def get_object_at_position(self, x: int, y: int, viewport_width: int, viewport_height: int,
                              view_matrix: np.ndarray, proj_matrix: np.ndarray) -> Optional[SceneObject]:
        """Get object at screen position (untuk picking)"""
        # Simple ray casting - bisa di-improve dengan proper ray-triangle intersection
        # Untuk sekarang, check bounding box saja

        for obj in self.objects:
            if not obj.visible:
                continue

            # Get bounding box corners
            min_pt, max_pt = obj.get_bounding_box()

            # Create 8 corners dari bounding box
            for i in range(8):
                corner = np.array([
                    min_pt[0] if i & 1 == 0 else max_pt[0],
                    min_pt[1] if i & 2 == 0 else max_pt[1],
                    min_pt[2] if i & 4 == 0 else max_pt[2],
                    1.0
                ])

                # Transform ke screen space
                clip_pos = proj_matrix @ view_matrix @ corner
                if clip_pos[3] != 0:
                    ndc = clip_pos[:3] / clip_pos[3]
                    screen_x = (ndc[0] + 1) * viewport_width / 2
                    screen_y = (1 - ndc[1]) * viewport_height / 2

                    # Check if click is near this corner
                    if abs(screen_x - x) < 50 and abs(screen_y - y) < 50:
                        return obj

        return None