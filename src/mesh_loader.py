"""
Mesh Loader dan Generator untuk primitive shapes
Untuk Computer Graphics - Universitas Indonesia

Fungsi untuk load/save mesh dan generate primitive shapes
"""

import numpy as np
from typing import Optional, List
from halfedge_mesh import HalfedgeMesh, Vertex

class MeshLoader:
    """Loader untuk berbagai format mesh"""

    @staticmethod
    def load_obj(filename: str) -> Optional[HalfedgeMesh]:
        """
        Load mesh dari file .OBJ

        Format OBJ:
        v x y z        - vertex
        f v1 v2 v3     - face (1-indexed)
        """
        mesh = HalfedgeMesh()
        vertices = []
        faces = []

        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0] == 'v':
                        # Vertex
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        v = mesh.add_vertex([x, y, z])
                        vertices.append(v)

                    elif parts[0] == 'f':
                        # Face - OBJ uses 1-based indexing
                        face_vertices = []
                        for i in range(1, len(parts)):
                            # Handle v/vt/vn format
                            v_idx = int(parts[i].split('/')[0]) - 1
                            if 0 <= v_idx < len(vertices):
                                face_vertices.append(vertices[v_idx])

                        if len(face_vertices) >= 3:
                            faces.append(face_vertices)

            # Build halfedge connectivity
            MeshLoader._build_halfedge_connectivity(mesh, faces)

            return mesh

        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            return None

    @staticmethod
    def save_obj(mesh: HalfedgeMesh, filename: str) -> bool:
        """
        Save mesh ke file .OBJ
        """
        try:
            with open(filename, 'w') as file:
                file.write("# Mesh exported from Python Mesh Editor\n")
                file.write(f"# Vertices: {len(mesh.vertices)}\n")
                file.write(f"# Faces: {len(mesh.faces)}\n\n")

                # Write vertices
                vertex_index = {}
                idx = 1
                for v in mesh.vertices:
                    if not v.deleted:
                        file.write(f"v {v.position[0]:.6f} {v.position[1]:.6f} {v.position[2]:.6f}\n")
                        vertex_index[v] = idx
                        idx += 1

                file.write("\n")

                # Write faces
                for f in mesh.faces:
                    if not f.deleted and not f.is_boundary:
                        vertices = f.vertices()
                        if len(vertices) >= 3:
                            file.write("f")
                            for v in vertices:
                                file.write(f" {vertex_index[v]}")
                            file.write("\n")

            return True

        except Exception as e:
            print(f"Error saving OBJ file: {e}")
            return False

    @staticmethod
    def _build_halfedge_connectivity(mesh: HalfedgeMesh, faces: List[List[Vertex]]):
        """
        Build halfedge connectivity dari vertex dan face lists
        """
        # Dictionary untuk track halfedge twins
        edge_halfedges = {}  # (v1, v2) -> halfedge

        for face_vertices in faces:
            if len(face_vertices) < 3:
                continue

            # Create face
            f = mesh.add_face()

            # Create halfedges untuk face ini
            halfedges = []
            for i in range(len(face_vertices)):
                h = mesh.add_halfedge()
                halfedges.append(h)

            # Set up connectivity
            for i in range(len(face_vertices)):
                h = halfedges[i]
                v_curr = face_vertices[i]
                v_next = face_vertices[(i + 1) % len(face_vertices)]

                # Set vertex
                h.vertex = v_next

                # Set next
                h.next = halfedges[(i + 1) % len(halfedges)]

                # Set face
                h.face = f

                # Create or find edge
                edge_key = (min(id(v_curr), id(v_next)), max(id(v_curr), id(v_next)))

                if edge_key not in edge_halfedges:
                    # Create new edge
                    e = mesh.add_edge()
                    h.edge = e
                    e.halfedge = h
                    edge_halfedges[edge_key] = [h]
                else:
                    # Edge already exists, set twin
                    other_h = edge_halfedges[edge_key][0]
                    h.twin = other_h
                    other_h.twin = h
                    h.edge = other_h.edge
                    edge_halfedges[edge_key].append(h)

                # Set vertex halfedge
                if not v_curr.halfedge:
                    v_curr.halfedge = h

            # Set face halfedge
            f.halfedge = halfedges[0]

            # Verify all next pointers are set
            for i, h in enumerate(halfedges):
                if not h.next:
                    print(f"Warning: Halfedge {h.id} in face {f.id} has no next pointer")
                    # Fix: set next pointer
                    h.next = halfedges[(i + 1) % len(halfedges)]

        # Handle boundary edges (edges with only one halfedge)
        for edge_key, halfedges_list in edge_halfedges.items():
            if len(halfedges_list) == 1:
                h = halfedges_list[0]

                # Create boundary halfedge
                h_boundary = mesh.add_halfedge()
                h_boundary.twin = h
                h.twin = h_boundary

                # Set vertices (opposite direction)
                # Find the source vertex of h by traversing to previous halfedge
                prev_h = h
                max_iterations = 1000 # Prevent infinite loops
                iterations = 0
                while prev_h.next != h and iterations < max_iterations:
                    prev_h = prev_h.next
                    iterations += 1

                if iterations >= max_iterations:
                    print(f"Warning: Could not find previous halfedge for boundary edge")
                    # Fallback: use the destination vertex as source (this creates degenerate boundary)
                    h_boundary.vertex = h.vertex
                else:
                    h_boundary.vertex = prev_h.vertex

                # Set edge
                h_boundary.edge = h.edge

                # Set next for boundary halfedge (point to itself for degenerate case)
                h_boundary.next = h_boundary

                # Create boundary face
                f_boundary = mesh.add_face()
                f_boundary.is_boundary = True
                h_boundary.face = f_boundary
                mesh.boundary_faces.append(f_boundary)


class PrimitiveGenerator:
    """Generator untuk primitive 3D shapes"""

    @staticmethod
    def create_cube(size: float = 1.0) -> HalfedgeMesh:
        """
        Create unit cube centered at origin

        """
        mesh = HalfedgeMesh()

        # Create vertices
        half = size / 2.0
        positions = [
            [-half, -half, -half],  # 0
            [-half, -half,  half],  # 1
            [ half, -half,  half],  # 2
            [ half, -half, -half],  # 3
            [-half,  half, -half],  # 4
            [-half,  half,  half],  # 5
            [ half,  half,  half],  # 6
            [ half,  half, -half]   # 7
        ]

        vertices = []
        for pos in positions:
            v = mesh.add_vertex(pos)
            vertices.append(v)

        # Define faces (quads)
        faces = [
            [0, 3, 2, 1],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Left
            [2, 3, 7, 6],  # Right
            [0, 4, 7, 3],  # Back
            [1, 2, 6, 5]   # Front
        ]

        face_list = []
        for face_indices in faces:
            face_vertices = [vertices[i] for i in face_indices]
            face_list.append(face_vertices)

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, face_list)

        return mesh

    @staticmethod
    def create_tetrahedron(size: float = 1.0) -> HalfedgeMesh:
        """
        Create regular tetrahedron
        """
        mesh = HalfedgeMesh()

        # Create vertices
        a = size / np.sqrt(2.0)
        positions = [
            [ a,  0, -a/np.sqrt(2.0)],
            [-a,  0, -a/np.sqrt(2.0)],
            [ 0,  a,  a/np.sqrt(2.0)],
            [ 0, -a,  a/np.sqrt(2.0)]
        ]

        vertices = []
        for pos in positions:
            v = mesh.add_vertex(pos)
            vertices.append(v)

        # Define faces (triangles)
        faces = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2]
        ]

        face_list = []
        for face_indices in faces:
            face_vertices = [vertices[i] for i in face_indices]
            face_list.append(face_vertices)

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, face_list)

        return mesh

    @staticmethod
    def create_icosahedron(size: float = 1.0) -> HalfedgeMesh:
        """
        Create regular icosahedron (20 faces)
        """
        mesh = HalfedgeMesh()

        # Golden ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0

        # Create vertices
        scale = size / np.sqrt(phi * phi + 1)
        positions = [
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],

            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],

            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ]

        vertices = []
        for pos in positions:
            v = mesh.add_vertex(np.array(pos) * scale)
            vertices.append(v)

        # Define faces (triangles)
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        face_list = []
        for face_indices in faces:
            face_vertices = [vertices[i] for i in face_indices]
            face_list.append(face_vertices)

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, face_list)

        return mesh

    @staticmethod
    def create_sphere(radius: float = 1.0, subdivisions: int = 2) -> HalfedgeMesh:
        """
        Create sphere by subdividing icosahedron
        """
        # Start with icosahedron
        mesh = PrimitiveGenerator.create_icosahedron(radius)

        # Subdivide
        from mesh_operations import MeshOperations
        ops = MeshOperations(mesh)

        for _ in range(subdivisions):
            ops.subdivide_linear()

            # Project vertices to sphere
            for v in mesh.vertices:
                if not v.deleted:
                    v.position = v.position / np.linalg.norm(v.position) * radius

        return mesh

    @staticmethod
    def create_cylinder(radius: float = 0.5, height: float = 2.0, segments: int = 16) -> HalfedgeMesh:
        """
        Create cylinder
        """
        mesh = HalfedgeMesh()
        vertices = []

        # Create vertices for top and bottom circles
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)

            # Bottom vertex
            v_bottom = mesh.add_vertex([x, -height/2, z])
            vertices.append(v_bottom)

            # Top vertex
            v_top = mesh.add_vertex([x, height/2, z])
            vertices.append(v_top)

        # Create center vertices for caps
        v_bottom_center = mesh.add_vertex([0, -height/2, 0])
        v_top_center = mesh.add_vertex([0, height/2, 0])

        faces = []

        # Create side faces (quads)
        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = vertices[i * 2]      # bottom current
            v1 = vertices[i * 2 + 1]  # top current
            v2 = vertices[next_i * 2 + 1]  # top next
            v3 = vertices[next_i * 2]  # bottom next

            faces.append([v0, v3, v2, v1])

        # Create bottom cap (triangles)
        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = vertices[i * 2]
            v1 = vertices[next_i * 2]
            faces.append([v0, v_bottom_center, v1])

        # Create top cap (triangles)
        for i in range(segments):
            next_i = (i + 1) % segments
            v0 = vertices[i * 2 + 1]
            v1 = vertices[next_i * 2 + 1]
            faces.append([v0, v1, v_top_center])

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, faces)

        return mesh

    @staticmethod
    def create_torus(major_radius: float = 1.0, minor_radius: float = 0.3,
                     major_segments: int = 16, minor_segments: int = 8) -> HalfedgeMesh:
        """
        Create torus
        """
        mesh = HalfedgeMesh()
        vertices = []

        # Create vertices
        for i in range(major_segments):
            theta = 2 * np.pi * i / major_segments
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            for j in range(minor_segments):
                phi = 2 * np.pi * j / minor_segments
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                x = (major_radius + minor_radius * cos_phi) * cos_theta
                y = minor_radius * sin_phi
                z = (major_radius + minor_radius * cos_phi) * sin_theta

                v = mesh.add_vertex([x, y, z])
                vertices.append(v)

        # Create faces (quads)
        faces = []
        for i in range(major_segments):
            next_i = (i + 1) % major_segments
            for j in range(minor_segments):
                next_j = (j + 1) % minor_segments

                v0 = vertices[i * minor_segments + j]
                v1 = vertices[next_i * minor_segments + j]
                v2 = vertices[next_i * minor_segments + next_j]
                v3 = vertices[i * minor_segments + next_j]

                faces.append([v0, v1, v2, v3])

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, faces)

        return mesh

    @staticmethod
    def create_cone(radius: float = 1.0, height: float = 2.0, segments: int = 16) -> HalfedgeMesh:
        """
        Create cone
        """
        mesh = HalfedgeMesh()
        vertices = []

        # Create apex vertex
        apex = mesh.add_vertex([0, height/2, 0])
        vertices.append(apex)

        # Create base center vertex
        base_center = mesh.add_vertex([0, -height/2, 0])
        vertices.append(base_center)

        # Create base rim vertices
        base_vertices = []
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            v = mesh.add_vertex([x, -height/2, z])
            vertices.append(v)
            base_vertices.append(v)

        faces = []

        # Create side triangles (from apex to base rim)
        for i in range(segments):
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % segments]
            faces.append([apex, v2, v1])

        # Create base triangles
        for i in range(segments):
            v1 = base_vertices[i]
            v2 = base_vertices[(i + 1) % segments]
            faces.append([base_center, v1, v2])

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, faces)

        return mesh

    @staticmethod
    def create_octahedron(size: float = 1.0) -> HalfedgeMesh:
        """
        Create regular octahedron (8 faces)
        """
        mesh = HalfedgeMesh()

        # Create vertices (6 vertices at unit positions along axes)
        scale = size / np.sqrt(2)
        positions = [
            [ scale, 0, 0],  # +X
            [-scale, 0, 0],  # -X
            [0,  scale, 0],  # +Y
            [0, -scale, 0],  # -Y
            [0, 0,  scale],  # +Z
            [0, 0, -scale],  # -Z
        ]

        vertices = []
        for pos in positions:
            v = mesh.add_vertex(pos)
            vertices.append(v)

        # Define faces (8 triangles)
        faces = [
            # Upper half
            [0, 2, 4],  # +X, +Y, +Z
            [2, 1, 4],  # +Y, -X, +Z
            [1, 3, 4],  # -X, -Y, +Z
            [3, 0, 4],  # -Y, +X, +Z
            # Lower half
            [2, 0, 5],  # +Y, +X, -Z
            [1, 2, 5],  # -X, +Y, -Z
            [3, 1, 5],  # -Y, -X, -Z
            [0, 3, 5],  # +X, -Y, -Z
        ]

        face_list = []
        for face_indices in faces:
            face_vertices = [vertices[i] for i in face_indices]
            face_list.append(face_vertices)

        # Build halfedge connectivity
        MeshLoader._build_halfedge_connectivity(mesh, face_list)

        return mesh