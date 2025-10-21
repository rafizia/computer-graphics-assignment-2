"""
Mesh Operations Implementation
Untuk Computer Graphics - Universitas Indonesia

Implementasi operasi-operasi mesh:
- Global Operations: triangulate, subdivide
"""

import numpy as np
from typing import Optional
from halfedge_mesh import HalfedgeMesh, Vertex, Edge, Face
from mesh_loader import MeshLoader

class MeshOperations:
    """Kelas untuk operasi-operasi mesh"""

    def __init__(self, mesh: HalfedgeMesh):
        self.mesh = mesh

    def triangulate(self) -> int:
        """
        Triangulate semua non-triangle faces dalam mesh.

        Definisi:
            - Triangulasi = membagi face dengan n sisi (n > 3)
            menjadi beberapa face berbentuk triangle.
            - Teknik yang umum: fan triangulation
            (ambil vertex pertama v0, lalu buat segitiga [v0, vi, vi+1]).

        Tujuan:
            - Mengubah semua face non-triangle menjadi triangle.
            - Penting agar mesh bisa dipakai untuk algoritma yang
            butuh triangular mesh (misalnya Loop subdivision, ray tracing, dll).

        Langkah yang perlu dilakukan (hint):
            1. Iterasi semua face dalam mesh.
            2. Lewati face yang sudah triangle, boundary, atau dihapus.
            3. Ambil list vertex dari face (f.vertices()).
            4. Jika face punya n > 3 vertex:
                - Pilih v0 sebagai pivot.
                - Buat triangle-triangle baru: [v0, v[i], v[i+1]].
                - Simpan face original ke daftar yang akan dihapus.
            5. Setelah semua face diproses:
                - Tandai face lama sebagai deleted.
                - Bangun kembali halfedge connectivity
                untuk face-face baru.
            6. Return jumlah face yang berhasil di-triangulate.

        Returns:
            int: jumlah face non-triangle yang sudah ditriangulasi.
        """
        faces_to_delete = [] # Face yg perlu ditriangulasi
        new_triangles = [] # Triangles baru
        triangulated_count = 0

        for face in self.mesh.faces:
            # Lewati face yang sudah triangle, boundary, atau dihapus
            if face.deleted or face.is_boundary:
                continue
            
            vertices = face.vertices()
            n = len(vertices)
            
            # Jika face punya n > 3 vertex, lakukan fan triangulation
            if n > 3:
                v0 = vertices[0] # Pivot
                
                # Buat triangle-triangle baru: [v0, v[i], v[i+1]]
                for i in range(1, n - 1):
                    triangle = [v0, vertices[i], vertices[i + 1]]
                    new_triangles.append(triangle)

                faces_to_delete.append(face)
                triangulated_count += 1
        
        # Jika tidak ada face yang perlu ditriangulasi
        if triangulated_count == 0:
            return 0
        
        for face in faces_to_delete:
            face.mark_deleted()
        
        MeshLoader._build_halfedge_connectivity(self.mesh, new_triangles)
        
        return triangulated_count

    def subdivide_linear(self) -> bool:
        """
        Linear subdivision: membagi setiap face menjadi 4 faces.

        Aturan:
            - Jika face = triangle -> dipecah jadi 4 triangle baru.
            - Jika face = quad -> dipecah jadi 4 quad baru.
            - General polygon (n-gon) -> tiap face dibagi jadi n sub-face.

        Tujuan:
            - Menambah resolusi mesh dengan cara sederhana
            (tanpa smoothing).
            - Dasar untuk metode subdivisi lebih lanjut
            seperti Catmull-Clark.

        Langkah implementasi (hint):
            1. Iterasi semua face dalam mesh, abaikan boundary/deleted.
            2. Simpan posisi semua vertex pada face.
            3. Hitung:
                - Titik tengah face (face center).
                - Titik tengah tiap edge (edge midpoints).
            4. Untuk setiap vertex pada face:
                - Bentuk sub-face dengan corner:
                [corner, edge_mid_next, face_center, edge_mid_prev].
            5. Kumpulkan semua sub-face baru.
            6. Kosongkan mesh lama, lalu rebuild mesh:
                - Buat ulang vertex (gunakan map agar tidak duplikat).
                - Bangun connectivity dengan `MeshLoader._build_halfedge_connectivity`.

        Return:
            bool: True jika subdivision berhasil,
                False jika tidak ada face valid.
        """
        # Mengumpulkan data posisi dari semua face yang valid
        old_faces_data = []
        for face in self.mesh.faces:
            if face.deleted or face.is_boundary:
                continue
            
            vertices = face.vertices()
            if len(vertices) < 3:
                continue
                
            face_data = {
                'vertices': [v.position.copy() for v in vertices]
            }
            old_faces_data.append(face_data)
        
        # Jika tidak ada face valid
        if len(old_faces_data) == 0:
            return False
        
        # Siapkan struktur untuk sub-faces baru
        new_faces = []
        
        # Mapping posisi ke vertex (buat menghindari duplikasi)
        position_to_vertex = {}
        
        def get_or_create_vertex(position):
            """Helper function untuk mendapatkan atau membuat vertex berdasarkan posisi"""
            pos_tuple = tuple(position)
            if pos_tuple not in position_to_vertex:
                position_to_vertex[pos_tuple] = position
            return position_to_vertex[pos_tuple]
        
        # Memproses setiap face lama untuk membuat sub-face baru
        for face_data in old_faces_data:
            vertices_pos = face_data['vertices']
            n = len(vertices_pos)
            
            # Menghitung face center (centroid)
            face_center = np.mean(vertices_pos, axis=0)
            face_center_vertex = get_or_create_vertex(face_center)
            
            # Menghitung edge midpoints
            edge_midpoints = []
            for i in range(n):
                v_curr = vertices_pos[i]
                v_next = vertices_pos[(i + 1) % n]
                midpoint = (v_curr + v_next) / 2.0
                edge_midpoints.append(get_or_create_vertex(midpoint))
            
            # Untuk setiap vertex pada face, bentuk sub-face
            # Pattern: [corner, edge_mid_next, face_center, edge_mid_prev]
            for i in range(n):
                corner = get_or_create_vertex(vertices_pos[i])
                edge_mid_next = edge_midpoints[i]  # edge dari i ke i+1
                edge_mid_prev = edge_midpoints[(i - 1) % n]  # edge dari i-1 ke i
                
                # Buat sub-face baru (quad untuk n-gon, atau triangle jika n=3)
                sub_face = [corner, edge_mid_next, face_center_vertex, edge_mid_prev]
                new_faces.append(sub_face)
        
        # Kosongkan mesh lama
        self.mesh.vertices.clear()
        self.mesh.edges.clear()
        self.mesh.faces.clear()
        self.mesh.halfedges.clear()
        self.mesh.boundary_faces.clear()
        
        # Buat vertex baru dari position_to_vertex map
        vertex_map = {}
        for pos_tuple, position in position_to_vertex.items():
            new_vertex = self.mesh.add_vertex(position)
            vertex_map[pos_tuple] = new_vertex
        
        # Convert new_faces dari positions ke vertex objects
        new_faces_vertices = []
        for face_positions in new_faces:
            face_vertices = [vertex_map[tuple(pos)] for pos in face_positions]
            new_faces_vertices.append(face_vertices)

        MeshLoader._build_halfedge_connectivity(self.mesh, new_faces_vertices)
        
        return True

    def subdivide_loop(self) -> bool:
        """
        Loop subdivision - skema subdivisi khusus untuk triangular meshes.

        Aturan:
            - Hanya berlaku jika semua face adalah triangle.
            - Setiap triangle akan dibagi menjadi 4 triangle yang lebih kecil.
            - Posisi vertex baru dihitung dengan aturan Loop:
                * Edge vertex:
                    - Jika edge interior: 
                    new_pos = 3/8*(v1 + v2) + 1/8*(opp1 + opp2)
                    - Jika edge boundary: midpoint (v1+v2)/2
                * Vertex lama:
                    - Digeser dengan bobot tetangganya:
                    new_pos = (1 - n*β) * v + β * Σ(neighbors)
                    dengan β = 3/16 jika valensi=3,
                  β = 3/(8n) jika valensi > 3.

    Langkah implementasi (hint):
        1. Periksa semua face, pastikan semuanya triangle.
        2. Bangun struktur adjacency edge -> face, dan edge -> opposite vertices.
        3. Hitung posisi vertex baru di setiap edge (edge vertices).
        4. Hitung posisi baru untuk semua vertex lama (pakai bobot Loop).
        5. Hapus mesh lama (clear vertices, edges, faces, halfedges).
        6. Buat ulang:
            - Tambahkan vertex lama dengan posisi baru.
            - Tambahkan edge vertices.
        7. Bangun 4 triangle baru untuk setiap face lama:
            - [corner0, edge0, edge2]
            - [corner1, edge1, edge0]
            - [corner2, edge2, edge1]
            - [edge0, edge1, edge2]
        8. Panggil `MeshLoader._build_halfedge_connectivity` untuk rebuild mesh.

    Return:
        bool: True jika subdivision berhasil,
              False jika mesh bukan triangular mesh.
    """
        # Validasi memastikan semua face adalah triangle
        for face in self.mesh.faces:
            if face.deleted or face.is_boundary:
                continue
            if len(face.vertices()) != 3:
                return False
        
        # Mengumpulkan data face dan vertex lama
        old_faces_data = []
        for face in self.mesh.faces:
            if face.deleted or face.is_boundary:
                continue
            vertices = face.vertices()
            old_faces_data.append({
                'vertices': [v.position.copy() for v in vertices],
                'vertex_objs': vertices
            })
        
        if len(old_faces_data) == 0:
            return False
        
        old_vertices_data = []
        for vertex in self.mesh.vertices:
            if not vertex.deleted:
                old_vertices_data.append({
                    'position': vertex.position.copy(),
                    'neighbors': [n.position.copy() for n in vertex.neighbors()],
                    'valence': vertex.degree()
                })
        
        # Membangun edge adjacency: edge -> (vertices, faces, opposite_vertices)
        edge_map = {}
        for face_data in old_faces_data:
            verts = face_data['vertices']
            for i in range(3):
                v1, v2 = verts[i], verts[(i + 1) % 3]
                v_opp = verts[(i + 2) % 3]
                edge_key = tuple(sorted([tuple(v1), tuple(v2)]))
                
                if edge_key not in edge_map:
                    edge_map[edge_key] = {'vertices': [v1, v2], 'opposites': [], 'faces': 0}
                edge_map[edge_key]['opposites'].append(v_opp)
                edge_map[edge_key]['faces'] += 1
        
        # Menghitung posisi edge vertices dengan formula Loop
        edge_vertex_positions = {}
        for edge_key, edge_data in edge_map.items():
            v1, v2 = edge_data['vertices']
            opposites = edge_data['opposites']
            
            if edge_data['faces'] == 2:  # Interior edge
                opp1, opp2 = opposites[0], opposites[1]
                new_pos = (3/8) * (v1 + v2) + (1/8) * (opp1 + opp2)
            else:  # Boundary edge
                new_pos = (v1 + v2) / 2.0
            
            edge_vertex_positions[edge_key] = new_pos
        
        # Menghitung posisi baru untuk vertex lama dengan formula Loop
        new_vertex_positions = []
        for vertex_data in old_vertices_data:
            pos = vertex_data['position']
            neighbors = vertex_data['neighbors']
            n = vertex_data['valence']
            
            if n == 0:
                new_vertex_positions.append(pos)
                continue
            
            # Menghitung beta berdasarkan valence
            if n == 3:
                beta = 3.0 / 16.0
            else:
                beta = 3.0 / (8.0 * n)
            
            # Formula Loop: (1 - n*β)*v + β*Σ(neighbors)
            neighbor_sum = np.sum(neighbors, axis=0)
            new_pos = (1.0 - n * beta) * pos + beta * neighbor_sum
            new_vertex_positions.append(new_pos)
        
        # Clear mesh lama
        self.mesh.vertices.clear()
        self.mesh.edges.clear()
        self.mesh.faces.clear()
        self.mesh.halfedges.clear()
        self.mesh.boundary_faces.clear()
        
        # Buat vertex baru (old vertices dengan posisi baru)
        old_vertex_map = {}
        for i, vertex_data in enumerate(old_vertices_data):
            old_pos = vertex_data['position']
            new_pos = new_vertex_positions[i]
            new_vertex = self.mesh.add_vertex(new_pos)
            old_vertex_map[tuple(old_pos)] = new_vertex
        
        # Buat edge vertices
        edge_vertex_map = {}
        for edge_key, new_pos in edge_vertex_positions.items():
            new_vertex = self.mesh.add_vertex(new_pos)
            edge_vertex_map[edge_key] = new_vertex
        
        # Build 4 triangle baru untuk setiap face lama
        new_faces = []
        for face_data in old_faces_data:
            verts = face_data['vertices']
            
            # Get corner vertices
            corners = [old_vertex_map[tuple(v)] for v in verts]
            
            # Get edge vertices
            edge_verts = []
            for i in range(3):
                v1, v2 = verts[i], verts[(i + 1) % 3]
                edge_key = tuple(sorted([tuple(v1), tuple(v2)]))
                edge_verts.append(edge_vertex_map[edge_key])
            
            # Pattern: 3 corner triangles + 1 center triangle
            new_faces.append([corners[0], edge_verts[0], edge_verts[2]])
            new_faces.append([corners[1], edge_verts[1], edge_verts[0]])
            new_faces.append([corners[2], edge_verts[2], edge_verts[1]])
            new_faces.append([edge_verts[0], edge_verts[1], edge_verts[2]])
        
        MeshLoader._build_halfedge_connectivity(self.mesh, new_faces)
        
        return True


    def subdivide_catmull_clark(self) -> bool:
        """
        Catmull-Clark subdivision - skema subdivisi khusus untuk quad/mesh umum.

        Aturan:
            - Bisa bekerja pada polygonal mesh (face dengan n ≥ 3).
            - Setiap face akan dibagi menjadi n quad baru (satu quad per vertex lama).
            - Posisi vertex baru dihitung dengan aturan Catmull-Clark:
                * Face point:
                    - Titik rata-rata (centroid) semua vertex di face.
                * Edge point:
                    - Jika interior: (v1+v2+f1+f2)/4
                    - Jika boundary: midpoint (v1+v2)/2
                * Vertex lama:
                    - Jika interior:
                    V' = (Q + 2R + (n-3)S) / n
                    dengan Q = rata-rata face point adjacent,
                        R = rata-rata midpoint edge adjacent,
                        S = posisi vertex lama,
                        n = valensi vertex.
                    - Jika boundary:
                    V' = 3/4 * S + 1/8 * (V_prev + V_next)

        Langkah implementasi (hint):
            1. Kumpulkan semua face lama, simpan posisi vertex.
            2. Hitung face points (centroid tiap face).
            3. Bangun struktur adjacency:
                - edge_adjacency: edge -> face & vertex
                - vertex_adjacency: vertex -> face, edge, neighbors, boundary_neighbors
            4. Hitung edge points:
                - Interior: (v1+v2+f1+f2)/4
                - Boundary: (v1+v2)/2
            5. Hitung posisi baru semua vertex lama (pakai aturan interior/boundary).
            6. Clear mesh lama (vertices, edges, faces, halfedges).
            7. Buat ulang:
                - Tambahkan vertex lama dengan posisi baru.
                - Tambahkan edge points.
                - Tambahkan face points.
            8. Buat quad baru untuk setiap face lama:
                - [curr, edge_next, face_center, edge_prev]
                (winding CCW harus dijaga).
            9. Panggil `MeshLoader._build_halfedge_connectivity` untuk rebuild mesh.

        Return:
            bool: True jika subdivision berhasil,
                False jika mesh kosong.
        """
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi subdivide_catmull_clark()")
