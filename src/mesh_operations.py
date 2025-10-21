"""
Mesh Operations Implementation
Untuk Computer Graphics - Universitas Indonesia

Implementasi operasi-operasi mesh:
- Global Operations: triangulate, subdivide
"""

import numpy as np
from typing import List, Optional
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
        # Kumpulkan vertex yang valid
        valid_vertices = [v for v in self.mesh.vertices if not v.deleted]
        if not valid_vertices:
            return 0

        # List baru untuk face, berisi objek Vertex
        new_faces_list_of_objects: List[List[Vertex]] = []
        faces_triangulated_count = 0
        
        faces_to_process = self.mesh.faces.copy()

        # Iterasi semua face
        for f in faces_to_process:
            if f.deleted or f.is_boundary:
                continue

            verts = f.vertices()
            num_verts = len(verts)

            if num_verts == 3:
                # Jika sudah segitiga, tambahkan
                new_faces_list_of_objects.append(verts)
            
            elif num_verts > 3:
                # Jika n > 3, lakukan fan triangulation
                faces_triangulated_count += 1
                v0 = verts[0] # Pivot
                
                for i in range(1, num_verts - 1):
                    new_tri = [v0, verts[i], verts[i+1]]
                    new_faces_list_of_objects.append(new_tri)

        if faces_triangulated_count == 0:
            return 0
 
        # Hapus data topologi lama
        self.mesh.vertices.clear()
        self.mesh.edges.clear()
        self.mesh.faces.clear()
        self.mesh.halfedges.clear()
        self.mesh.boundary_faces.clear()
        
        # Tetapkan list vertex baru
        self.mesh.vertices = valid_vertices

        # Panggil builder
        loader = MeshLoader()
        loader._build_halfedge_connectivity(self.mesh, new_faces_list_of_objects)

        return faces_triangulated_count

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
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi subdivide_linear()")

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
        pass


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
        pass
