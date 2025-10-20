"""
Halfedge Mesh Data Structure Implementation
Untuk Computer Graphics - Universitas Indonesia

Struktur data halfedge adalah representasi mesh yang efisien untuk operasi topologi.
Setiap edge dibagi menjadi dua halfedge yang berlawanan arah.
"""

import numpy as np
from typing import Optional, List, Tuple
import uuid

class HalfedgeElement:
    """Base class untuk semua elemen halfedge mesh"""
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]  # ID unik untuk debugging
        self.is_boundary = False
        self._deleted = False

    def mark_deleted(self):
        """Tandai elemen sebagai deleted (untuk lazy deletion)"""
        self._deleted = True

    @property
    def deleted(self):
        return self._deleted

class Vertex(HalfedgeElement):
    """Vertex dalam halfedge mesh"""
    def __init__(self, position=None):
        super().__init__()
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0])
        self.halfedge: Optional['Halfedge'] = None  # Outgoing halfedge

    def degree(self) -> int:
        """Hitung degree (valence) dari vertex"""
        if not self.halfedge:
            return 0

        count = 0
        h = self.halfedge
        start = h
        while True:
            count += 1
            h = h.twin.next
            if h == start:
                break
        return count


    def neighbors(self) -> List['Vertex']:
        """
        Dapatkan semua vertex tetangga (adjacent vertices) dari vertex ini.

        Definisi:
            - Vertex tetangga adalah semua vertex yang terhubung langsung
            oleh satu edge ke vertex ini.
        
        Cara menghitung (hint):
            - Mulai dari self.halfedge (halfedge yang keluar dari vertex ini).
            - Untuk setiap langkah:
                * Ambil target dari h.twin (itu adalah vertex tetangga).
                * Simpan vertex tersebut dalam list.
                * Lanjutkan traversal ke h.twin.next.
            - Ulangi sampai kembali ke halfedge awal.
            - Pastikan traversal berhenti, jangan sampai loop infinite.

        Return:
            List[Vertex]: daftar vertex tetangga.

        Catatan:
            - Jika vertex tidak punya halfedge (isolated), return list kosong.
        """
        if not self.halfedge:
            return []
            
        neighbors_list = []
        start_halfedge = self.halfedge
        current_halfedge = self.halfedge

        while True:
            # Simpan vertex ke dalam list
            if current_halfedge and current_halfedge.vertex:
                neighbors_list.append(current_halfedge.vertex)
            else:
                break

            # Pindah ke halfedge.twin.next
            if not current_halfedge.twin:
                break 
            current_halfedge = current_halfedge.twin.next

            # Berhenti jika sudah kembali ke awal
            if current_halfedge == start_halfedge:
                break
            
            if current_halfedge is None:
                break

        return neighbors_list


    def normal(self) -> np.ndarray:
        """
        Hitung normal vertex sebagai rata-rata berbobot dari normal face yang adjacent.

        Definisi:
            - Normal vertex biasanya dihitung dengan merata-ratakan normal
            semua face yang berbagi vertex ini.
            - Pembobotan bisa berdasarkan luas face (area-weighted) agar hasilnya lebih akurat.

        Cara menghitung (hint):
            - Mulai dari self.halfedge.
            - Untuk setiap face yang adjacent ke vertex ini:
                * Ambil normal face (f.normal()).
                * Ambil luas face (f.area()).
                * Tambahkan (normal_face * area_face) ke accumulator.
            - Setelah semua face dihitung, normalisasi vektor normal hasil.
            - Jika vertex tidak punya face valid, return vektor default, misalnya (0, 1, 0).

        Return:
            np.ndarray: vektor normal 3D (sudah ternormalisasi).

        Catatan:
            - Traversal bisa dilakukan dengan h = h.twin.next.
            - Berhenti jika kembali ke halfedge awal.
        """
        # TODO: Implement this
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi normal()")


    def faces(self) -> List['Face']:
        """
        Dapatkan semua face yang adjacent ke vertex ini.

        Definisi:
            - Face adjacent = face yang menggunakan vertex ini sebagai salah satu sudutnya.

        Cara menghitung (hint):
            - Mulai dari self.halfedge.
            - Untuk setiap langkah:
                * Ambil face dari halfedge (h.face).
                * Jika bukan boundary, tambahkan ke list.
                * Lanjutkan traversal ke h.twin.next.
            - Ulangi sampai kembali ke halfedge awal.

        Return:
            List[Face]: daftar face yang adjacent.

        Catatan:
            - Jika vertex tidak punya halfedge, return list kosong.
            - Hati-hati jangan masukkan boundary face.
        """
        if not self.halfedge:
            return []
            
        faces_list = []
        start_halfedge = self.halfedge
        current_halfedge = self.halfedge

        while True:
            # Ambil face dari halfedge saat ini
            if current_halfedge and current_halfedge.face:
                # Skip boundary faces
                if not current_halfedge.face.is_boundary:
                    faces_list.append(current_halfedge.face)

            # Pindah ke halfedge.twin.next
            if not current_halfedge.twin:
                break 
            current_halfedge = current_halfedge.twin.next

            # Berhenti jika sudah kembali ke awal
            if current_halfedge == start_halfedge:
                break
            
            if current_halfedge is None:
                break

        return faces_list


class Edge(HalfedgeElement):
    """Edge dalam halfedge mesh"""
    def __init__(self):
        super().__init__()
        self.halfedge: Optional['Halfedge'] = None

    def vertices(self) -> Tuple[Vertex, Vertex]:
        """Dapatkan kedua vertex dari edge"""
        if not self.halfedge or not self.halfedge.twin:
            return (None, None)
        h = self.halfedge
        # h.vertex is the target, h.twin.vertex is actually h's target too
        # The source of h is the target of h.twin
        v1 = h.twin.vertex if h.twin else None
        v2 = h.vertex
        return (v1, v2)

    def center(self) -> np.ndarray:
        """Hitung posisi tengah edge"""
        v1, v2 = self.vertices()
        if v1 is None or v2 is None:
            return np.array([0.0, 0.0, 0.0])
        return (v1.position + v2.position) / 2.0

class Face(HalfedgeElement):
    """Face dalam halfedge mesh"""
    def __init__(self):
        super().__init__()
        self.halfedge: Optional['Halfedge'] = None

    def vertices(self) -> List['Vertex']:
        """
        Dapatkan semua vertex yang menyusun face ini.

        Cara menghitung (hint):
            - Mulai dari self.halfedge (salah satu halfedge pada face ini).
            - Traversal dengan mengikuti pointer h.next.
            - Pada setiap langkah, ambil h.vertex.
            - Ulangi sampai kembali ke halfedge awal.
        
        Return:
            List[Vertex]: daftar vertex yang menyusun face.

        Catatan:
            - Jika face tidak punya halfedge (kosong), return list kosong.
            - Urutan vertex mengikuti urutan traversal halfedge (loop tertutup).
        """
        if not self.halfedge:
            return []

        vertices_list = []
        start_halfedge = self.halfedge
        current_halfedge = self.halfedge

        while True:
            if current_halfedge and current_halfedge.vertex:
                vertices_list.append(current_halfedge.vertex)
            else:
                break
        
            # Bergerak ke halfedge berikutnya di dalam face
            current_halfedge = current_halfedge.next
        
            # Berhenti jika sudah kembali ke awal 
            if current_halfedge == start_halfedge:
                break

        return vertices_list

    def edges(self) -> List['Edge']:
        """
        Dapatkan semua edge yang menyusun face ini.

        Cara menghitung (hint):
            - Mulai dari self.halfedge.
            - Traversal dengan mengikuti pointer h.next.
            - Pada setiap langkah, ambil h.edge.
            - Ulangi sampai kembali ke halfedge awal.
        
        Return:
            List[Edge]: daftar edge yang menyusun face.

        Catatan:
            - Jika face tidak punya halfedge (kosong), return list kosong.
            - Urutan edge mengikuti urutan traversal halfedge.
        """
        if not self.halfedge:
            return []

        edges_list = []
        start_halfedge = self.halfedge
        current_halfedge = self.halfedge

        while True:
            if current_halfedge and current_halfedge.edge:
                edges_list.append(current_halfedge.edge)
            else:
                break
            
            # Bergerak ke halfedge berikutnya di dalam face
            current_halfedge = current_halfedge.next
            
            # Berhenti jika sudah kembali ke awal
            if current_halfedge == start_halfedge:
                break
                
        return edges_list

    def normal(self) -> np.ndarray:
        """
        Hitung normal dari face ini.

        Definisi:
            - Normal face adalah vektor tegak lurus terhadap bidang face.
            - Untuk polygon umum (>= 3 vertex), bisa dihitung dengan metode Newell.

        Cara menghitung (hint):
            - Ambil semua vertex dari face (self.vertices()).
            - Jika jumlah vertex < 3, return default normal (misalnya (0,0,1)).
            - Gunakan Newell's method:
                n.x += (y_i - y_{i+1}) * (z_i + z_{i+1})
                n.y += (z_i - z_{i+1}) * (x_i + x_{i+1})
                n.z += (x_i - x_{i+1}) * (y_i + y_{i+1})
            dengan indeks i berjalan keliling vertex.
            - Normalisasi vektor hasil (bagi dengan panjang vektor).
        
        Return:
            np.ndarray: vektor normal 3D (sudah ternormalisasi).

        Catatan:
            - Jika panjang normal hampir nol, gunakan default (0,0,1).
        """
        verts = self.vertices()
        
        # Jika jumlah vertex < 3, return default normal
        if len(verts) < 3:
            return np.array([0.0, 0.0, 1.0])

        # Inisialisasi
        normal_vec = np.array([0.0, 0.0, 0.0])
        num_verts = len(verts)

        # Newell's method
        for i in range(num_verts):
            v_i = verts[i].position
            v_i_plus_1 = verts[(i + 1) % num_verts].position
            
            normal_vec[0] += (v_i[1] - v_i_plus_1[1]) * (v_i[2] + v_i_plus_1[2])
            normal_vec[1] += (v_i[2] - v_i_plus_1[2]) * (v_i[0] + v_i_plus_1[0])
            normal_vec[2] += (v_i[0] - v_i_plus_1[0]) * (v_i[1] + v_i_plus_1[1])

        # Normalisasi 
        magnitude = np.linalg.norm(normal_vec)
        
        # Handle panjang normal hampir nol
        if magnitude < 1e-9:
            return np.array([0.0, 0.0, 1.0])
            
        return normal_vec / magnitude

    def area(self) -> float:
        """
        Hitung luas area dari face ini.

        Cara menghitung (hint):
            - Ambil semua vertex dari face (self.vertices()).
            - Jika jumlah vertex < 3, return 0.0.
            - Lakukan triangulasi fan:
                * Pilih vertex pertama sebagai pusat (v0).
                * Untuk setiap pasangan (v_i, v_{i+1}) bentuk segitiga (v0, v_i, v_{i+1}).
                * Luas segitiga = 0.5 * || (v_i - v0) x (v_{i+1} - v0) ||.
                * Jumlahkan semua luas segitiga.
        
        Return:
            float: luas area face.
        """
        verts = self.vertices()
        
        # Jika jumlah vertex < 3, return 0.0
        if len(verts) < 3:
            return 0.0

        # Pilih vertex pertama sebagai pusat (v0)
        v0 = verts[0].position
        total_area = 0.0
        num_verts = len(verts)

        # Iterasi untuk membentuk segitiga (v0, v_i, v_{i+1})
        for i in range(1, num_verts - 1):
            v_i = verts[i].position
            v_i_plus_1 = verts[i + 1].position
            
            # Buat dua vektor dari pusat v0
            vec_a = v_i - v0
            vec_b = v_i_plus_1 - v0
            
            # Hitung cross product
            cross_product = np.cross(vec_a, vec_b)
            
            # Luas segitiga
            triangle_area = 0.5 * np.linalg.norm(cross_product)
            
            # Jumlahkan semua luas segitiga
            total_area += triangle_area
            
        return total_area

    def is_triangle(self) -> bool:
        """Check apakah face adalah triangle"""
        return len(self.vertices()) == 3

    def is_quad(self) -> bool:
        """Check apakah face adalah quad"""
        return len(self.vertices()) == 4

class Halfedge(HalfedgeElement):
    """Halfedge dalam halfedge mesh"""
    def __init__(self):
        super().__init__()
        self.next: Optional['Halfedge'] = None
        self.twin: Optional['Halfedge'] = None
        self.vertex: Optional[Vertex] = None  # Target vertex
        self.edge: Optional[Edge] = None
        self.face: Optional[Face] = None

    def source(self) -> Vertex:
        """Dapatkan source vertex dari halfedge"""
        if self.twin:
            return self.twin.vertex
        return None

    def target(self) -> Vertex:
        """Dapatkan target vertex dari halfedge"""
        return self.vertex

class HalfedgeMesh:
    """
    Halfedge Mesh - struktur data utama untuk mesh editing
    """
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []
        self.halfedges: List[Halfedge] = []
        self.boundary_faces: List[Face] = []  # Virtual boundary faces

    def add_vertex(self, position) -> Vertex:
        """Tambah vertex baru ke mesh"""
        v = Vertex(position)
        self.vertices.append(v)
        return v

    def add_edge(self) -> Edge:
        """Tambah edge baru ke mesh"""
        e = Edge()
        self.edges.append(e)
        return e

    def add_face(self) -> Face:
        """Tambah face baru ke mesh"""
        f = Face()
        self.faces.append(f)
        return f

    def add_halfedge(self) -> Halfedge:
        """Tambah halfedge baru ke mesh"""
        h = Halfedge()
        self.halfedges.append(h)
        return h

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Hitung bounding box dari mesh"""
        if not self.vertices:
            return (np.zeros(3), np.zeros(3))

        positions = np.array([v.position for v in self.vertices])
        return (np.min(positions, axis=0), np.max(positions, axis=0))

    def center(self) -> np.ndarray:
        """Hitung center dari mesh"""
        if not self.vertices:
            return np.zeros(3)

        positions = np.array([v.position for v in self.vertices])
        return np.mean(positions, axis=0)

    def statistics(self) -> dict:
        """
        Dapatkan statistik mesh ini.

        Statistik yang dihitung:
            - vertices   : jumlah vertex yang tidak dihapus (deleted = False).
            - edges      : jumlah edge yang tidak dihapus.
            - faces      : jumlah face yang tidak dihapus dan bukan boundary.
            - halfedges  : jumlah halfedge yang tidak dihapus.
            - triangles  : jumlah face yang berbentuk segitiga.
            - quads      : jumlah face yang berbentuk segiempat.
            - other_faces: jumlah face dengan sisi > 4.

        Hint:
            - Gunakan comprehension list dengan filter (not deleted, not boundary).
            - Gunakan method f.is_triangle() dan f.is_quad() untuk klasifikasi.

        Return:
            dict: dictionary dengan semua statistik mesh.
        """
        # TODO: Implement this
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi statistics()")

    def surface_area(self) -> float:
        """
        Hitung total luas permukaan mesh.

        Cara menghitung (hint):
            - Iterasi semua face dalam mesh.
            - Abaikan face yang boundary atau deleted.
            - Panggil f.area() untuk setiap face.
            - Jumlahkan semua hasilnya.

        Return:
            float: luas permukaan total.
        """
        # TODO: Implement this
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi surface_area()")

    def volume(self) -> float:
        """
        Hitung volume mesh menggunakan Divergence Theorem.
        (Asumsi mesh tertutup dan terorientasi dengan benar.)

        Cara menghitung (hint):
            - Iterasi semua face non-boundary.
            - Ambil vertex dari face.
            - Jika face punya >= 3 vertex:
                * Lakukan triangulasi fan dari vertex[0].
                * Untuk setiap segitiga (v0, v1, v2), hitung kontribusi volume:
                    vol = dot(v0, cross(v1, v2)) / 6.0
                * Jumlahkan semua kontribusi.
            - Ambil nilai absolut dari volume.

        Return:
            float: volume mesh (selalu non-negatif).
        """
        # TODO: Implement this
        raise NotImplementedError("Mahasiswa harus mengimplementasikan fungsi volume()")
