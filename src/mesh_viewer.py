"""
3D Mesh Viewer dengan Scene Support
Untuk Computer Graphics - Universitas Indonesia

Viewer untuk visualisasi scene dengan multiple objects, grid, dan axis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, List, Tuple
import math
from halfedge_mesh import HalfedgeMesh, Vertex, Edge, Face
from scene import Scene, SceneObject, Transform

class Camera:
    """Camera untuk 3D viewing"""
    def __init__(self):
        self.position = np.array([5.0, 5.0, 5.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])

        # Spherical coordinates untuk orbit camera
        self.radius = 10.0
        self.theta = 45.0  # Azimuth angle (degrees)
        self.phi = 45.0    # Elevation angle (degrees)

        # Projection parameters
        self.fov = 60.0  # Field of view (degrees)
        self.near = 0.1
        self.far = 1000.0

        self.update_position()

    def update_position(self):
        """Update camera position dari spherical coordinates"""
        # Convert degrees to radians
        theta_rad = math.radians(self.theta)
        phi_rad = math.radians(self.phi)

        # Calculate position
        x = self.radius * math.sin(phi_rad) * math.cos(theta_rad)
        y = self.radius * math.cos(phi_rad)
        z = self.radius * math.sin(phi_rad) * math.sin(theta_rad)

        self.position = np.array([x, y, z]) + self.target

    def orbit(self, d_theta: float, d_phi: float):
        """Orbit camera around target"""
        self.theta += d_theta
        self.theta = self.theta % 360

        self.phi += d_phi
        self.phi = max(1, min(179, self.phi))  # Clamp to avoid gimbal lock

        self.update_position()

    def zoom(self, delta: float):
        """Zoom in/out"""
        self.radius *= (1.0 - delta * 0.1)
        self.radius = max(1.0, min(100.0, self.radius))
        self.update_position()

    def pan(self, dx: float, dy: float):
        """Pan camera"""
        # Calculate right and up vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Move target
        self.target += right * dx * 0.01 * self.radius
        self.target += up * dy * 0.01 * self.radius

        self.update_position()

    def get_view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix"""
        # Look-at matrix
        f = self.target - self.position
        f = f / np.linalg.norm(f)

        s = np.cross(f, self.up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        view = np.eye(4)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[0, 3] = -np.dot(s, self.position)
        view[1, 3] = -np.dot(u, self.position)
        view[2, 3] = np.dot(f, self.position)

        return view

    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get 4x4 projection matrix"""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1

        return proj

    def reset(self):
        """Reset camera ke posisi default"""
        self.radius = 10.0
        self.theta = 45.0
        self.phi = 45.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.update_position()

    def focus_on_bounds(self, min_pt: np.ndarray, max_pt: np.ndarray):
        """Focus camera on bounding box"""
        center = (min_pt + max_pt) / 2
        size = np.linalg.norm(max_pt - min_pt)

        self.target = center
        self.radius = size * 1.5
        self.update_position()


class MeshViewer3D:
    """3D Viewer untuk mesh dengan TKinter"""

    def __init__(self, parent, scene: Scene = None):
        self.parent = parent
        self.scene = scene if scene else Scene()
        self.camera = Camera()

        # UI setup
        self.setup_ui()

        # Mouse control state
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_rotating = False
        self.is_panning = False

        # Display options
        self.show_wireframe = True
        self.show_faces = True
        self.show_vertices = False
        self.show_face_normals = False
        self.show_vertex_normals = False

        # Selection
        self.selected_object = None

        # Initial render
        self.render()

    def setup_ui(self):
        """Setup UI components"""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Canvas untuk rendering
        self.canvas = tk.Canvas(self.frame, bg='#262626', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)  # Middle button untuk pan
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        self.canvas.bind("<Configure>", self.on_resize)

        # Right click untuk selection
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)

        # Keyboard shortcuts
        self.parent.bind("<g>", lambda e: self.toggle_grid())
        self.parent.bind("<x>", lambda e: self.toggle_axes())  # Changed from 'a' to 'x'
        self.parent.bind("<e>", lambda e: self.toggle_wireframe())  # Changed from 'w' to 'e'
        self.parent.bind("<f>", lambda e: self.focus_on_selection())
        self.parent.bind("<r>", lambda e: self.reset_camera())
        self.parent.bind("<n>", lambda e: self.toggle_normals())  # Toggle normal visualization

    def project_point(self, point: np.ndarray, view_matrix: np.ndarray,
                     proj_matrix: np.ndarray) -> Optional[Tuple[float, float]]:
        """Project 3D point ke 2D screen coordinates"""
        # Transform to homogeneous coordinates
        p = np.ones(4)
        p[:3] = point

        # Apply view and projection matrices
        p = view_matrix @ p
        p = proj_matrix @ p

        # Perspective divide
        if abs(p[3]) < 0.0001:
            return None

        p = p / p[3]

        # Check if behind camera
        if p[2] < -1 or p[2] > 1:
            return None

        # Convert to screen coordinates
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x = (p[0] + 1) * width / 2
        y = (1 - p[1]) * height / 2

        return x, y

    def draw_grid(self, view_matrix: np.ndarray, proj_matrix: np.ndarray):
        """Draw grid on XZ plane"""
        if not self.scene.show_grid:
            return

        grid_size = self.scene.grid_size
        grid_spacing = self.scene.grid_spacing

        # Draw grid lines
        for i in range(-grid_size, grid_size + 1):
            # Lines parallel to X axis
            p1 = self.project_point(np.array([i * grid_spacing, 0, -grid_size * grid_spacing]),
                                   view_matrix, proj_matrix)
            p2 = self.project_point(np.array([i * grid_spacing, 0, grid_size * grid_spacing]),
                                   view_matrix, proj_matrix)

            if p1 and p2:
                color = '#404040' if i != 0 else '#606060'
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=1)

            # Lines parallel to Z axis
            p1 = self.project_point(np.array([-grid_size * grid_spacing, 0, i * grid_spacing]),
                                   view_matrix, proj_matrix)
            p2 = self.project_point(np.array([grid_size * grid_spacing, 0, i * grid_spacing]),
                                   view_matrix, proj_matrix)

            if p1 and p2:
                color = '#404040' if i != 0 else '#606060'
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=1)

    def draw_axes(self, view_matrix: np.ndarray, proj_matrix: np.ndarray):
        """Draw coordinate axes"""
        if not self.scene.show_axes:
            return

        origin = np.array([0, 0, 0])
        axis_length = 5.0

        # X axis (red)
        p_origin = self.project_point(origin, view_matrix, proj_matrix)
        p_x = self.project_point(np.array([axis_length, 0, 0]), view_matrix, proj_matrix)
        if p_origin and p_x:
            self.canvas.create_line(p_origin[0], p_origin[1], p_x[0], p_x[1],
                                   fill='#FF4444', width=3, arrow=tk.LAST)
            self.canvas.create_text(p_x[0] + 10, p_x[1], text="X", fill='#FF4444', font=('Arial', 12, 'bold'))

        # Y axis (green)
        p_y = self.project_point(np.array([0, axis_length, 0]), view_matrix, proj_matrix)
        if p_origin and p_y:
            self.canvas.create_line(p_origin[0], p_origin[1], p_y[0], p_y[1],
                                   fill='#44FF44', width=3, arrow=tk.LAST)
            self.canvas.create_text(p_y[0], p_y[1] - 10, text="Y", fill='#44FF44', font=('Arial', 12, 'bold'))

        # Z axis (blue)
        p_z = self.project_point(np.array([0, 0, axis_length]), view_matrix, proj_matrix)
        if p_origin and p_z:
            self.canvas.create_line(p_origin[0], p_origin[1], p_z[0], p_z[1],
                                   fill='#4444FF', width=3, arrow=tk.LAST)
            self.canvas.create_text(p_z[0] - 10, p_z[1], text="Z", fill='#4444FF', font=('Arial', 12, 'bold'))

    def draw_object(self, obj: SceneObject, view_matrix: np.ndarray, proj_matrix: np.ndarray):
        """Draw a scene object"""
        if not obj.visible:
            return

        mesh = obj.mesh

        # Apply object transform
        transform_matrix = obj.transform.get_matrix()

        # Project all vertices
        projected_vertices = {}
        for vertex in mesh.vertices:
            if vertex.deleted:
                continue

            # Transform vertex position
            p = np.ones(4)
            p[:3] = vertex.position
            p = transform_matrix @ p
            screen_pos = self.project_point(p[:3], view_matrix, proj_matrix)

            if screen_pos:
                projected_vertices[vertex.id] = screen_pos

        # Draw faces
        if self.show_faces and not obj.wireframe:
            for face in mesh.faces:
                if face.is_boundary or face.deleted:
                    continue

                vertices = face.vertices()
                if len(vertices) < 3:
                    continue

                # Get projected positions
                points = []
                for v in vertices:
                    if v.id in projected_vertices:
                        points.append(projected_vertices[v.id])

                if len(points) >= 3:
                    # Simple flat shading
                    flat_points = [coord for point in points for coord in point]

                    # Color based on selection
                    if obj.selected:
                        color = '#FFD700'  # Gold for selected
                    else:
                        r = int(obj.color[0] * 200 + 55)
                        g = int(obj.color[1] * 200 + 55)
                        b = int(obj.color[2] * 200 + 55)
                        color = f'#{r:02x}{g:02x}{b:02x}'

                    self.canvas.create_polygon(flat_points, fill=color, outline='')

        # Draw edges
        if self.show_wireframe or obj.wireframe:
            for edge in mesh.edges:
                if edge.deleted:
                    continue

                try:
                    v1, v2 = edge.vertices()
                    if v1 is None or v2 is None:
                        continue

                    if v1.id in projected_vertices and v2.id in projected_vertices:
                        p1 = projected_vertices[v1.id]
                        p2 = projected_vertices[v2.id]

                        # Edge color
                        if obj.selected:
                            color = '#FFFF00'  # Yellow for selected
                        else:
                            color = '#888888'

                        width = 2 if obj.selected else 1
                        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=width)
                except:
                    continue

        # Draw vertices
        if self.show_vertices:
            for vertex_id, pos in projected_vertices.items():
                radius = 4 if obj.selected else 3
                color = '#FF0000' if obj.selected else '#00FF00'
                self.canvas.create_oval(pos[0] - radius, pos[1] - radius,
                                       pos[0] + radius, pos[1] + radius,
                                       fill=color, outline='')

        # Draw object name if selected
        if obj.selected:
            # Get object center
            min_pt, max_pt = obj.get_bounding_box()
            center = (min_pt + max_pt) / 2
            center_screen = self.project_point(center, view_matrix, proj_matrix)

            if center_screen:
                self.canvas.create_text(center_screen[0], center_screen[1] - 30,
                                       text=obj.name, fill='yellow',
                                       font=('Arial', 12, 'bold'))

        # Draw normals if enabled
        if self.show_face_normals or self.show_vertex_normals:
            self.draw_normals(obj, view_matrix, proj_matrix, transform_matrix)

    def draw_normals(self, obj: SceneObject, view_matrix: np.ndarray,
                    proj_matrix: np.ndarray, transform_matrix: np.ndarray):
        """Draw face and vertex normals"""
        normal_length = 0.2  # Length of normal vectors

        # Draw face normals
        if self.show_face_normals:
            for face in obj.mesh.faces:
                if face.deleted or face.is_boundary:
                    continue

                # Get face center and normal
                vertices = face.vertices()
                if len(vertices) < 3:
                    continue

                center = np.mean([v.position for v in vertices], axis=0)
                normal = face.normal()  # Face has normal() method

                # Transform center
                p = np.ones(4)
                p[:3] = center
                center_transformed = (transform_matrix @ p)[:3]

                # Transform normal (use inverse transpose for normals)
                n = np.zeros(4)
                n[:3] = normal
                n[3] = 0
                normal_transformed = (transform_matrix @ n)[:3]
                normal_transformed = normal_transformed / np.linalg.norm(normal_transformed)

                # Project points
                p1 = self.project_point(center_transformed, view_matrix, proj_matrix)
                p2 = self.project_point(center_transformed + normal_transformed * normal_length,
                                      view_matrix, proj_matrix)

                if p1 and p2:
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                          fill='#00FF00', width=1, arrow=tk.LAST)

        # Draw vertex normals
        if self.show_vertex_normals:
            for vertex in obj.mesh.vertices:
                if vertex.deleted:
                    continue

                # Get vertex position and normal
                pos = vertex.position
                # Use the normal() method to calculate weighted normal
                normal = vertex.normal()  # Vertex has normal() method

                # Transform position
                p = np.ones(4)
                p[:3] = pos
                pos_transformed = (transform_matrix @ p)[:3]

                # Transform normal
                n = np.zeros(4)
                n[:3] = normal
                n[3] = 0
                normal_transformed = (transform_matrix @ n)[:3]
                normal_transformed = normal_transformed / np.linalg.norm(normal_transformed)

                # Project points
                p1 = self.project_point(pos_transformed, view_matrix, proj_matrix)
                p2 = self.project_point(pos_transformed + normal_transformed * normal_length,
                                      view_matrix, proj_matrix)

                if p1 and p2:
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                          fill='#FF00FF', width=1, arrow=tk.LAST)

    def render(self):
        """Render scene"""
        # Clear canvas
        self.canvas.delete("all")

        # Get view and projection matrices
        aspect_ratio = self.canvas.winfo_width() / max(1, self.canvas.winfo_height())
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect_ratio)

        # Draw grid first (behind everything)
        self.draw_grid(view_matrix, proj_matrix)

        # Draw all objects
        for obj in self.scene.objects:
            self.draw_object(obj, view_matrix, proj_matrix)

        # Draw axes on top
        self.draw_axes(view_matrix, proj_matrix)

        # Draw UI info
        self.draw_info()

    def draw_info(self):
        """Draw UI information"""
        info_text = []

        if self.scene.selected_object:
            info_text.append(f"Selected: {self.scene.selected_object.name}")
            stats = self.scene.selected_object.mesh.statistics()
            info_text.append(f"Vertices: {stats['vertices']}")
            info_text.append(f"Edges: {stats['edges']}")
            info_text.append(f"Faces: {stats['faces']}")

        info_text.append(f"Objects: {len(self.scene.objects)}")
        info_text.append(f"Grid: {'ON' if self.scene.show_grid else 'OFF'}")
        info_text.append(f"Axes: {'ON' if self.scene.show_axes else 'OFF'}")
        info_text.append(f"Wireframe: {'ON' if self.show_wireframe else 'OFF'}")

        # Normal display status
        if self.show_face_normals:
            info_text.append(f"Normals: Face")
        elif self.show_vertex_normals:
            info_text.append(f"Normals: Vertex")
        else:
            info_text.append(f"Normals: OFF")

        y = 10
        for text in info_text:
            self.canvas.create_text(10, y, text=text, anchor='nw',
                                   fill='white', font=('Consolas', 10))
            y += 15

    # Mouse event handlers
    def on_mouse_press(self, event):
        """Handle mouse press"""
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.is_rotating = True

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y

        if self.is_rotating:
            self.camera.orbit(dx * 0.5, dy * 0.5)
            self.render()

        self.mouse_x = event.x
        self.mouse_y = event.y

    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.is_rotating = False

    def on_middle_press(self, event):
        """Handle middle mouse press"""
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.is_panning = True

    def on_middle_drag(self, event):
        """Handle middle mouse drag"""
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y

        if self.is_panning:
            self.camera.pan(-dx, dy)
            self.render()

        self.mouse_x = event.x
        self.mouse_y = event.y

    def on_middle_release(self, event):
        """Handle middle mouse release"""
        self.is_panning = False

    def on_mouse_wheel(self, event):
        """Handle mouse wheel"""
        if event.delta:  # Windows
            delta = event.delta / 120.0
        elif event.num == 4:  # Linux scroll up
            delta = 1.0
        elif event.num == 5:  # Linux scroll down
            delta = -1.0
        else:
            delta = 0

        self.camera.zoom(delta * 0.1)
        self.render()

    def on_right_click(self, event):
        """Handle right click untuk object selection"""
        # Get view and projection matrices
        aspect_ratio = self.canvas.winfo_width() / max(1, self.canvas.winfo_height())
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect_ratio)

        # Try to select object
        selected = self.scene.get_object_at_position(
            event.x, event.y,
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
            view_matrix, proj_matrix
        )

        self.scene.select_object(selected)
        self.render()

    def on_resize(self, event):
        """Handle canvas resize"""
        self.render()

    # Display toggles
    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        self.show_wireframe = not self.show_wireframe
        self.render()

    def toggle_grid(self):
        """Toggle grid display"""
        self.scene.show_grid = not self.scene.show_grid
        self.render()

    def toggle_axes(self):
        """Toggle axes display"""
        self.scene.show_axes = not self.scene.show_axes
        self.render()

    def focus_on_selection(self):
        """Focus camera on selected object"""
        if self.scene.selected_object:
            min_pt, max_pt = self.scene.selected_object.get_bounding_box()
            self.camera.focus_on_bounds(min_pt, max_pt)
        else:
            # Focus on entire scene
            min_pt, max_pt = self.scene.get_scene_bounds()
            self.camera.focus_on_bounds(min_pt, max_pt)
        self.render()

    def reset_camera(self):
        """Reset camera to default"""
        self.camera.reset()
        self.render()

    def toggle_normals(self):
        """Toggle normal visualization"""
        if not self.show_face_normals and not self.show_vertex_normals:
            self.show_face_normals = True
        elif self.show_face_normals and not self.show_vertex_normals:
            self.show_face_normals = False
            self.show_vertex_normals = True
        else:
            self.show_vertex_normals = False
        self.render()
