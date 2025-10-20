#!/usr/bin/env python3
"""
Python Mesh Editor dengan Scene Support
Untuk Computer Graphics Assignment - Universitas Indonesia

Main aplikasi untuk mesh editing dengan support untuk multiple objects, grid, dan axis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

# Add src directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halfedge_mesh import HalfedgeMesh
from mesh_loader import MeshLoader, PrimitiveGenerator
from mesh_operations import MeshOperations
from mesh_viewer import MeshViewer3D
from scene import Scene, SceneObject
from object_dialog import ObjectInsertDialog, TransformDialog
import numpy as np

class MeshEditorApp:
    """Main aplikasi mesh editor"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Python Mesh Editor - Computer Graphics UI")
        self.root.geometry("1200x800")

        # Scene management
        self.scene = Scene()

        # Setup UI
        self.setup_ui()

        # Initialize dengan cube
        self.new_scene()

    def setup_ui(self):
        """Setup UI components"""
        # Menu bar
        self.setup_menu()

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Split into toolbar and content
        self.setup_toolbar(main_frame)

        # Content area with paned window
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Scene tree and properties
        left_panel = ttk.Frame(paned, width=300)
        paned.add(left_panel, weight=1)

        self.setup_scene_panel(left_panel)

        # Right panel - 3D Viewer
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=3)

        # Create viewer
        self.viewer = MeshViewer3D(right_panel, self.scene)

    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Scene", command=self.new_scene, accelerator="Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command(label="Import OBJ...", command=self.import_obj, accelerator="Ctrl+O")
        file_menu.add_command(label="Export OBJ...", command=self.export_obj, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Delete Object", command=self.delete_selected, accelerator="Del")
        edit_menu.add_command(label="Duplicate Object", command=self.duplicate_selected, accelerator="Ctrl+D")
        edit_menu.add_command(label="Transform Object...", command=self.transform_selected, accelerator="T")

        # Insert menu
        insert_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Insert", menu=insert_menu)
        insert_menu.add_command(label="New Object...", command=self.insert_object, accelerator="Shift+A")
        insert_menu.add_separator()
        insert_menu.add_command(label="Cube", command=lambda: self.quick_insert("cube"))
        insert_menu.add_command(label="Sphere", command=lambda: self.quick_insert("sphere"))
        insert_menu.add_command(label="Cylinder", command=lambda: self.quick_insert("cylinder"))
        insert_menu.add_command(label="Cone", command=lambda: self.quick_insert("cone"))
        insert_menu.add_command(label="Torus", command=lambda: self.quick_insert("torus"))

        # Mesh menu (Global Operations only)
        mesh_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Mesh", menu=mesh_menu)
        mesh_menu.add_command(label="Triangulate", command=lambda: self.mesh_operation("triangulate"))
        mesh_menu.add_command(label="Linear Subdivision", command=lambda: self.mesh_operation("linear_subdivide"))
        mesh_menu.add_command(label="Catmull-Clark Subdivision", command=lambda: self.mesh_operation("catmull_clark"))
        mesh_menu.add_command(label="Loop Subdivision", command=lambda: self.mesh_operation("loop"))

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Grid", command=self.toggle_grid, accelerator="G")
        view_menu.add_checkbutton(label="Show Axes", command=self.toggle_axes, accelerator="X")
        view_menu.add_checkbutton(label="Wireframe", command=self.toggle_wireframe, accelerator="E")
        view_menu.add_checkbutton(label="Show Normals", command=self.toggle_normals, accelerator="N")
        view_menu.add_separator()
        view_menu.add_command(label="Focus Selected", command=self.focus_selected, accelerator="F")
        view_menu.add_command(label="Reset Camera", command=self.reset_camera, accelerator="R")
        view_menu.add_separator()
        view_menu.add_command(label="Camera Controls:", state="disabled")
        view_menu.add_command(label="  W/S - Forward/Backward", state="disabled")
        view_menu.add_command(label="  A/D - Left/Right", state="disabled")
        view_menu.add_command(label="  Q/Z - Up/Down", state="disabled")

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Shortcuts", command=self.show_shortcuts)

        # Bind shortcuts
        self.root.bind("<Control-n>", lambda e: self.new_scene())
        self.root.bind("<Control-o>", lambda e: self.import_obj())
        self.root.bind("<Control-s>", lambda e: self.export_obj())
        self.root.bind("<Control-d>", lambda e: self.duplicate_selected())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Shift-A>", lambda e: self.insert_object())
        self.root.bind("<t>", lambda e: self.transform_selected())

    def setup_toolbar(self, parent):
        """Setup toolbar"""
        toolbar = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(fill=tk.X)

        # New Scene
        ttk.Button(toolbar, text="New", command=self.new_scene).pack(side=tk.LEFT, padx=2)

        # Import/Export
        ttk.Button(toolbar, text="Import", command=self.import_obj).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export", command=self.export_obj).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Insert objects
        ttk.Button(toolbar, text="+ Object", command=self.insert_object).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # Mesh operations
        ttk.Button(toolbar, text="Triangulate",
                  command=lambda: self.mesh_operation("triangulate")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Subdivide",
                  command=lambda: self.mesh_operation("linear_subdivide")).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, padx=5, fill=tk.Y)

        # View controls
        self.grid_var = tk.BooleanVar(value=True)
        self.axes_var = tk.BooleanVar(value=True)
        self.wireframe_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(toolbar, text="Grid", variable=self.grid_var,
                       command=self.toggle_grid).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="Axes", variable=self.axes_var,
                       command=self.toggle_axes).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="Wireframe", variable=self.wireframe_var,
                       command=self.toggle_wireframe).pack(side=tk.LEFT, padx=2)

    def setup_scene_panel(self, parent):
        """Setup scene panel dengan object list dan properties"""
        # Scene objects list
        list_frame = ttk.LabelFrame(parent, text="Scene Objects", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview untuk objects
        self.scene_tree = ttk.Treeview(list_frame, height=10)
        self.scene_tree.pack(fill=tk.BOTH, expand=True)

        # Configure columns
        self.scene_tree["columns"] = ("Type", "Vertices", "Faces")
        self.scene_tree.column("#0", width=150, minwidth=100)
        self.scene_tree.column("Type", width=80, minwidth=50)
        self.scene_tree.column("Vertices", width=60, minwidth=40)
        self.scene_tree.column("Faces", width=60, minwidth=40)

        self.scene_tree.heading("#0", text="Name")
        self.scene_tree.heading("Type", text="Type")
        self.scene_tree.heading("Vertices", text="Verts")
        self.scene_tree.heading("Faces", text="Faces")

        # Bind selection
        self.scene_tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Object controls
        controls_frame = ttk.Frame(list_frame)
        controls_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(controls_frame, text="Add", command=self.insert_object).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Delete", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Duplicate", command=self.duplicate_selected).pack(side=tk.LEFT, padx=2)

        # Properties panel
        props_frame = ttk.LabelFrame(parent, text="Properties", padding="10")
        props_frame.pack(fill=tk.X, padx=5, pady=5)

        # Name
        ttk.Label(props_frame, text="Name:").grid(row=0, column=0, sticky='w')
        self.name_entry = ttk.Entry(props_frame)
        self.name_entry.grid(row=0, column=1, sticky='ew', padx=(5, 0))
        self.name_entry.bind("<Return>", self.update_object_name)

        # Transform button
        ttk.Button(props_frame, text="Transform...",
                  command=self.transform_selected).grid(row=1, column=0, columnspan=2, pady=5)

        # Visibility
        self.visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(props_frame, text="Visible", variable=self.visible_var,
                       command=self.update_visibility).grid(row=2, column=0, columnspan=2)

        # Wireframe
        self.object_wireframe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(props_frame, text="Wireframe Only", variable=self.object_wireframe_var,
                       command=self.update_object_wireframe).grid(row=3, column=0, columnspan=2)

        props_frame.columnconfigure(1, weight=1)

        # Statistics
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="No object selected")
        self.stats_label.pack()

    def update_scene_tree(self):
        """Update scene tree dengan current objects"""
        # Clear tree
        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)

        # Add objects
        for obj in self.scene.objects:
            stats = obj.mesh.statistics()
            item = self.scene_tree.insert("", "end", text=obj.name,
                                         values=("Mesh", stats['vertices'], stats['faces']))

            # Select if it's selected in scene
            if obj.selected:
                self.scene_tree.selection_set(item)

    def on_tree_select(self, event):
        """Handle tree selection"""
        selection = self.scene_tree.selection()
        if not selection:
            self.scene.select_object(None)
        else:
            # Get selected index
            idx = self.scene_tree.index(selection[0])
            if idx < len(self.scene.objects):
                self.scene.select_object(self.scene.objects[idx])

        self.update_properties()
        self.viewer.render()

    def update_properties(self):
        """Update properties panel untuk selected object"""
        if self.scene.selected_object:
            obj = self.scene.selected_object
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, obj.name)
            self.visible_var.set(obj.visible)
            self.object_wireframe_var.set(obj.wireframe)

            # Update stats
            stats = obj.mesh.statistics()
            surface_area = obj.mesh.surface_area()
            volume = obj.mesh.volume()

            stats_text = f"Vertices: {stats['vertices']}\n"
            stats_text += f"Edges: {stats['edges']}\n"
            stats_text += f"Faces: {stats['faces']}\n"
            stats_text += f"Triangles: {stats['triangles']}\n"
            stats_text += f"Quads: {stats['quads']}\n"
            stats_text += f"─────────────────\n"
            stats_text += f"Surface Area: {surface_area:.3f}\n"
            stats_text += f"Volume: {volume:.3f}"
            self.stats_label.config(text=stats_text)
        else:
            self.name_entry.delete(0, tk.END)
            self.stats_label.config(text="No object selected")

    def update_object_name(self, event=None):
        """Update selected object name"""
        if self.scene.selected_object:
            self.scene.selected_object.name = self.name_entry.get()
            self.update_scene_tree()

    def update_visibility(self):
        """Update selected object visibility"""
        if self.scene.selected_object:
            self.scene.selected_object.visible = self.visible_var.get()
            self.viewer.render()

    def update_object_wireframe(self):
        """Update selected object wireframe mode"""
        if self.scene.selected_object:
            self.scene.selected_object.wireframe = self.object_wireframe_var.get()
            self.viewer.render()

    # Scene operations
    def new_scene(self):
        """Create new scene dengan default cube"""
        self.scene.clear()

        # Add default cube
        cube = PrimitiveGenerator.create_cube(2.0)
        obj = self.scene.add_object(cube, "Default Cube")
        self.scene.select_object(obj)

        self.update_scene_tree()
        self.update_properties()
        self.viewer.render()

    def insert_object(self):
        """Insert new object dengan dialog"""
        dialog = ObjectInsertDialog(self.root)
        result = dialog.show()

        if result:
            obj = self.scene.add_object(result['mesh'], result['name'])
            obj.transform.position = result['position']
            obj.transform.rotation = result['rotation']
            obj.transform.scale = result['scale']
            obj.color = result['color']

            self.scene.select_object(obj)
            self.update_scene_tree()
            self.update_properties()
            self.viewer.render()

    def quick_insert(self, obj_type: str):
        """Quick insert primitive without dialog"""
        mesh = None
        name = obj_type.capitalize()

        if obj_type == "cube":
            mesh = PrimitiveGenerator.create_cube(2.0)
        elif obj_type == "sphere":
            mesh = PrimitiveGenerator.create_sphere(1.0, 2)
        elif obj_type == "cylinder":
            mesh = PrimitiveGenerator.create_cylinder(1.0, 2.0, 16)
        elif obj_type == "cone":
            mesh = PrimitiveGenerator.create_cone(1.0, 2.0, 16)
        elif obj_type == "torus":
            mesh = PrimitiveGenerator.create_torus(1.0, 0.3, 16, 8)

        if mesh:
            obj = self.scene.add_object(mesh, name)
            # Offset sedikit dari origin
            obj.transform.position = np.random.uniform(-2, 2, 3)
            self.scene.select_object(obj)
            self.update_scene_tree()
            self.update_properties()
            self.viewer.render()

    def delete_selected(self):
        """Delete selected object"""
        if self.scene.selected_object:
            if messagebox.askyesno("Delete Object",
                                  f"Delete '{self.scene.selected_object.name}'?"):
                self.scene.remove_object(self.scene.selected_object)
                self.update_scene_tree()
                self.update_properties()
                self.viewer.render()

    def duplicate_selected(self):
        """Duplicate selected object"""
        if self.scene.selected_object:
            new_obj = self.scene.duplicate_selected()
            if new_obj:
                self.scene.select_object(new_obj)
                self.update_scene_tree()
                self.update_properties()
                self.viewer.render()

    def transform_selected(self):
        """Open transform dialog untuk selected object"""
        if not self.scene.selected_object:
            messagebox.showwarning("No Selection", "Please select an object first")
            return

        obj = self.scene.selected_object
        current_transform = {
            'position': obj.transform.position,
            'rotation': obj.transform.rotation,
            'scale': obj.transform.scale
        }

        dialog = TransformDialog(self.root, current_transform)
        result = dialog.show()

        if result:
            obj.transform.position = result['position']
            obj.transform.rotation = result['rotation']
            obj.transform.scale = result['scale']
            self.viewer.render()

    # Mesh operations
    def mesh_operation(self, operation: str):
        """Perform mesh operation on selected object"""
        if not self.scene.selected_object:
            messagebox.showwarning("No Selection", "Please select an object first")
            return

        mesh = self.scene.selected_object.mesh
        ops = MeshOperations(mesh)

        try:
            if operation == "triangulate":
                count = ops.triangulate()
                messagebox.showinfo("Triangulate", f"Triangulated {count} faces")
            elif operation == "linear_subdivide":
                if ops.subdivide_linear():
                    messagebox.showinfo("Subdivide", "Linear subdivision completed")
                else:
                    messagebox.showwarning("Subdivide", "Linear subdivision failed: No valid faces found")
            elif operation == "catmull_clark":
                if ops.subdivide_catmull_clark():
                    messagebox.showinfo("Subdivide", "Catmull-Clark subdivision completed")
                else:
                    messagebox.showwarning("Subdivide", "Catmull-Clark subdivision failed: No valid faces found")
            elif operation == "loop":
                if ops.subdivide_loop():
                    messagebox.showinfo("Subdivide", "Loop subdivision completed")
                else:
                    messagebox.showwarning("Subdivide", "Loop subdivision failed: Requires triangular mesh. Please triangulate first.")
            else:
                messagebox.showwarning("Not Implemented",
                                      f"Operation '{operation}' not implemented yet")

            self.update_scene_tree()
            self.update_properties()
            self.viewer.render()

        except Exception as e:
            messagebox.showerror("Operation Error", f"Error: {str(e)}")

    # File operations
    def import_obj(self):
        """Import OBJ file"""
        filename = filedialog.askopenfilename(
            title="Import OBJ",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            mesh = MeshLoader.load_obj(filename)
            if mesh:
                name = os.path.basename(filename).split('.')[0]
                obj = self.scene.add_object(mesh, name)
                self.scene.select_object(obj)
                self.update_scene_tree()
                self.update_properties()
                self.viewer.render()
            else:
                messagebox.showerror("Import Error", "Failed to load OBJ file")

    def export_obj(self):
        """Export selected object to OBJ"""
        if not self.scene.selected_object:
            messagebox.showwarning("No Selection", "Please select an object to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export OBJ",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            # Export the transformed mesh
            mesh = self.scene.selected_object.get_transformed_mesh()
            if MeshLoader.save_obj(mesh, filename):
                messagebox.showinfo("Export", "Export successful!")
            else:
                messagebox.showerror("Export Error", "Failed to export OBJ file")

    # View operations
    def toggle_grid(self):
        """Toggle grid display"""
        self.scene.show_grid = self.grid_var.get()
        self.viewer.render()

    def toggle_axes(self):
        """Toggle axes display"""
        self.scene.show_axes = self.axes_var.get()
        self.viewer.render()

    def toggle_wireframe(self):
        """Toggle wireframe display"""
        self.viewer.show_wireframe = self.wireframe_var.get()
        self.viewer.render()

    def focus_selected(self):
        """Focus camera on selected object"""
        self.viewer.focus_on_selection()

    def reset_camera(self):
        """Reset camera to default"""
        self.viewer.reset_camera()

    def toggle_normals(self):
        """Toggle normal visualization"""
        self.viewer.toggle_normals()

    # Help
    def show_about(self):
        """Show about dialog"""
        about_text = """Python Mesh Editor v1.0

Dibuat untuk Computer Graphics Course
Universitas Indonesia

Berdasarkan Stanford Cardinal3D
Assignment 2: MeshEdit

2024"""
        messagebox.showinfo("About", about_text)

    def show_shortcuts(self):
        """Show shortcuts dialog"""
        shortcuts = """Keyboard Shortcuts:

File:
Ctrl+N - New Scene
Ctrl+O - Import OBJ
Ctrl+S - Export OBJ

Edit:
Delete - Delete Selected Object
Ctrl+D - Duplicate Object
T - Transform Object
Shift+A - Insert New Object

View:
G - Toggle Grid
X - Toggle Axes
E - Toggle Wireframe
N - Toggle Normals (Face/Vertex)
F - Focus on Selected
R - Reset Camera

Mouse Controls:
Left Click + Drag - Rotate Camera
Middle Click + Drag - Pan Camera
Scroll Wheel - Zoom In/Out
Right Click - Select Object"""
        messagebox.showinfo("Shortcuts", shortcuts)

    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = MeshEditorApp()
    app.run()