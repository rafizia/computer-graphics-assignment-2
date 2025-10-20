"""
Object Insertion Dialog
Untuk Computer Graphics - Universitas Indonesia

Dialog untuk insert dan configure objects sebelum ditambahkan ke scene
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional
from mesh_loader import PrimitiveGenerator, MeshLoader
from halfedge_mesh import HalfedgeMesh

class ObjectInsertDialog:
    """Dialog untuk insert object baru ke scene"""

    def __init__(self, parent):
        self.parent = parent
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Insert New Object")
        self.dialog.geometry("450x650")
        self.dialog.resizable(True, True)
        self.dialog.minsize(450, 500)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Object type
        self.object_type = tk.StringVar(value="cube")

        # Object parameters
        self.param_vars = {}

        # Transform parameters
        self.position_vars = {
            'x': tk.DoubleVar(value=0.0),
            'y': tk.DoubleVar(value=0.0),
            'z': tk.DoubleVar(value=0.0)
        }

        self.rotation_vars = {
            'x': tk.DoubleVar(value=0.0),
            'y': tk.DoubleVar(value=0.0),
            'z': tk.DoubleVar(value=0.0)
        }

        self.scale_vars = {
            'x': tk.DoubleVar(value=1.0),
            'y': tk.DoubleVar(value=1.0),
            'z': tk.DoubleVar(value=1.0)
        }

        self.name_var = tk.StringVar(value="")
        self.color_vars = {
            'r': tk.DoubleVar(value=0.7),
            'g': tk.DoubleVar(value=0.7),
            'b': tk.DoubleVar(value=0.7)
        }

        self.setup_ui()

        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

    def setup_ui(self):
        """Setup UI components"""
        # Add separator line
        ttk.Separator(self.dialog, orient='horizontal').pack(side="bottom", fill="x")

        self.button_container = ttk.Frame(self.dialog)
        self.button_container.pack(side="bottom", fill="x", padx=10, pady=(5, 10))

        # Content frame for canvas and scrollbar
        content_frame = ttk.Frame(self.dialog)
        content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 0))

        # Create canvas and scrollbar for scrollable content
        canvas = tk.Canvas(content_frame)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar inside content frame
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main container (now inside scrollable frame)
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            if event.delta:  # Windows
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif event.num == 4:  # Linux scroll up
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Linux scroll down
                canvas.yview_scroll(1, "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind("<Button-4>", _on_mousewheel)    # Linux scroll up
        canvas.bind("<Button-5>", _on_mousewheel)    # Linux scroll down

        # Make the scrollable frame expand to fill canvas width
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas.find_all()[0], width=e.width))

        # Object Type Selection
        type_frame = ttk.LabelFrame(main_frame, text="Object Type", padding="10")
        type_frame.pack(fill=tk.X, pady=(0, 10))

        object_types = [
            ("Cube", "cube"),
            ("Sphere", "sphere"),
            ("Cylinder", "cylinder"),
            ("Cone", "cone"),
            ("Torus", "torus"),
            ("Tetrahedron", "tetrahedron"),
            ("Octahedron", "octahedron"),
            ("Icosahedron", "icosahedron")
        ]

        for i, (label, value) in enumerate(object_types):
            row = i // 4
            col = i % 4
            ttk.Radiobutton(type_frame, text=label, variable=self.object_type,
                           value=value, command=self.on_type_change).grid(
                               row=row, column=col, sticky='w', padx=5, pady=2)

        # Object Parameters
        self.params_frame = ttk.LabelFrame(main_frame, text="Object Parameters", padding="10")
        self.params_frame.pack(fill=tk.X, pady=(0, 10))

        self.update_parameter_fields()

        # Object Name
        name_frame = ttk.LabelFrame(main_frame, text="Object Name", padding="10")
        name_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Entry(name_frame, textvariable=self.name_var, width=30).pack(fill=tk.X)

        # Transform
        transform_frame = ttk.LabelFrame(main_frame, text="Transform", padding="10")
        transform_frame.pack(fill=tk.X, pady=(0, 10))

        # Position
        pos_label = ttk.Label(transform_frame, text="Position:")
        pos_label.grid(row=0, column=0, sticky='w', pady=2)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=0, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.position_vars[axis],
                     width=8).grid(row=0, column=2+i*2, sticky='w', padx=(0, 5))

        # Rotation
        rot_label = ttk.Label(transform_frame, text="Rotation:")
        rot_label.grid(row=1, column=0, sticky='w', pady=2)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=1, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.rotation_vars[axis],
                     width=8).grid(row=1, column=2+i*2, sticky='w', padx=(0, 5))

        # Scale
        scale_label = ttk.Label(transform_frame, text="Scale:")
        scale_label.grid(row=2, column=0, sticky='w', pady=2)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=2, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.scale_vars[axis],
                     width=8).grid(row=2, column=2+i*2, sticky='w', padx=(0, 5))

        # Uniform scale button
        ttk.Button(transform_frame, text="Uniform Scale",
                  command=self.set_uniform_scale).grid(row=3, column=0, columnspan=7,
                                                       pady=(5, 0))

        # Color
        color_frame = ttk.LabelFrame(main_frame, text="Color", padding="10")
        color_frame.pack(fill=tk.X, pady=(0, 10))

        for i, (comp, label) in enumerate([('r', 'Red'), ('g', 'Green'), ('b', 'Blue')]):
            ttk.Label(color_frame, text=f"{label}:").grid(row=0, column=i*2, sticky='e', padx=(0, 2))
            scale = ttk.Scale(color_frame, from_=0, to=1, variable=self.color_vars[comp],
                            orient='horizontal', length=100, command=lambda v: self.update_color_preview())
            scale.grid(row=0, column=i*2+1, padx=(0, 10))

        # Color preview
        self.color_preview = tk.Canvas(color_frame, width=30, height=30, bg='gray')
        self.color_preview.grid(row=0, column=6, padx=(10, 0))
        self.update_color_preview()

        # Buttons - centered at bottom
        button_inner_frame = ttk.Frame(self.button_container)
        button_inner_frame.pack(expand=True)

        ttk.Button(button_inner_frame, text="Cancel", command=self.on_cancel).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_inner_frame, text="Insert", command=self.on_insert).pack(
            side=tk.LEFT, padx=(5, 0))

    def on_type_change(self):
        """Handle object type change"""
        self.update_parameter_fields()

    def update_parameter_fields(self):
        """Update parameter fields based on object type"""
        # Clear existing fields
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        self.param_vars.clear()

        # Add parameters based on type
        params = self.get_parameters_for_type(self.object_type.get())

        row = 0
        for param_name, param_info in params.items():
            label = ttk.Label(self.params_frame, text=f"{param_info['label']}:")
            label.grid(row=row, column=0, sticky='w', pady=2)

            var = tk.DoubleVar(value=param_info['default']) if param_info['type'] == 'float' else \
                  tk.IntVar(value=param_info['default'])
            self.param_vars[param_name] = var

            if param_info['type'] in ['float', 'int']:
                entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
                entry.grid(row=row, column=1, sticky='w', padx=(10, 0))

                if 'min' in param_info and 'max' in param_info:
                    info = ttk.Label(self.params_frame,
                                   text=f"({param_info['min']} - {param_info['max']})",
                                   foreground='gray')
                    info.grid(row=row, column=2, sticky='w', padx=(5, 0))

            row += 1

        if not params:
            ttk.Label(self.params_frame, text="No parameters for this object type",
                     foreground='gray').grid(row=0, column=0)

    def get_parameters_for_type(self, obj_type: str) -> dict:
        """Get parameters for specific object type"""
        params = {
            'cube': {
                'size': {'label': 'Size', 'type': 'float', 'default': 2.0, 'min': 0.1, 'max': 10.0}
            },
            'sphere': {
                'radius': {'label': 'Radius', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0},
                'subdivisions': {'label': 'Subdivisions', 'type': 'int', 'default': 2, 'min': 0, 'max': 5}
            },
            'cylinder': {
                'radius': {'label': 'Radius', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0},
                'height': {'label': 'Height', 'type': 'float', 'default': 2.0, 'min': 0.1, 'max': 10.0},
                'segments': {'label': 'Segments', 'type': 'int', 'default': 16, 'min': 3, 'max': 64}
            },
            'cone': {
                'radius': {'label': 'Radius', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0},
                'height': {'label': 'Height', 'type': 'float', 'default': 2.0, 'min': 0.1, 'max': 10.0},
                'segments': {'label': 'Segments', 'type': 'int', 'default': 16, 'min': 3, 'max': 64}
            },
            'torus': {
                'major_radius': {'label': 'Major Radius', 'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 10.0},
                'minor_radius': {'label': 'Minor Radius', 'type': 'float', 'default': 0.3, 'min': 0.1, 'max': 5.0},
                'major_segments': {'label': 'Major Segments', 'type': 'int', 'default': 16, 'min': 3, 'max': 64},
                'minor_segments': {'label': 'Minor Segments', 'type': 'int', 'default': 8, 'min': 3, 'max': 32}
            },
            'tetrahedron': {},
            'octahedron': {},
            'icosahedron': {}
        }

        return params.get(obj_type, {})

    def set_uniform_scale(self):
        """Set uniform scale from X value"""
        scale = self.scale_vars['x'].get()
        self.scale_vars['y'].set(scale)
        self.scale_vars['z'].set(scale)

    def update_color_preview(self):
        """Update color preview canvas"""
        r = int(self.color_vars['r'].get() * 255)
        g = int(self.color_vars['g'].get() * 255)
        b = int(self.color_vars['b'].get() * 255)
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_preview.configure(bg=color)

    def create_mesh(self) -> Optional[HalfedgeMesh]:
        """Create mesh based on selected type and parameters"""
        obj_type = self.object_type.get()

        try:
            if obj_type == 'cube':
                size = self.param_vars.get('size', tk.DoubleVar(value=2.0)).get()
                return PrimitiveGenerator.create_cube(size)

            elif obj_type == 'sphere':
                radius = self.param_vars.get('radius', tk.DoubleVar(value=1.0)).get()
                subdivisions = self.param_vars.get('subdivisions', tk.IntVar(value=2)).get()
                return PrimitiveGenerator.create_sphere(radius, subdivisions)

            elif obj_type == 'cylinder':
                radius = self.param_vars.get('radius', tk.DoubleVar(value=1.0)).get()
                height = self.param_vars.get('height', tk.DoubleVar(value=2.0)).get()
                segments = self.param_vars.get('segments', tk.IntVar(value=16)).get()
                return PrimitiveGenerator.create_cylinder(radius, height, segments)

            elif obj_type == 'cone':
                radius = self.param_vars.get('radius', tk.DoubleVar(value=1.0)).get()
                height = self.param_vars.get('height', tk.DoubleVar(value=2.0)).get()
                segments = self.param_vars.get('segments', tk.IntVar(value=16)).get()
                return PrimitiveGenerator.create_cone(radius, height, segments)

            elif obj_type == 'torus':
                major_radius = self.param_vars.get('major_radius', tk.DoubleVar(value=1.0)).get()
                minor_radius = self.param_vars.get('minor_radius', tk.DoubleVar(value=0.3)).get()
                major_segments = self.param_vars.get('major_segments', tk.IntVar(value=16)).get()
                minor_segments = self.param_vars.get('minor_segments', tk.IntVar(value=8)).get()
                return PrimitiveGenerator.create_torus(major_radius, minor_radius,
                                                      major_segments, minor_segments)

            elif obj_type == 'tetrahedron':
                return PrimitiveGenerator.create_tetrahedron()

            elif obj_type == 'octahedron':
                return PrimitiveGenerator.create_octahedron()

            elif obj_type == 'icosahedron':
                return PrimitiveGenerator.create_icosahedron()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create mesh: {str(e)}")
            return None

        return None

    def on_insert(self):
        """Handle insert button"""
        mesh = self.create_mesh()
        if not mesh:
            return

        # Get name
        name = self.name_var.get().strip()
        if not name:
            name = f"{self.object_type.get().capitalize()}_object"

        # Create result
        self.result = {
            'mesh': mesh,
            'name': name,
            'position': np.array([self.position_vars[a].get() for a in ['x', 'y', 'z']]),
            'rotation': np.array([np.radians(self.rotation_vars[a].get()) for a in ['x', 'y', 'z']]),
            'scale': np.array([self.scale_vars[a].get() for a in ['x', 'y', 'z']]),
            'color': [self.color_vars[c].get() for c in ['r', 'g', 'b']]
        }

        self.dialog.destroy()

    def on_cancel(self):
        """Handle cancel button"""
        self.result = None
        self.dialog.destroy()

    def show(self) -> Optional[dict]:
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result


class TransformDialog:
    """Dialog untuk transform selected object"""

    def __init__(self, parent, current_transform: dict):
        self.parent = parent
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Transform Object")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Transform parameters
        self.position_vars = {
            'x': tk.DoubleVar(value=current_transform['position'][0]),
            'y': tk.DoubleVar(value=current_transform['position'][1]),
            'z': tk.DoubleVar(value=current_transform['position'][2])
        }

        self.rotation_vars = {
            'x': tk.DoubleVar(value=np.degrees(current_transform['rotation'][0])),
            'y': tk.DoubleVar(value=np.degrees(current_transform['rotation'][1])),
            'z': tk.DoubleVar(value=np.degrees(current_transform['rotation'][2]))
        }

        self.scale_vars = {
            'x': tk.DoubleVar(value=current_transform['scale'][0]),
            'y': tk.DoubleVar(value=current_transform['scale'][1]),
            'z': tk.DoubleVar(value=current_transform['scale'][2])
        }

        self.setup_ui()

        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

    def setup_ui(self):
        """Setup UI components"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Transform
        transform_frame = ttk.LabelFrame(main_frame, text="Transform", padding="10")
        transform_frame.pack(fill=tk.X, pady=(0, 10))

        # Position
        pos_label = ttk.Label(transform_frame, text="Position:")
        pos_label.grid(row=0, column=0, sticky='w', pady=5)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=0, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.position_vars[axis],
                     width=10).grid(row=0, column=2+i*2, sticky='w', padx=(0, 5))

        # Rotation
        rot_label = ttk.Label(transform_frame, text="Rotation (deg):")
        rot_label.grid(row=1, column=0, sticky='w', pady=5)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=1, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.rotation_vars[axis],
                     width=10).grid(row=1, column=2+i*2, sticky='w', padx=(0, 5))

        # Scale
        scale_label = ttk.Label(transform_frame, text="Scale:")
        scale_label.grid(row=2, column=0, sticky='w', pady=5)

        for i, axis in enumerate(['x', 'y', 'z']):
            ttk.Label(transform_frame, text=f"{axis.upper()}:").grid(
                row=2, column=1+i*2, sticky='e', padx=(10, 2))
            ttk.Entry(transform_frame, textvariable=self.scale_vars[axis],
                     width=10).grid(row=2, column=2+i*2, sticky='w', padx=(0, 5))

        # Quick actions
        actions_frame = ttk.LabelFrame(main_frame, text="Quick Actions", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(actions_frame, text="Reset Position",
                  command=self.reset_position).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Reset Rotation",
                  command=self.reset_rotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Reset Scale",
                  command=self.reset_scale).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="Uniform Scale",
                  command=self.set_uniform_scale).pack(side=tk.LEFT, padx=2)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # Center buttons
        button_inner_frame = ttk.Frame(button_frame)
        button_inner_frame.pack(expand=True)

        ttk.Button(button_inner_frame, text="Cancel", command=self.on_cancel).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_inner_frame, text="Apply", command=self.on_apply).pack(
            side=tk.LEFT, padx=(5, 0))

    def reset_position(self):
        """Reset position to origin"""
        for axis in ['x', 'y', 'z']:
            self.position_vars[axis].set(0.0)

    def reset_rotation(self):
        """Reset rotation to zero"""
        for axis in ['x', 'y', 'z']:
            self.rotation_vars[axis].set(0.0)

    def reset_scale(self):
        """Reset scale to 1"""
        for axis in ['x', 'y', 'z']:
            self.scale_vars[axis].set(1.0)

    def set_uniform_scale(self):
        """Set uniform scale from X value"""
        scale = self.scale_vars['x'].get()
        self.scale_vars['y'].set(scale)
        self.scale_vars['z'].set(scale)

    def on_apply(self):
        """Handle apply button"""
        self.result = {
            'position': np.array([self.position_vars[a].get() for a in ['x', 'y', 'z']]),
            'rotation': np.array([np.radians(self.rotation_vars[a].get()) for a in ['x', 'y', 'z']]),
            'scale': np.array([self.scale_vars[a].get() for a in ['x', 'y', 'z']])
        }
        self.dialog.destroy()

    def on_cancel(self):
        """Handle cancel button"""
        self.result = None
        self.dialog.destroy()

    def show(self) -> Optional[dict]:
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result