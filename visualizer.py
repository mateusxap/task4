import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TensorVisualizer:
    def __init__(self, tensor=None):
        self.tensor = tensor
        # If there is no tensor, we use the standard form for the interface (20,20,20,20)
        self.shape = tensor.shape if tensor is not None else (20, 20, 20, 20)

        # If the tensor is given, we check its dimension
        if self.tensor is not None and len(self.shape) != 4:
            raise ValueError("Входной тензор должен быть 4D")

        # Creating the main Tkinter window
        self.root = tk.Tk()
        self.root.title("4D Tensor Visualization")
        self.root.geometry("1200x800")

        # Interface setup
        self.setup_ui()
        # Update visualizations (if there is no tensor, empty graphs will be shown)
        self.update_plots()

    def setup_ui(self):
        # Frames for control and visualization
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Creating tabs for different visualization modes
        self.tab_control = ttk.Notebook(viz_frame)

        self.tab_2d = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_2d, text='2D Slices')

        self.tab_3d = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_3d, text='3D Projection')

        self.tab_control.pack(expand=True, fill=tk.BOTH)

        # Setting up sliders for selecting measurement indices
        self.setup_dimension_controls(control_frame)

        # Setting up charts
        self.setup_2d_visualization()
        self.setup_3d_visualization()

        # Binding the tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.active_tab = 0

    def setup_dimension_controls(self, parent):
        dims_frame = ttk.LabelFrame(parent, text="Dimension Controls", padding=10)
        dims_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dim_vars = []
        self.dim_scales = []

        for i in range(4):
            frame = ttk.Frame(dims_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"Dimension {i + 1}:")
            label.pack(side=tk.LEFT, padx=5)

            var = tk.IntVar(value=0)
            self.dim_vars.append(var)

            scale = ttk.Scale(
                frame,
                from_=0,
                to=self.shape[i] - 1,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda value, idx=i: self.on_slider_change(value, idx)
            )
            scale.bind("<ButtonRelease-1>", lambda e, idx=i: self.on_slider_release(idx))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            value_label = ttk.Label(frame, textvariable=var, width=3)
            value_label.pack(side=tk.LEFT, padx=5)
            self.dim_scales.append(scale)

        # Additional controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        cmap_frame = ttk.Frame(controls_frame)
        cmap_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(cmap_frame, text="Colormap:").pack(side=tk.LEFT)

        self.cmap_var = tk.StringVar(value="plasma")
        cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self.cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "rainbow"]
        )
        cmap_combo.pack(side=tk.LEFT, padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        # Control Button Panel
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(btn_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Current View", command=self.export_current_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Generate Random Tensor", command=self.generate_random_tensor).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Import Tensor", command=self.import_tensor).pack(side=tk.LEFT, padx=5)

    def on_slider_change(self, value, dim_idx):
        value = round(float(value))
        self.dim_vars[dim_idx].set(value)

    def on_slider_release(self, dim_idx):
        value = round(self.dim_scales[dim_idx].get())
        self.dim_vars[dim_idx].set(value)

        if self.active_tab == 0:
            self.update_2d_plots()
        elif self.active_tab == 1:
            self.update_3d_plot()

    def setup_2d_visualization(self):
        self.fig_2d = plt.Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        gs = gridspec.GridSpec(2, 3, figure=self.fig_2d, hspace=0.3, wspace=0.3)
        self.axes_2d = []
        self.img_plots_2d = []

        for i in range(6):
            row, col = divmod(i, 3)
            ax = self.fig_2d.add_subplot(gs[row, col])
            ax.set_title(f"Slice {i + 1}")
            self.axes_2d.append(ax)
            img = ax.imshow(np.zeros((2, 2)), cmap=self.cmap_var.get())
            self.img_plots_2d.append(img)
            self.fig_2d.colorbar(img, ax=ax, shrink=0.7)

        self.update_2d_plots()

    def setup_3d_visualization(self):
        self.fig_3d = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Projection", pad=20)
        self.ax_3d.set_xlabel('Dimension 1')
        self.ax_3d.set_ylabel('Dimension 2')
        self.ax_3d.set_zlabel('Dimension 3')

        self.ax_3d.set_xlim(0, self.shape[0] - 1)
        self.ax_3d.set_ylim(0, self.shape[1] - 1)
        self.ax_3d.set_zlim(0, self.shape[2] - 1)

        self.scatter_plot = self.ax_3d.scatter([], [], [], c=[], cmap=self.cmap_var.get())
        self.colorbar = self.fig_3d.colorbar(self.scatter_plot, ax=self.ax_3d, shrink=0.7, label='Value')

        self.update_3d_plot()

    def on_tab_changed(self, event):
        self.active_tab = self.tab_control.index(self.tab_control.select())
        if self.active_tab == 1:
            for i in range(3):
                self.dim_scales[i].state(['disabled'])
            self.update_3d_plot()
        else:
            for i in range(4):
                self.dim_scales[i].state(['!disabled'])
            self.update_2d_plots()

    def update_plots(self):
        if self.active_tab == 0:
            self.update_2d_plots()
        else:
            self.update_3d_plot()

    def update_2d_plots(self):
        # Checking for the presence of a tensor
        if self.tensor is None:
            for i, ax in enumerate(self.axes_2d):
                ax.clear()
                ax.text(0.5, 0.5, "No tensor imported", horizontalalignment="center", verticalalignment="center")
                ax.set_xticks([])
                ax.set_yticks([])
                # Zero out the corresponding image object
                self.img_plots_2d[i] = None
            self.canvas_2d.draw_idle()
            return

        self.fig_2d.clear()
        gs = gridspec.GridSpec(2, 3, figure=self.fig_2d, hspace=0.3, wspace=0.3)
        self.axes_2d = []
        self.img_plots_2d = []

        d1, d2, d3, d4 = [var.get() for var in self.dim_vars]

        # Form all the sections
        slice_data = [
            self.tensor[d1, d2, :, :],
            self.tensor[d1, :, d3, :],
            self.tensor[d1, :, :, d4],
            self.tensor[:, d2, d3, :],
            self.tensor[:, d2, :, d4],
            self.tensor[:, :, d3, d4]
        ]

        # Forming headings for graphs
        titles = [
            f"Dim1={d1}, Dim2={d2}",
            f"Dim1={d1}, Dim3={d3}",
            f"Dim1={d1}, Dim4={d4}",
            f"Dim2={d2}, Dim3={d3}",
            f"Dim2={d2}, Dim4={d4}",
            f"Dim3={d3}, Dim4={d4}"
        ]

        # Forming boundaries for graphs
        extents = [
            [-0.5, self.shape[2] - 0.5, self.shape[3] - 0.5, -0.5],  # Для среза 1
            [-0.5, self.shape[1] - 0.5, self.shape[3] - 0.5, -0.5],  # Для среза 2
            [-0.5, self.shape[1] - 0.5, self.shape[2] - 0.5, -0.5],  # Для среза 3
            [-0.5, self.shape[0] - 0.5, self.shape[3] - 0.5, -0.5],  # Для среза 4
            [-0.5, self.shape[0] - 0.5, self.shape[2] - 0.5, -0.5],  # Для среза 5
            [-0.5, self.shape[0] - 0.5, self.shape[1] - 0.5, -0.5]  # Для среза 6
        ]

        for i in range(6):
            row, col = divmod(i, 3)
            ax = self.fig_2d.add_subplot(gs[row, col])
            ax.set_title(titles[i], y=1.0)
            self.axes_2d.append(ax)

            img = ax.imshow(slice_data[i], cmap=self.cmap_var.get(),
                            extent=extents[i], vmin=0, vmax=255)
            self.img_plots_2d.append(img)

            self.fig_2d.colorbar(img, ax=ax, shrink=0.7)

        self.canvas_2d.draw_idle()
    def update_3d_plot(self):
        if self.tensor is None:
            self.ax_3d.clear()
            self.ax_3d.text(0.5, 0.5, 0.5, "No tensor imported", horizontalalignment="center", verticalalignment="center")
            self.canvas_3d.draw()
            return

        d4 = self.dim_vars[3].get()
        self.ax_3d.clear()

        # Extract 3D slice (fix the fourth dimension)
        volume = self.tensor[:, :, :, d4]
        vmin, vmax = volume.min(), volume.max()
        norm_volume = (volume - vmin) / (vmax - vmin + 1e-10)

        threshold = 0.5
        x, y, z = np.where(norm_volume > threshold)

        max_points = 2000
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
            values = norm_volume[x, y, z]
        else:
            values = norm_volume[x, y, z]

        if len(x) > 0:
            size = 50
            self.scatter_plot = self.ax_3d.scatter3D(x, y, z, c=values, cmap=self.cmap_var.get(),
                                                     s=size, alpha=0.7)

        self.ax_3d.set_xlim(0, volume.shape[0] - 1)
        self.ax_3d.set_ylim(0, volume.shape[1] - 1)
        self.ax_3d.set_zlim(0, volume.shape[2] - 1)
        self.ax_3d.set_xlabel('Dimension 1')
        self.ax_3d.set_ylabel('Dimension 2')
        self.ax_3d.set_zlabel('Dimension 3')
        self.ax_3d.set_title(f'3D Projection (Dimension 4 = {d4})')

        if len(x) > 0:
            self.colorbar.update_normal(self.scatter_plot)

        self.canvas_3d.draw()

    def reset_view(self):
        for var in self.dim_vars:
            var.set(0)
        self.update_plots()

    def export_current_view(self):
        save_window = tk.Toplevel(self.root)
        save_window.title("Export View")
        save_window.geometry("300x150")

        ttk.Label(save_window, text="Export Options").pack(pady=10)

        export_type = tk.StringVar(value="2D")
        ttk.Radiobutton(save_window, text="Export 2D Views", variable=export_type, value="2D").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(save_window, text="Export 3D View", variable=export_type, value="3D").pack(anchor=tk.W, padx=20)

        def do_export():
            try:
                if export_type.get() == "2D":
                    self.fig_2d.savefig("tensor_2d_slices.png", dpi=300)
                else:
                    self.fig_3d.savefig("tensor_3d_projection.png", dpi=300)
                ttk.Label(save_window, text="Exported successfully!").pack(pady=10)
            except Exception as e:
                ttk.Label(save_window, text=f"Error: {str(e)}").pack(pady=10)

        ttk.Button(save_window, text="Export", command=do_export).pack(pady=10)

    def import_tensor(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npy")],
            title="Import Tensor"
        )
        if file_path:
            try:
                tensor = np.load(file_path)
                if tensor.shape != (20, 20, 20, 20) or tensor.dtype != np.uint8:
                    messagebox.showerror("Error", "Tensor must have shape (20,20,20,20) and dtype uint8")
                    return
                self.tensor = tensor
                self.shape = tensor.shape
                for i in range(4):
                    self.dim_scales[i].configure(from_=0, to=self.shape[i] - 1)
                    self.dim_vars[i].set(0)
                self.update_plots()
                messagebox.showinfo("Success", "Tensor imported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import tensor:\n{str(e)}")

    def generate_random_tensor(self):
        self.tensor = np.random.randint(0, 256, size=(20, 20, 20, 20), dtype=np.uint8)
        self.shape = self.tensor.shape
        for i in range(4):
            self.dim_scales[i].configure(from_=0, to=self.shape[i] - 1)
            self.dim_vars[i].set(0)
        self.update_plots()

    def run(self):
        self.root.mainloop()

def generate_sample_tensor(shape=(20, 20, 20, 20)):
    print("Generating sample tensor...")
    tensor = np.zeros(shape, dtype=np.uint8)
    x = np.linspace(-3, 3, shape[0])
    y = np.linspace(-3, 3, shape[1])
    z = np.linspace(-3, 3, shape[2])
    w = np.linspace(-3, 3, shape[3])
    for i, xi in enumerate(x):
        print(f"Generating slice {i + 1}/{shape[0]}...", end="\r")
        for j, yi in enumerate(y):
            for k, zi in enumerate(z):
                distance = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2 + w ** 2)
                pattern = np.exp(-distance)
                wave = np.sin(xi * yi) * np.cos(zi * w)
                combined = pattern + 0.5 * wave
                tensor[i, j, k, :] = np.clip((combined * 255), 0, 255).astype(np.uint8)
    print("\nSample tensor generated successfully!")
    return tensor

def main():
    parser = argparse.ArgumentParser(description='4D Tensor Visualization')
    parser.add_argument('--file', type=str, help='Path to numpy file containing 4D tensor (optional)')
    args = parser.parse_args()

    # Regardless of the presence of the file, we start without a tensor
    visualizer = TensorVisualizer(tensor=None)
    visualizer.run()

if __name__ == "__main__":
    main()