import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


class TensorVisualizer:
    """
    A class for visualizing 4D tensors using interactive slicing and projections.
    Optimized for better performance with large tensors.
    """

    def __init__(self, tensor):
        """Initialize with a 4D tensor"""
        self.tensor = tensor
        self.shape = tensor.shape

        if len(self.shape) != 4:
            raise ValueError("Input tensor must be 4D")

        # Create the main Tkinter window
        self.root = tk.Tk()
        self.root.title("4D Tensor Visualization")
        self.root.geometry("1200x800")

        # Set up the layout
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        # Create main frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create tabs for different visualization modes
        self.tab_control = ttk.Notebook(viz_frame)

        # Create tab for 2D slices
        self.tab_2d = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_2d, text='2D Slices')

        # Create tab for 3D projections
        self.tab_3d = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_3d, text='3D Projection')

        self.tab_control.pack(expand=True, fill=tk.BOTH)

        # Set up dimension selectors
        self.setup_dimension_controls(control_frame)

        # Set up the plots
        self.setup_2d_visualization()
        self.setup_3d_visualization()

        # Removed the "Update 3D View" button as requested

        # Bind tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.active_tab = 0  # Default to 2D tab active

    def setup_dimension_controls(self, parent):
        """Set up sliders for selecting dimensions"""
        # Create frame for dimension controls
        dims_frame = ttk.LabelFrame(parent, text="Dimension Controls", padding=10)
        dims_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create sliders for each dimension
        self.dim_vars = []
        self.dim_scales = []

        for i in range(4):
            frame = ttk.Frame(dims_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"Dimension {i + 1}:")
            label.pack(side=tk.LEFT, padx=5)

            var = tk.IntVar(value=0)
            self.dim_vars.append(var)

            # Create discrete slider with steps
            scale = ttk.Scale(
                frame,
                from_=0,
                to=self.shape[i] - 1,
                orient=tk.HORIZONTAL,
                variable=var
            )
            # Only update when released, not during drag
            scale.bind("<ButtonRelease-1>", lambda e, idx=i: self.on_slider_release(idx))

            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.dim_scales.append(scale)

            value_label = ttk.Label(frame, textvariable=var, width=3)
            value_label.pack(side=tk.LEFT, padx=5)

        # Additional controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Colormap selection
        cmap_frame = ttk.Frame(controls_frame)
        cmap_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(cmap_frame, text="Colormap:").pack(side=tk.LEFT)

        self.cmap_var = tk.StringVar(value="plasma")  # Default to plasma as requested
        cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self.cmap_var,
            values=["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "rainbow"]
        )
        cmap_combo.pack(side=tk.LEFT, padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        # Buttons for navigation
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(btn_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Current View", command=self.export_current_view).pack(side=tk.LEFT, padx=5)

    def on_slider_release(self, dim_idx):
        """Handle slider release events"""
        # Round to nearest integer and update the variable
        value = round(self.dim_scales[dim_idx].get())
        self.dim_vars[dim_idx].set(value)

        # Update plots based on the active tab
        if self.active_tab == 0:  # 2D tab
            self.update_2d_plots()
        elif self.active_tab == 1:  # 3D tab
            # Update the 3D view automatically when a slider is released
            self.update_3d_plot()

    def setup_2d_visualization(self):
        """Set up the 2D visualization tab"""
        # Create a figure for the 2D slices
        self.fig_2d = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Set up the grid for 2D slices with more space for titles
        gs = gridspec.GridSpec(2, 3, figure=self.fig_2d, hspace=0.3, wspace=0.3)

        # Create 6 subplots for various 2D slices
        self.axes_2d = []
        self.img_plots_2d = []

        for i in range(6):
            row, col = divmod(i, 3)
            ax = self.fig_2d.add_subplot(gs[row, col])
            ax.set_title(f"Slice {i + 1}")
            self.axes_2d.append(ax)
            # Initialize with empty image
            img = ax.imshow(np.zeros((2, 2)), cmap=self.cmap_var.get())
            self.img_plots_2d.append(img)
            self.fig_2d.colorbar(img, ax=ax, shrink=0.7)

        self.fig_2d = plt.Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        self.update_2d_plots()

    def setup_3d_visualization(self):
        """Set up the 3D visualization tab"""
        self.fig_3d = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create a 3D projection plot
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_title("3D Projection", pad=20)  # Increased padding

        # Set labels
        self.ax_3d.set_xlabel('Dimension 1')
        self.ax_3d.set_ylabel('Dimension 2')
        self.ax_3d.set_zlabel('Dimension 3')

        # Set initial view limits
        self.ax_3d.set_xlim(0, self.shape[0] - 1)
        self.ax_3d.set_ylim(0, self.shape[1] - 1)
        self.ax_3d.set_zlim(0, self.shape[2] - 1)

        # Dummy scatter plot for colorbar
        self.scatter_plot = self.ax_3d.scatter([], [], [], c=[], cmap=self.cmap_var.get())
        self.colorbar = self.fig_3d.colorbar(self.scatter_plot, ax=self.ax_3d, shrink=0.7, label='Value')

        # Update 3D plot
        self.update_3d_plot()

    def on_tab_changed(self, event):
        """Handle tab change events"""
        self.active_tab = self.tab_control.index(self.tab_control.select())

        # If switching to 3D tab, update the 3D view and disable dimensions 1-3 sliders
        if self.active_tab == 1:  # 3D tab
            # Disable dimensions 1-3 sliders since they don't affect 3D view
            for i in range(3):
                self.dim_scales[i].state(['disabled'])
            self.update_3d_plot()
        else:  # 2D tab
            # Re-enable all sliders
            for i in range(4):
                self.dim_scales[i].state(['!disabled'])
            self.update_2d_plots()

    def update_plots(self):
        """Update all visualizations based on active tab"""
        if self.active_tab == 0:
            self.update_2d_plots()
        else:
            self.update_3d_plot()

    def update_2d_plots(self):
        """Update 2D slice visualizations"""
        # Get current dimension values
        d1, d2, d3, d4 = [var.get() for var in self.dim_vars]

        # Create 6 different 2D slices using different dimension combinations
        # Reuse image objects instead of recreating them

        # Slice 1: Fix d1, d2, vary d3, d4
        slice1 = self.tensor[d1, d2, :, :]
        self.img_plots_2d[0].set_data(slice1)
        self.img_plots_2d[0].set_cmap(self.cmap_var.get())
        self.img_plots_2d[0].set_clim(0, 255)  # Assuming uint8 data
        self.axes_2d[0].set_title(f"Dim1={d1}, Dim2={d2}", y=1.0)

        # Slice 2: Fix d1, d3, vary d2, d4
        slice2 = self.tensor[d1, :, d3, :]
        self.img_plots_2d[1].set_data(slice2)
        self.img_plots_2d[1].set_cmap(self.cmap_var.get())
        self.img_plots_2d[1].set_clim(0, 255)
        self.axes_2d[1].set_title(f"Dim1={d1}, Dim3={d3}", y=1.0)

        # Slice 3: Fix d1, d4, vary d2, d3
        slice3 = self.tensor[d1, :, :, d4]
        self.img_plots_2d[2].set_data(slice3)
        self.img_plots_2d[2].set_cmap(self.cmap_var.get())
        self.img_plots_2d[2].set_clim(0, 255)
        self.axes_2d[2].set_title(f"Dim1={d1}, Dim4={d4}", y=1.0)

        # Slice 4: Fix d2, d3, vary d1, d4
        slice4 = self.tensor[:, d2, d3, :]
        self.img_plots_2d[3].set_data(slice4)
        self.img_plots_2d[3].set_cmap(self.cmap_var.get())
        self.img_plots_2d[3].set_clim(0, 255)
        self.axes_2d[3].set_title(f"Dim2={d2}, Dim3={d3}", y=1.0)

        # Slice 5: Fix d2, d4, vary d1, d3
        slice5 = self.tensor[:, d2, :, d4]
        self.img_plots_2d[4].set_data(slice5)
        self.img_plots_2d[4].set_cmap(self.cmap_var.get())
        self.img_plots_2d[4].set_clim(0, 255)
        self.axes_2d[4].set_title(f"Dim2={d2}, Dim4={d4}", y=1.0)

        # Slice 6: Fix d3, d4, vary d1, d2
        slice6 = self.tensor[:, :, d3, d4]
        self.img_plots_2d[5].set_data(slice6)
        self.img_plots_2d[5].set_cmap(self.cmap_var.get())
        self.img_plots_2d[5].set_clim(0, 255)
        self.axes_2d[5].set_title(f"Dim3={d3}, Dim4={d4}", y=1.0)

        # Update axes limits for all plots
        for i, img in enumerate(self.img_plots_2d):
            extent = None
            if i == 0:  # dim3, dim4
                extent = [-0.5, self.shape[2] - 0.5, self.shape[3] - 0.5, -0.5]
            elif i == 1:  # dim2, dim4
                extent = [-0.5, self.shape[1] - 0.5, self.shape[3] - 0.5, -0.5]
            elif i == 2:  # dim2, dim3
                extent = [-0.5, self.shape[1] - 0.5, self.shape[2] - 0.5, -0.5]
            elif i == 3:  # dim1, dim4
                extent = [-0.5, self.shape[0] - 0.5, self.shape[3] - 0.5, -0.5]
            elif i == 4:  # dim1, dim3
                extent = [-0.5, self.shape[0] - 0.5, self.shape[2] - 0.5, -0.5]
            elif i == 5:  # dim1, dim2
                extent = [-0.5, self.shape[0] - 0.5, self.shape[1] - 0.5, -0.5]

            if extent:
                img.set_extent(extent)

        # Just redraw the canvas, don't recalculate layout
        self.canvas_2d.draw_idle()

    def update_3d_plot(self):
        """Update 3D projection visualization using scatter plot with improved performance"""
        # Get fourth dimension value
        d4 = self.dim_vars[3].get()

        # Clear the 3D plot but keep axes
        self.ax_3d.clear()

        # Extract 3D slice (fix the fourth dimension)
        volume = self.tensor[:, :, :, d4]

        # Normalize values to 0-1 for better visualization
        vmin, vmax = volume.min(), volume.max()
        norm_volume = (volume - vmin) / (vmax - vmin + 1e-10)

        # Use a threshold to reduce visual clutter and improve performance
        threshold = 0.5

        # Reduce the number of points for better performance
        # Only sample a portion of the points above threshold
        x, y, z = np.where(norm_volume > threshold)

        # Limit the number of points for better performance
        max_points = 2000  # Adjust this value based on your hardware capabilities
        if len(x) > max_points:
            # Randomly sample points
            indices = np.random.choice(len(x), max_points, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
            values = norm_volume[x, y, z]
        else:
            values = norm_volume[x, y, z]

        # Create a colormap based on values
        cmap = plt.get_cmap(self.cmap_var.get())

        # Plot using scatter3D
        if len(x) > 0:
            # Use fixed size for better performance
            size = 50  # Fixed size instead of variable
            self.scatter_plot = self.ax_3d.scatter3D(x, y, z, c=values, cmap=self.cmap_var.get(),
                                                     s=size, alpha=0.7)

        # Set view limits
        self.ax_3d.set_xlim(0, volume.shape[0] - 1)
        self.ax_3d.set_ylim(0, volume.shape[1] - 1)
        self.ax_3d.set_zlim(0, volume.shape[2] - 1)

        # Set labels and title
        self.ax_3d.set_xlabel('Dimension 1')
        self.ax_3d.set_ylabel('Dimension 2')
        self.ax_3d.set_zlabel('Dimension 3')
        self.ax_3d.set_title(f'3D Projection (Dimension 4 = {d4})')

        # Update colorbar if points exist
        if len(x) > 0:
            self.colorbar.update_normal(self.scatter_plot)

        # Draw the canvas
        self.canvas_3d.draw()

    def reset_view(self):
        """Reset all dimensions to initial values"""
        for i, var in enumerate(self.dim_vars):
            var.set(0)

        # Update based on active tab
        self.update_plots()

    def export_current_view(self):
        """Export current view as an image"""
        # Create a simple dialog to save the current view
        save_window = tk.Toplevel(self.root)
        save_window.title("Export View")
        save_window.geometry("300x150")

        ttk.Label(save_window, text="Export Options").pack(pady=10)

        export_type = tk.StringVar(value="2D")
        ttk.Radiobutton(save_window, text="Export 2D Views", variable=export_type, value="2D").pack(anchor=tk.W,
                                                                                                    padx=20)
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

    def run(self):
        """Run the application"""
        self.root.mainloop()


def generate_sample_tensor(shape=(20, 20, 20, 20)):
    """Generate a sample 4D tensor with interesting patterns (optimized)"""
    print("Generating sample tensor...")

    # Create a pattern with interesting features but avoid full meshgrid operation
    tensor = np.zeros(shape, dtype=np.uint8)

    # Create coordinate arrays
    x = np.linspace(-3, 3, shape[0])
    y = np.linspace(-3, 3, shape[1])
    z = np.linspace(-3, 3, shape[2])
    w = np.linspace(-3, 3, shape[3])

    # Fill tensor using nested loops - more memory efficient than meshgrid
    # Use vectorized operations where possible within the loops
    for i, xi in enumerate(x):
        print(f"Generating slice {i + 1}/{shape[0]}...", end="\r")
        for j, yi in enumerate(y):
            # Vectorized operations for the inner dimensions
            for k, zi in enumerate(z):
                # Gaussian pattern based on distance from origin
                distance = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2 + w ** 2)
                pattern = np.exp(-distance)

                # Add some sine wave patterns
                wave = np.sin(xi * yi) * np.cos(zi * w)

                # Combine patterns
                combined = pattern + 0.5 * wave

                # Scale to uint8 range (0-255)
                tensor[i, j, k, :] = np.clip((combined * 255), 0, 255).astype(np.uint8)

    print("\nSample tensor generated successfully!")
    return tensor


def main():
    """Main function to run the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='4D Tensor Visualization')
    parser.add_argument('--file', type=str, help='Path to numpy file containing 4D tensor (optional)')
    args = parser.parse_args()

    # Load tensor or generate sample
    if args.file:
        try:
            tensor = np.load(args.file)
            if tensor.shape != (20, 20, 20, 20) or tensor.dtype != np.uint8:
                print(
                    f"Warning: Expected tensor of shape (20, 20, 20, 20) with dtype uint8, got {tensor.shape} with {tensor.dtype}")
                print("Generating sample tensor instead...")
                tensor = generate_sample_tensor()
        except Exception as e:
            print(f"Error loading tensor: {str(e)}")
            print("Generating sample tensor instead...")
            tensor = generate_sample_tensor()
    else:
        print("No input file provided. Generating sample tensor...")
        tensor = generate_sample_tensor()

    # Create and run visualizer
    visualizer = TensorVisualizer(tensor)
    visualizer.run()


if __name__ == "__main__":
    main()