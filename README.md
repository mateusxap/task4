# 4D Tensor Visualization
This is a Python-based demo application for visualizing a 4D tensor (size 20×20×20×20, uint8 data type) in an interactive, user-friendly way. It supports two visualization modes: 2D slices (showing six 2D cross-sections) and 3D projection (a scatter plot of a 3D slice). High values are displayed as bright pixels, and low values as dark ones, with customizable color schemes.

The project is built using Python with libraries like NumPy, Matplotlib, and Tkinter. You can explore an existing tensor, generate a random one, or import your own from a .npy file.

Repository: https://github.com/mateusxap/task4

## Features
- View six 2D slices of the tensor by fixing two dimensions using sliders.
- Explore a 3D scatter plot by fixing the fourth dimension.
- Customize visualizations with different Matplotlib colormaps (e.g., "plasma", "viridis").
- Import a tensor from a .npy file or generate a random tensor for testing.
- Export current 2D or 3D views as PNG images.
- Reset the view to default settings with a single click.

## Prerequisites
Python 3.6+: Ensure you have Python installed
Git: To clone the repository

## Installation

Follow these steps to set up and run the program:

1. Clone the Repository
Open a terminal and run:
```bash
git clone https://github.com/mateusxap/task4.git
cd task4
```

2. Set Up a Virtual Environment (optional but recommended)
Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
Install the required libraries listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

```
python3 visualizer.py
```

