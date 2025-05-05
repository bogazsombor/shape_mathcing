# Shape Matching Morphing Library

## Short Description
This repository implements the shape-matching morphing algorithm described in [Lévy et al. 2016](https://www.sciencedirect.com/science/article/pii/S1631073X16300802). It provides:

1. **A native C++/C implementation** (in `Morphing/`) based on the original ISCD Toolbox.  
2. **A Python wrapper** (in `morphing_py/`) leveraging the FEniCS/DOLFINx library for finite element operations.

The code transforms a template mesh into a target mesh by computing an elastic deformation that minimizes an energy functional, enabling smooth shape interpolation and registration.

---

## Repository Structure

```
.
├── Morphing/            # Original C implementation (ISCD toolbox port)
│   ├── src/             # C source files
│   ├── ...
│
├── morphing_py/         # Python implementation using DOLFINx
│   ├── shape_matching.py# Main Python script
│   ├── ...
│   └── 
│
├── target.mesh          # Example target mesh
├── template.mesh        # Example template mesh
```

## Setup

1. **Add project root to Python path**:

   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)/"
   ```

2. **Create env if needed**:

   ```bash
   conda env create -f environment.yml
   ```

3. **Run the Python version**:

   ```bash
   python3 morphing_py/shape_matching.py template.mesh target.mesh --c_mode
   ```

4. **Build and install the original C code (optional)**:

   ```bash
   cd Morphing/build
   cmake ..
   make
   make install
   cd ../demo
   morphing -dref 1 2 -elref 1 1 -bref 1 1 target.mesh template.mesh
   cd ../../
   ```

## Usage

- C implementation parameters (`-dref`, `-elref`, `-bref`) control deformation scales and energy weights.  
- Python version supports the same options via the `--c_mode` flag for comparable results.

## Dependencies

- **C build**: CMake, a C compiler (GCC/Clang).  
- **Python**: DOLFINx (FEniCSx), NumPy, MeshIO.

## Review Summary

This project faithfully reproduces the elastic shape-matching algorithm from Lévy et al. in both C and Python. The C code retains performance of the original toolbox, while the Python version enhances usability and leverages modern finite-element tooling. The dual implementation facilitates benchmarking, extension, and integration into Python-based pipelines.

---

## References

- Maya de Buhan, Charles Dapogny, Pascal Frey ,Chiara Nardoni : An optimization method for elastic shape matching (https://www.sciencedirect.com/science/article/pii/S1631073X16300802)
- ISCD Toolbox original morphing code: https://github.com/ISCDtoolbox/Morphing
