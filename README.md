# R-VGGT: Sparse-View 3D Medical CT Reconstruction via Radiative Gaussian Splatting

**R-VGGT** is a physics-informed deep learning pipeline designed to reconstruct full 3D Computed Tomography (CT) volumes from extremely sparse-view 2D X-ray projections (as few as 4 orthogonal views). By combining a pre-trained **Visual Geometry Grounded Transformer (VGGT)** with **Differentiable Beer-Lambert Volume Rendering** and **3D Gaussian Splatting**, this framework drastically reduces patient radiation exposure while preserving anatomical fidelity.

---

## 📌 Problem Statement

Traditional clinical CT scanners require over 1,000 X-ray projections taken across a full $360^\circ$ rotation to reconstruct a diagnostic-quality 3D volume, exposing patients to high doses of ionizing radiation.

* **The Core Challenge:** Reconstructing a complete 3D volume from only 4 orthogonal views ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) leaves more than 99% of the projection data missing. Classical reconstruction algorithms (such as Filtered Back-Projection) break down under such extreme sparsity, introducing severe streaking artifacts and depth ambiguity.

---

## 🏗️ End-to-End Pipeline Workflow

The R-VGGT pipeline bridges 2D multi-view vision with 3D volumetric X-ray physics across six sequential stages:

```text
[3D CT Volume] ──(Phase 1)──▶ [4 Synthetic X-Rays] ──(Phase 2)──▶ [1024D VGGT Tokens]
                                                                          │
[3D NIfTI Grid] ◀─(Phase 6)── [Trained 3D Gaussians] ◀─(Phase 4 & 5)── [3D Gaussian Cloud] (Phase 3)
```

### Phase 1: Physics-Based Data Preparation

* **Clinical Dataset & Standardization:** Uses low-dose clinical CT volumes from the AAPM Grand Challenge dataset (9 volumes for training, 1 held out for testing). Hounsfield Units (HU) are windowed to $[-1000, +1000]$, normalized to $[0, 1]$, and spatially centered at the origin $(0,0,0)$.
* **Virtual Cone-Beam Scanner (DRR):** Simulates Digitally Reconstructed Radiographs (DRRs) from 4 sparse orthogonal angles using cone-beam geometry.
* **Beer-Lambert Attenuation & Noise:** Applies physical X-ray transmission attenuation ($I = I_0 \cdot e^{-\sum \mu \Delta x}$) alongside Poisson noise injection to replicate realistic low-dose clinical radiographs.

### Phase 2: Cross-View AI Feature Extraction

* **Geometric Backbone:** Leverages Meta's pre-trained Visual Geometry Grounded Transformer (VGGT) to reason across multiple 2D views simultaneously.
* **Global Attention Hook Interception:** Instead of relying on standard depth map predictions, raw feature activations are intercepted directly from VGGT's final global attention block—where all 4 X-ray views have cross-attended—encoding multi-view 3D geometric correspondence into 1024-dimensional tokens.
* **Spatial Resolution Alignment:** Upsamples the $37 \times 37$ patch token grid via bilinear interpolation back to the full $256 \times 256$ radiograph resolution, equipping every pixel with a dense geometric descriptor.

### Phase 3: Geometric Lifting into 3D Space

* **Volume-Centered Ray Casting:** Casts $65,536$ rays ($256 \times 256$ pixels per view) through the 3D coordinate space from each virtual detector pose.
* **3D Gaussian Scaffold Initialization:** Seeds 32 trainable 3D Gaussian primitives at uniform depth intervals along every ray, forming a volumetric cloud of ~8.3 million primitives.
* **Feature Inheritance & Bounding Mask:** Each 3D Gaussian inherits the 1024D VGGT feature token from its parent pixel, while primitives falling outside the physical CT bounding volume are masked out to optimize compute and memory.

### Phase 4: Differentiable Beer-Lambert Volume Rendering

* **Transmission Physics Engine:** Replaces conventional surface-reflection rendering with a custom, fully differentiable volume renderer built specifically for X-ray transmission physics.
* **Front-to-Back Depth Integration:** Sorts Gaussian primitives along each ray by depth ($t$) and integrates their attenuated radiative densities using the Beer-Lambert exponential attenuation law.
* **Multi-Objective Optimization Loss:** Combines **L1 Loss** (radiographic intensity accuracy), **SSIM Loss** (structural anatomical preservation), and **Spatial Regularization** (penalizing oversized or overly opaque primitives) to guide gradient updates.

### Phase 5: Multi-Patient Training & LoRA Domain Adaptation

* **Shared Radiative MLP Head:** A neural network head processes each Gaussian's 1024D VGGT token together with its ray depth to predict volumetric density and opacity refinements. Training across all 9 patients simultaneously allows the network to learn generalizable anatomical priors.
* **Parameter-Efficient LoRA Fine-Tuning:** Applies Low-Rank Adaptation ($r = 8$) to VGGT's attention layers (`qkv`, `proj`), adapting the transformer's visual representations from natural RGB images to radiographic X-ray physics while updating less than 1% of backbone parameters.
* **Cross-Patient Generalization Evaluation:** Evaluates reconstruction performance on an unseen test patient under both **Zero-Shot** (direct inference) and **Few-Shot** (rapid per-patient adaptation) regimes.

### Phase 6: Volumetric Voxel Splatting & Clinical Export

* **Trilinear 3D Splatting:** Evaluates the final $(x, y, z)$ positions and densities of the trained 3D Gaussians and projects them onto a solid $256 \times 256 \times 256$ voxel grid via trilinear weighted splatting.
* **Volumetric Smoothing:** Applies a 3D Gaussian smoothing filter across the voxel grid to eliminate discretization gaps between adjacent primitives.
* **Clinical Format Export:** Exports the reconstructed 3D volume into standard medical **NIfTI (`.nii.gz`)** format alongside orthogonal cross-sectional comparisons across **Axial**, **Coronal**, and **Sagittal** planes.

---

## 📊 Experimental Results & Key Takeaways

Evaluated on unseen test patient **L506** using 4 sparse orthogonal views and 32 Gaussians per ray:

| Evaluation Stage | Configuration | PSNR (dB) | SSIM | Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **2D Projection (Zero-Shot)** | Frozen Backbone + Radiative Head | 11.60 | — | Baseline cross-patient generalization |
| **2D Projection (Few-Shot)** | 32 Gaussians/Ray + Depth-Aware Head | **32.29** | **0.9806** | Near-perfect novel 2D X-ray synthesis |
| **3D Volume (Phase 6)** | Trilinear Splatting to Voxel Grid | 10.72 | 0.3322 | Constrained by Z-axis depth ambiguity |

### Key Findings & Current Limitations

* **High-Fidelity 2D Radiograph Synthesis:** With 32 Gaussians per ray and depth-aware density prediction, the differentiable Beer-Lambert renderer reconstructs unseen 2D X-ray projections with remarkable precision (**32.29 dB PSNR**, **0.9806 SSIM**).
* **The Volumetric Depth Ambiguity Bottleneck:** When splatted into a dense $256^3$ voxel grid, active voxel occupancy remains sparse (~12.38%). Because supervision occurs strictly through 2D projection losses without explicit 3D volumetric constraints, primitives can cluster along ray intersections—producing characteristic "X-shaped" sparse-view artifacts in 3D cross-sections.
