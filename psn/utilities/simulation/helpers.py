"""Internal helpers for simulate_data.

Low-rank orthonormal basis construction, image-based true_signal coercion,
and the heterogeneous-population diagnostic figure. Imported by simulate_data;
not part of the public simulation API.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def _random_orthonormal_columns(rng: np.random.RandomState, nvox: int, r: int) -> np.ndarray:
    """Return an (nvox, r) matrix with orthonormal columns."""
    if r <= 0:
        raise ValueError("r must be positive")
    A = rng.randn(nvox, r)
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q


def _align_noise_basis_lowrank(
    U_signal: np.ndarray,
    U_noise_init: np.ndarray,
    alpha: float,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Low-rank alignment of top-k noise PCs to signal PCs.

    Produces an orthonormal (nvox, rN) matrix whose first k columns satisfy
    dot(U_noise[:,i], U_signal[:,i]) == alpha (approximately, but very close).

    This is a scalable alternative to _adjust_alignment_gradient_descent, which
    requires full (nvox, nvox) matrices.
    """
    if k <= 0:
        return U_noise_init

    if not (0.0 <= alpha <= 1.0):
        warnings.warn("align_alpha must be in [0,1]; will be clamped.")
        alpha = float(max(0.0, min(1.0, alpha)))

    nvox = U_signal.shape[0]
    rS = U_signal.shape[1]
    rN = U_noise_init.shape[1]
    k_eff = int(min(k, rS, rN, nvox // 2))
    if k_eff <= 0:
        return U_noise_init

    Usk = U_signal[:, :k_eff]

    # Construct V: k_eff orthonormal vectors orthogonal to span(Usk)
    A = rng.randn(nvox, k_eff)
    A = A - Usk @ (Usk.T @ A)
    V, _ = np.linalg.qr(A, mode="reduced")

    # Aligned block; columns remain orthonormal because Usk ⟂ V
    aligned = alpha * Usk + np.sqrt(max(0.0, 1.0 - alpha**2)) * V

    # Fill remaining noise directions from the initial noise basis, projected to the complement
    B = np.concatenate([Usk, V], axis=1)  # (nvox, 2k)
    rest = U_noise_init[:, k_eff:]
    if rest.size == 0:
        return aligned

    rest = rest - B @ (B.T @ rest)
    rest, _ = np.linalg.qr(rest, mode="reduced")

    U_noise = np.concatenate([aligned, rest], axis=1)
    return U_noise


_IMAGE_EXTS = ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp')


def _resolve_image_path(path):
    """Resolve a true_signal image string to an actual file on disk.

    If the string is already a real path, use it as-is. Otherwise treat it
    as a name and look it up in this package's bundled images/ directory, so
    e.g. true_signal='pliny' (or 'pliny.jpg') resolves to images/pliny.jpg.
    """
    import glob
    p = os.fspath(path)
    if os.path.exists(p):
        return p
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    name = os.path.basename(p)
    # 1) exact stem with a known image extension: images/<name>.<ext>
    # 2) substring match: images/*<name>*.<ext>
    for pattern in (name + '.*', '*' + name + '*'):
        for cand in sorted(glob.glob(os.path.join(images_dir, pattern))):
            if cand.rsplit('.', 1)[-1].lower() in _IMAGE_EXTS:
                return cand
    raise FileNotFoundError(
        f"true_signal {path!r} is not an existing path, and no image matching "
        f"'*{name}*' was found in {images_dir}")


def _load_image_as_array(path):
    """Load an image file into a float array normalized to [0, 1].

    Returns an (H, W) array for grayscale images or (H, W, C) for color.
    Tries Pillow first (broadest format support), then falls back to
    matplotlib's imread so the function still works in environments
    without Pillow.
    """
    path = _resolve_image_path(path)
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im)
    except ImportError:
        import matplotlib.image as mpimg
        arr = mpimg.imread(path)
    arr = np.asarray(arr, dtype=float)
    # Normalize integer pixel ranges (e.g. uint8 0-255) to [0, 1]; leave
    # already-normalized float images untouched.
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _coerce_true_signal(true_signal):
    """Normalize a user-supplied true_signal into a 2D (ncond, nvox) array.

    Accepts:
      - a filepath / Path to an image file (loaded via _load_image_as_array)
      - an ndarray that is already 2D (ncond, nvox), returned as-is
      - an image-shaped ndarray (H, W) or (H, W, C); spatial dimensions are
        flattened to 2D

    This is what lets natural images stand in as the ground-truth signal so
    denoising can be eyeballed on a recognizable picture.

    For image-derived inputs the SMALLER flattened dimension is placed on the
    units axis (nvox) and the larger on conditions, purely so the O(nvox^3)
    covariance work stays cheap at runtime. Explicit 2D (ncond, nvox) arrays
    are left in the caller's orientation.
    """
    from_image = False
    if isinstance(true_signal, (str, bytes, os.PathLike)):
        true_signal = _load_image_as_array(true_signal)
        from_image = True
    arr = np.asarray(true_signal, dtype=float)
    if arr.ndim == 3:
        # (H, W, C) -> (H, W*C); flatten width × channels into one axis.
        h, w, c = arr.shape
        arr = arr.reshape(h, w * c)
        from_image = True
    elif arr.ndim != 2:
        raise ValueError(
            f"true_signal must be a filepath, a 2D (ncond, nvox) array, or an "
            f"image-shaped 2D/3D array; got an array with {arr.ndim} dims "
            f"(shape {arr.shape}).")
    # Smaller dim -> units (nvox), larger -> conditions. Image-derived only.
    if from_image and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    return arr


def _visualize_heterogeneous_populations(train_data, true_signal, ground_truth, 
                                         signal_cov, noise_cov, ntrial):
    """Visualize heterogeneous population data structure."""
    
    population_labels = ground_truth['population_labels']
    n_populations = ground_truth['n_populations']
    units_per_pop = ground_truth['units_per_pop']
    nvox = len(population_labels)
    ncond = true_signal.shape[0]
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Population colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_populations))
    
    # Plot 1: Ground truth signal with population boundaries
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_signal.T, aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Ground Truth Signal\n(with population structure)')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax1.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=3, linestyle='--')
    
    # Plot 2: Signal covariance (should show block structure)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(signal_cov, aspect='equal', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Signal Covariance\n(block structure)')
    ax2.set_xlabel('Unit')
    ax2.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax2.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2)
        ax2.axvline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2)
    
    # Plot 3: Noise covariance
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(noise_cov, aspect='equal', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Noise Covariance\n(global structure)')
    ax3.set_xlabel('Unit')
    ax3.set_ylabel('Unit')
    
    # Plot 4: Population labels
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.barh(np.arange(nvox), np.ones(nvox), color=[colors[pop] for pop in population_labels])
    ax4.set_yticks(np.arange(0, nvox, max(1, nvox // 10)))
    ax4.set_xlabel('Population')
    ax4.set_ylabel('Unit')
    ax4.set_title(f'Population Labels\n({n_populations} populations)')
    ax4.set_xlim([0, 1.5])
    
    # Plot 5-7: Per-population signal patterns
    for pop_idx in range(min(3, n_populations)):
        ax = fig.add_subplot(gs[1, pop_idx])
        pop_start = pop_idx * units_per_pop
        pop_end = pop_start + units_per_pop
        pop_signal = true_signal[:, pop_start:pop_end].T
        
        im = ax.imshow(pop_signal, aspect='auto', cmap='RdBu_r', interpolation='none')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Population {pop_idx + 1} Signal\n({units_per_pop} units)', color=colors[pop_idx])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Unit (within pop)')
    
    # Plot 8: Trial-averaged data
    ax8 = fig.add_subplot(gs[1, 3])
    trial_avg = np.mean(train_data, axis=2)
    im8 = ax8.imshow(trial_avg, aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.colorbar(im8, ax=ax8)
    ax8.set_title(f'Trial-Averaged Data\n({ntrial} trials)')
    ax8.set_xlabel('Condition')
    ax8.set_ylabel('Unit')
    
    # Add population boundaries
    for pop_idx in range(1, n_populations):
        ax8.axhline(pop_idx * units_per_pop - 0.5, color='yellow', linewidth=2, linestyle='--')
    
    # Plot 9: Cross-population basis alignment
    ax9 = fig.add_subplot(gs[2, 0:2])
    if n_populations >= 2:
        # Show alignment between first two populations
        U1 = ground_truth['population_bases'][0]
        U2 = ground_truth['population_bases'][1]
        alignment = np.abs(U1.T @ U2)
        
        im9 = ax9.imshow(alignment, aspect='equal', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im9, ax=ax9)
        ax9.set_title('Cross-Population Basis Alignment\n(Pop 1 vs Pop 2)')
        ax9.set_xlabel('Population 2 basis dimension')
        ax9.set_ylabel('Population 1 basis dimension')
    
    # Plot 10: Signal variance per population
    ax10 = fig.add_subplot(gs[2, 2])
    pop_variances = []
    for pop_idx in range(n_populations):
        pop_start = pop_idx * units_per_pop
        pop_end = pop_start + units_per_pop
        pop_var = np.var(true_signal[:, pop_start:pop_end])
        pop_variances.append(pop_var)
    
    ax10.bar(range(n_populations), pop_variances, color=colors[:n_populations])
    ax10.set_xlabel('Population')
    ax10.set_ylabel('Signal Variance')
    ax10.set_title('Signal Variance per Population')
    ax10.set_xticks(range(n_populations))
    
    # Plot 11: Explanation text
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    explanation = (
        f"HETEROGENEOUS POPULATIONS\n\n"
        f"• {n_populations} distinct subpopulations\n"
        f"• {units_per_pop} units per population\n"
        f"• Orthogonality: {ground_truth['population_orthogonality']:.2f}\n\n"
        f"Each population has different\n"
        f"optimal basis orderings.\n\n"
        f"Global approaches will be\n"
        f"suboptimal because they\n"
        f"average across conflicting\n"
        f"preferences."
    )
    ax11.text(0.1, 0.5, explanation, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Heterogeneous Population Data Structure', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig
