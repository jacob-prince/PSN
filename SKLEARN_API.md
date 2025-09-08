# PSN Sklearn API Implementation

## Overview

The PSN (Partitioning Signal and Noise) package now supports both functional use and sklearn-compatible APIs, providing maximum flexibility for users.

## New PSN Class

The `PSN` class implements sklearn's `BaseEstimator` and `TransformerMixin` interfaces, providing:

- **fit()**: Estimate the optimal denoiser
- **transform()**: Apply denoiser to data
- **fit_transform()**: Combined fitting and transformation
- **score()**: Quality assessment using noise ceiling
- **get_params()** / **set_params()**: Parameter management

## Utility Functions

The package includes reusable scoring functions in `psn.utils`:

- **`negative_mse_columns()`**: Negative mean squared error for each column
- **`r2_score_columns()`**: R² (coefficient of determination) for each column

These can be used independently or as part of the PSN sklearn API.

## Parameters

### basis (string or numpy array)
- `'signal'`: GSN cSb (V = 0) - **DEFAULT**
- `'whitened-signal'`: GSN cNb * GSN cSb (V = 1)
- `'noise'`: GSN cNb (V = 2)
- `'pca'`: naive PCA (V = 3)
- `'random'`: random basis (V = 4) - not recommended
- `matrix`: user-supplied orthonormal basis

### cv (cross-validation strategy)
- `'unit'`: unit thresholding, separate CV threshold per unit (**default**)
- `'population'`: population thresholding, one threshold for all units
- `None`: magnitude thresholding, retains dimensions for 95% signal variance

### scoring (for CV modes 'unit' or 'population')
- `'mse'`: Mean Squared Error (**default**)
- `'r2'`: Coefficient of determination (R²)
- `callable`: any sklearn scoring function or custom function

### mag_threshold
- `0.95` (**default**): proportion of variance to keep when cv=None
- Any scalar between 0 and 1

### unit_groups
- `None` (**default**): each unit gets its own threshold ('unit' mode) or all units share ('population' mode)
- `array-like`: integer array specifying which units share CV thresholds (only for 'unit' mode)

### Other Parameters
- `verbose`: bool - print progress messages
- `wantfig`: bool (**default True**) - generate diagnostic figures
- `gsn_kwargs`: dict - additional GSN parameters

## Usage Examples

### Basic Usage
```python
from psn import PSN

# Fit and transform with defaults
denoiser = PSN()
denoiser.fit(data)  # data shape: (nunits, nconds, ntrials)
denoised_data = denoiser.transform(data)
```

### Population Thresholding with PCA
```python
denoiser = PSN(basis='pca', cv='population')
denoiser.fit(data)
denoised_data = denoiser.transform(data)
```

### Magnitude Thresholding
```python
denoiser = PSN(basis='signal', cv=None, mag_threshold=0.90)
denoiser.fit(data)
denoised_data = denoiser.transform(data)
```

### Custom Basis
```python
custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
denoiser = PSN(basis=custom_basis, cv='unit')
denoiser.fit(data)
denoised_data = denoiser.transform(data)
```

### Sklearn Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = Pipeline([
    ('denoiser', PSN()),
    ('classifier', SomeClassifier())
])

# Grid search over PSN parameters
param_grid = {
    'denoiser__basis': ['signal', 'pca'],
    'denoiser__cv': ['unit', 'population'],
    'denoiser__mag_threshold': [0.90, 0.95, 0.99]
}

grid_search = GridSearchCV(pipeline, param_grid)
grid_search.fit(X, y)
```

## Backward Compatibility

The original functional interface remains unchanged:

```python
from psn import psn

# Original functional API still works
results = psn(data, V=0, opt={'cv_mode': 0}, wantfig=True)
```

## Key Features

1. **Full sklearn compatibility**: Supports pipelines, grid search, cross-validation
2. **Flexible data shapes**: Handles both 2D (trial-averaged) and 3D (single-trial) data
3. **Multiple basis options**: Signal, whitened-signal, noise, PCA, random, or custom
4. **Multiple thresholding strategies**: Unit-wise, population, or magnitude-based
5. **Diagnostic plotting**: Automatic figure generation with `wantfig=True`
6. **Quality scoring**: Built-in noise ceiling scoring for model evaluation

## Files Modified

- `psn/psn.py`: Added `PSN` class with sklearn interface
- `psn/__init__.py`: Added `PSN` to exports
- `examples/sklearn_api_demo.py`: Comprehensive demonstration script

## Dependencies

- scikit-learn (already in requirements.txt)
- All existing PSN dependencies (numpy, scipy, matplotlib, GSN)
