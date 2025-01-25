mod phase_correlation;
use crate::phase_correlation::{
    blurred_crosscov, calculate_ambient_flow, fft_crosscov, fft_flowvectors, FftError,
};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Convert FftError to PyErr
impl From<FftError> for PyErr {
    fn from(err: FftError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Calculate the global shift vector between two images.
///
/// Parameters
/// ----------
/// im1:
///     A 2D NumPy array representing the first image.
/// im2:
///     A 2D NumPy array representing the second image.
///
/// Returns
/// -------
/// A list containing the global shift vector as `[row_shift, col_shift]`.
#[pyfunction]
fn get_global_shift<'py>(
    _py: Python<'py>,
    im1: PyReadonlyArray2<'py, f64>,
    im2: PyReadonlyArray2<'py, f64>,
) -> PyResult<Option<[f64; 2]>> {
    let im1 = im1.as_array().to_owned();
    let im2 = im2.as_array().to_owned();
    Ok(fft_flowvectors(&im1, &im2, true).map_err(PyErr::from)?)
}

/// Calculate the ambient flow vector around an object in two images.
///
/// This function returns the ambient flow vector calculated around a specific object.
///
/// Parameters
/// ----------
/// obj_extent:
///     A dictionary containing object properties such as `obj_center_row`, `obj_center_col`, and `obj_radius`.
/// img1:
///     A 2D NumPy array representing the first image.
/// img2:
///     A 2D NumPy array representing the second image.
/// params:
///     A dictionary containing configuration parameters such as `FLOW_MARGIN`.
/// grid_size:
///     A tuple representing the grid dimensions.
///
/// Returns
/// -------
/// A list containing the ambient flow vector as `[row_flow, col_flow]`.
#[pyfunction]
fn get_ambient_flow<'py>(
    _py: Python<'py>,
    obj_extent: HashMap<String, f64>,
    img1: PyReadonlyArray2<'py, f64>,
    img2: PyReadonlyArray2<'py, f64>,
    params: HashMap<String, f64>,
    grid_size: (f64, f64, f64),
) -> PyResult<Option<[f64; 2]>> {
    let img1 = img1.as_array().to_owned();
    let img2 = img2.as_array().to_owned();
    Ok(
        calculate_ambient_flow(&obj_extent, &img1, &img2, &params, grid_size)
            .map_err(PyErr::from)?,
    )
}

#[pyfunction]
fn fft_crosscov_rust<'py>(
    _py: Python<'py>,
    img1: PyReadonlyArray2<'py, f64>,
    img2: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let result = fft_crosscov(&img1.as_array().to_owned(), &img2.as_array().to_owned())
        .map_err(PyErr::from)?;
    Ok(PyArray2::from_owned_array(_py, result).unbind())
}

#[pyfunction]
fn blurred_crosscov_rust<'py>(
    _py: Python<'py>,
    img1: PyReadonlyArray2<'py, f64>,
    img2: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let result = blurred_crosscov(&img1.as_array().to_owned(), &img2.as_array().to_owned())
        .map_err(PyErr::from)?;
    Ok(PyArray2::from_owned_array(_py, result).unbind())
}

/// Python module definition for the Rust bindings.
///
/// This module provides functionality for calculating global shifts and ambient flows
/// using FFT-based methods.
#[pymodule]
fn tintx_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(get_global_shift, module)?)?;
    module.add_function(wrap_pyfunction!(get_ambient_flow, module)?)?;
    module.add_function(wrap_pyfunction!(fft_crosscov_rust, module)?)?;
    module.add_function(wrap_pyfunction!(blurred_crosscov_rust, module)?)?;
    Ok(())
}
