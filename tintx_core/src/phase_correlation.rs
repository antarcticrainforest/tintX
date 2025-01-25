use fft2d::slice::{fft_2d, fftshift, ifft_2d};
use libblur::{gaussian_blur_f32, EdgeMode, FastBlurChannels, ThreadingPolicy};
use ndarray::{s, Array2};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FftError {
    #[error("Input arrays must have the same shape")]
    ShapeMismatch,

    #[error("FFT processing failed")]
    FftProcessingError,
}

fn make_uneven(n: u32) -> u32 {
    (n - n % 2) + 1
}
/// Computes the cross-correlation matrix between two 2D images using the FFT method.
///
/// This function performs the following steps:
/// 1. Validates that the input images have the same dimensions.
/// 2. Converts the input images into complex buffers for FFT processing.
/// 3. Computes the 2D FFT of both input images.
/// 4. Calculates the cross-power spectrum by multiplying the FFT of the second image with the
///    conjugate of the FFT of the first image, normalizing to prevent division by zero.
/// 5. Computes the inverse 2D FFT of the cross-power spectrum to obtain the cross-covariance matrix.
/// 6. Normalizes the result by dividing each element by the total number of elements in the matrix.
/// 7. Applies a frequency shift to rearrange the quadrants, placing the zero-frequency component at the center.
///
/// # Arguments
/// * `im1` - A 2D array (`Array2<f64>`) representing the first image.
/// * `im2` - A 2D array (`Array2<f64>`) representing the second image.
///
/// # Returns
/// Returns a 2D array (`Array2<f64>`) representing the shifted cross-correlation matrix.
/// The zero-frequency component of the matrix is centered.
///
/// # Errors
/// Returns an `FftError::ShapeMismatch` error if the dimensions of the two input images do not match.
///
/// # Example
/// ```
/// use ndarray::array;
/// use my_module::fft_crosscov;
///
/// let im1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let im2 = array![[4.0, 3.0], [2.0, 1.0]];
/// let result = fft_crosscov(&im1, &im2).unwrap();
/// println!("{:?}", result);
/// ```
///
pub fn fft_crosscov(im1: &Array2<f64>, im2: &Array2<f64>) -> Result<Array2<f64>, FftError> {
    if im1.shape() != im2.shape() {
        return Err(FftError::ShapeMismatch);
    }

    let (rows, cols) = (im1.nrows(), im1.ncols());
    let mut im1_buffer: Vec<Complex<f64>> = im1.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut im2_buffer: Vec<Complex<f64>> = im2.iter().map(|&v| Complex::new(v, 0.0)).collect();

    fft_2d(rows, cols, &mut im1_buffer);
    fft_2d(rows, cols, &mut im2_buffer);

    let mut cross_power_spectrum: Vec<Complex<f64>> = im1_buffer
        .par_iter()
        .zip(&im2_buffer)
        .map(|(fft1, fft2)| {
            let normalize = fft1.norm() * fft2.norm();
            if normalize == 0.0 {
                Complex::new(0.0, 0.0)
            } else {
                (*fft2 * fft1.conj()) / normalize
            }
        })
        .collect();

    // Perform inverse 2D FFT to compute the cross-covariance
    ifft_2d(rows, cols, &mut cross_power_spectrum);

    // Normalize the result
    let normalization_factor = (rows * cols) as f64;
    let shifted_result: Vec<f64> = fftshift(
        cols,
        rows,
        &cross_power_spectrum
            .par_iter()
            .map(|v| v.re / normalization_factor)
            .collect::<Vec<f64>>(),
    );

    // Convert the shifted result back to an Array2
    Array2::from_shape_vec((rows, cols), shifted_result).map_err(|_| FftError::ShapeMismatch)
}

/// Create a gaussian blur of the 2D cross-correlation of two consecutuve images
///
/// # Arguments
/// * `im1` - A 2D array (`Array2<f64>`) representing the first image.
/// * `im2` - A 2D array (`Array2<f64>`) representing the second image.
///
/// # Returns
/// Returns a 2D array (`Array2<f64>`) representing the blurred cross-correlation
/// matrix.
///
/// # Errors
/// Returns an `FftError::ShapeMismatch` error if the dimensions of the two input images do not match.
///
pub fn blurred_crosscov(im1: &Array2<f64>, im2: &Array2<f64>) -> Result<Array2<f64>, FftError> {
    // Compute cross-covariance using the optimized fft_crosscov
    let crosscov = fft_crosscov(im1, im2)?;

    let (rows, cols) = crosscov.dim();
    let sigma = (1.0 / 8.0) * im1.shape().iter().cloned().min().unwrap() as f32;
    let kernel_size = make_uneven((sigma * 4.0 + 0.5).ceil() as u32);
    // Flatten and blur in parallel
    let blurred = {
        let flattened: Vec<f32> = crosscov.iter().map(|&v| v as f32).collect();
        let mut blurred = vec![0.0; flattened.len()];
        gaussian_blur_f32(
            &flattened,
            &mut blurred,
            cols as u32,
            rows as u32,
            kernel_size,
            sigma,
            FastBlurChannels::Plane,
            EdgeMode::Reflect,
            ThreadingPolicy::Adaptive,
        );
        blurred // Return the blurred vector
    };
    // Convert the blurred result back to an Array2 in parallel
    Array2::from_shape_vec(
        (rows, cols),
        blurred.par_iter().map(|&v| v as f64).collect::<Vec<f64>>(),
    )
    .map_err(|_| FftError::ShapeMismatch)
}

/// Estimates flow vectors between two images using FFT-based cross-correlation.
///
/// # Arguments
/// - `im1`: The first input image.
/// - `im2`: The second input image.
/// - `global_shift`: Whether to compute the global shift vector.
///
/// # Returns
/// The flow vector as a 2D displacement vector.
pub fn fft_flowvectors(
    im1: &Array2<f64>,
    im2: &Array2<f64>,
    global_shift: bool,
) -> Result<Option<[f64; 2]>, FftError> {
    if !global_shift && (im1.iter().all(|&v| v.abs() == 0.0) || im2.iter().all(|&v| v.abs() == 0.0))
    {
        return Ok(None);
    }
    let (rows, cols) = im1.dim();
    let smoothed = blurred_crosscov(im1, im2)?;
    let max_pos = smoothed
        .indexed_iter()
        .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap());

    if let Some(((x, y), _)) = max_pos {
        // Compute the flow vectors relative to the center
        Ok(Some([
            (x as f64 + 1.0) - (rows as f64 / 2.0),
            (y as f64 + 1.0) - (cols as f64 / 2.0),
        ]))
    } else {
        Ok(None)
    }
}

/// Calculates the ambient flow vector around an object.
///
/// # Arguments
/// - `obj_extent`: A dictionary containing the center and radius of the object.
/// - `img1`: The first input image.
/// - `img2`: The second input image.
/// - `params`: Configuration parameters containing `FLOW_MARGIN`.
/// - `grid_size`: Grid size as a tuple of dimensions.
///
/// # Returns
/// An optional ambient flow vector if calculable.
pub fn calculate_ambient_flow(
    obj_extent: &HashMap<String, f64>,
    img1: &Array2<f64>,
    img2: &Array2<f64>,
    params: &HashMap<String, f64>,
    grid_size: (f64, f64, f64),
) -> Result<Option<[f64; 2]>, FftError> {
    // Calculate margins
    let (margin_r, margin_c) = (
        params["FLOW_MARGIN"] / grid_size.1,
        params["FLOW_MARGIN"] / grid_size.2,
    );

    // Compute bounds for the region of interest (row_lb, row_ub, col_lb, col_ub)
    let (row_lb, row_ub, col_lb, col_ub) = {
        let row_lb =
            (obj_extent["obj_center_row"] - obj_extent["obj_radius"] - margin_r).max(0.0) as usize;
        let row_ub = (obj_extent["obj_center_row"] + obj_extent["obj_radius"] + margin_r)
            .min(img1.shape()[0] as f64) as usize;
        let col_lb =
            (obj_extent["obj_center_col"] - obj_extent["obj_radius"] - margin_c).max(0.0) as usize;
        let col_ub = (obj_extent["obj_center_col"] + obj_extent["obj_radius"] + margin_c)
            .min(img1.shape()[1] as f64) as usize;
        (row_lb, row_ub, col_lb, col_ub)
    };

    // Ensure the bounds are valid
    if row_lb >= row_ub || col_lb >= col_ub {
        return Ok(None);
    }

    // Create binary flow regions in parallel
    let (flow_region1, flow_region2) = {
        let flow_region1 =
            img1.slice(s![row_lb..row_ub, col_lb..col_ub])
                .mapv(|v| if v != 0.0 { 1.0 } else { 0.0 });
        let flow_region2 =
            img2.slice(s![row_lb..row_ub, col_lb..col_ub])
                .mapv(|v| if v != 0.0 { 1.0 } else { 0.0 });
        (flow_region1, flow_region2)
    };
    Ok(fft_flowvectors(&flow_region1, &flow_region2, false)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::HashMap;

    #[test]
    fn test_fft_crosscov() {
        let im1 = array![[1.0, 0.0], [0.0, 0.0]];
        let im2 = array![[0.0, 1.0], [0.0, 0.0]];
        let crosscov = fft_crosscov(&im1, &im2).unwrap();
        assert_eq!(crosscov.shape(), &[2, 2]);
    }

    #[test]
    fn test_fft_flowvectors() {
        let im1 = array![[1.0, 0.0], [0.0, 0.0]];
        let im2 = array![[0.0, 1.0], [0.0, 0.0]];
        let flow = fft_flowvectors(&im1, &im2, false).unwrap();
        assert!(flow.is_some());
    }

    #[test]
    fn test_get_global_shift() {
        let im1 = array![[1.0, 0.0], [0.0, 0.0]];
        let im2 = array![[0.0, 1.0], [0.0, 0.0]];
        let shift = fft_flowvectors(&im1, &im2, true).unwrap();
        assert!(shift.is_some());
    }

    #[test]
    fn test_calculate_ambient_flow() {
        let mut obj_extent = HashMap::new();
        obj_extent.insert("obj_center_row".to_string(), 50.0);
        obj_extent.insert("obj_center_col".to_string(), 50.0);
        obj_extent.insert("obj_radius".to_string(), 10.0);

        let mut params = HashMap::new();
        params.insert("FLOW_MARGIN".to_string(), 5.0);

        let img1 = Array2::from_elem((100, 100), 1.0);
        let img2 = Array2::from_elem((100, 100), 1.0);

        let flow =
            calculate_ambient_flow(&obj_extent, &img1, &img2, &params, (1.0, 1.0, 1.0)).unwrap();
        assert!(flow.is_some());
    }

    #[test]
    fn test_large_fft_flowvectors() {
        let im1 = Array2::from_shape_fn((128, 128), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let im2 = Array2::from_shape_fn(
            (128, 128),
            |(i, j)| if i == (j + 1) % 128 { 1.0 } else { 0.0 },
        );
        let flow = fft_flowvectors(&im1, &im2, false).unwrap();
        assert!(flow.is_some());
        let flow = flow.unwrap();
        assert!(flow[0].abs() > 60.);
        assert!((flow[1]).abs() > 60.);
    }

    #[test]
    fn test_partial_overlap_fft_flowvectors() {
        let mut im1 = Array2::zeros((64, 64));
        let mut im2 = Array2::zeros((64, 64));
        im1.slice_mut(s![0..32, 0..32]).fill(1.0);
        im2.slice_mut(s![32..64, 32..64]).fill(1.0);
        let flow = fft_flowvectors(&im1, &im2, false).unwrap();
        assert!(flow.is_some());
    }

    #[test]
    fn test_no_overlap_fft_flowvectors() {
        let im1 = Array2::zeros((64, 64));
        let mut im2 = Array2::zeros((64, 64));
        im2.slice_mut(s![32..64, 32..64]).fill(1.0);
        let flow = fft_flowvectors(&im1, &im2, false).unwrap();
        assert!(flow.is_none());
    }
}
