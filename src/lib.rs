//! Tensor wrapper that exposes the `ndarray` API.
//!
//! This crate provides a small wrapper around the `Tensor` data
//! structure of the `tensorflow` crate, to make it possible to use
//! the `ndarray` API. This wrapper, `NdTensor`, provides the
//! `view` and `view_mut` methods to respectively obtain `ArrayView`
//! and `ArrayViewMut` instances.
//!
//! The following example shows how to wrap a `Tensor` and obtain
//! an `ArrayView`:
//!
//! ~~~
//! use ndarray::{arr2, Ix2};
//! use ndarray_tensorflow::NdTensor;
//! use tensorflow::Tensor;
//!
//! let tensor = Tensor::new(&[2, 3])
//!     .with_values(&[0u32, 1, 2, 3, 4, 5])
//!     .unwrap();
//! let array: NdTensor<_, Ix2> =
//!     NdTensor::from_tensor(tensor)
//!     .unwrap();
//! assert_eq!(array.view(),
//!     arr2(&[[0, 1, 2], [3, 4, 5]]));
//! ~~~

use std::error::Error;
use std::fmt;

use ndarray::{ArrayView, ArrayViewMut, Dimension, IntoDimension};
use tensorflow::{Tensor, TensorType};

/// Mismatch between the tensor shape dimensionality and shape type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeError;

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mismatch between the tensor shape dimensionality and the shape type"
        )
    }
}

impl Error for ShapeError {}

/// A wrapper for `Tensor` that provides views.
///
/// A Tensorflow `Tensor` only provides a limited API. This type is a
/// wrapper around `Tensor` that makes it possible to use a tensor as
/// an `ArrayView` or `ArrayViewMut` from the `ndarray` crate.
pub struct NdTensor<T, D>
where
    T: TensorType,
{
    inner: Tensor<T>,
    shape: D,
}

impl<T, D> NdTensor<T, D>
where
    T: TensorType,
    D: Dimension,
{
    /// Construct an `ArrayTensor` from a `Tensor`.
    ///
    /// This function will return `Err` when the shape is incompatible
    /// with the shape type.
    pub fn from_tensor(tensor: Tensor<T>) -> Result<Self, ShapeError> {
        let mut shape = D::default();

        if shape.ndim() != tensor.dims().len() {
            return Err(ShapeError);
        }

        for idx in 0..shape.ndim() {
            let mut shape_mut = shape.as_array_view_mut();
            shape_mut[idx] = tensor.dims()[idx] as usize;
        }

        Ok(NdTensor {
            inner: tensor,
            shape,
        })
    }

    /// Construct a new zero-initialized wrapped Tensor with the given shape.
    pub fn zeros<I>(shape: I) -> Self
    where
        I: IntoDimension<Dim = D>,
    {
        let shape = shape.into_dimension();

        let shape_vec = shape
            .as_array_view()
            .iter()
            .map(|&d| d as u64)
            .collect::<Vec<_>>();

        NdTensor {
            inner: Tensor::new(&shape_vec),
            shape,
        }
    }

    /// Get reference to the wrapped tensor.
    pub fn inner_ref(&self) -> &Tensor<T> {
        &self.inner
    }

    /// Convert into the wrapped tensor.
    pub fn into_inner(self) -> Tensor<T> {
        self.inner
    }

    /// Get a view of the tensor.
    pub fn view(&self) -> ArrayView<T, D> {
        // Unwrapping is safe here, since the shape/size compatibility
        // is guaranteed by Tensor itself.
        ArrayView::from_shape(self.shape.clone(), &self.inner).unwrap()
    }

    /// Get a mutable view of the tensor.
    pub fn view_mut(&mut self) -> ArrayViewMut<T, D> {
        // Unwrapping is safe here, since the shape/size compatibility
        // is guaranteed by Tensor itself.
        ArrayViewMut::from_shape(self.shape.clone(), &mut self.inner).unwrap()
    }
}

impl<'a, T, D> Into<ArrayView<'a, T, D>> for &'a NdTensor<T, D>
where
    T: TensorType,
    D: Dimension,
{
    fn into(self) -> ArrayView<'a, T, D> {
        self.view()
    }
}

impl<'a, T, D> Into<ArrayViewMut<'a, T, D>> for &'a mut NdTensor<T, D>
where
    T: TensorType,
    D: Dimension,
{
    fn into(self) -> ArrayViewMut<'a, T, D> {
        self.view_mut()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Ix1, Ix2};
    use tensorflow::Tensor;

    use super::NdTensor;

    #[test]
    fn view() {
        let tensor = Tensor::new(&[2, 3])
            .with_values(&[0u32, 1, 2, 3, 4, 5])
            .unwrap();
        let array = NdTensor::from_tensor(tensor).unwrap();
        assert_eq!(array.view(), arr2(&[[0, 1, 2], [3, 4, 5]]));
    }

    #[test]
    fn view_mut() {
        let tensor = Tensor::new(&[2, 3])
            .with_values(&[0u32, 1, 2, 3, 4, 5])
            .unwrap();
        let mut array = NdTensor::from_tensor(tensor).unwrap();
        array.view_mut()[(0, 2)] = 42;

        assert_eq!(array.view(), arr2(&[[0, 1, 42], [3, 4, 5]]));
    }

    #[test]
    #[should_panic]
    fn incorrect_dimensions() {
        let tensor = Tensor::new(&[2, 3])
            .with_values(&[0u32, 1, 2, 3, 4, 5])
            .unwrap();
        let _array: NdTensor<u32, Ix1> = NdTensor::from_tensor(tensor).unwrap();
    }

    #[test]
    fn zeros() {
        let mut array: NdTensor<i32, Ix2> = NdTensor::zeros([2usize, 3]);
        array.view_mut().row_mut(0).assign(&arr1(&[1i32, 2, 3]));
    }
}
