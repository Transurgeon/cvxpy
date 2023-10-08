#![warn(clippy::disallowed_types)]

use crate::linalg::sparse_reshape;
use crate::linops::Shape;
use crate::sparse_tensor::TensorRepresentation;
use numpy::{IntoPyArray, Ix1, PyArray};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
//use std::collections::HashMap;
use rustc_hash::FxHashMap as HashMap;

pub mod linalg;
pub mod linops;
mod sparse_tensor;
pub mod view;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, FromPyObject)]
#[repr(transparent)]
pub struct VariableID(u64);

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[repr(transparent)]
pub struct ParameterID(u64);

pub struct Context {
    pub param_size_p_1: u64, // NonZeroU64,
    pub id_to_column: HashMap<i64, u64>,
    pub param_to_size: HashMap<i64, u64>,
    pub param_to_column: HashMap<i64, u64>,
    pub var_length: u64,
}

///
#[pyfunction]
fn build_matrix<'a>(
    _py: Python<'a>,
    linops: Vec<linops::Linop<'a>>,
    param_size_p_1: u64, // NonZeroU64,
    id_to_column: HashMap<i64, u64>,
    param_to_size: HashMap<i64, u64>,
    param_to_column: HashMap<i64, u64>,
    var_length: u64,
) -> PyResult<(
    &'a PyArray<f64, Ix1>,
    (&'a PyArray<u64, Ix1>, &'a PyArray<u64, Ix1>),
    (u64, u64),
)> {
    let context = Context {
        param_size_p_1,
        id_to_column,
        param_to_size,
        param_to_column,
        var_length,
    };
    let mut offset = 0;

    let mut tensor = TensorRepresentation::new();
    for linop in linops {
        let lin_op_tensor = process_constraint(&linop, &context)?;
        tensor += lin_op_tensor.get_tensor_repr(offset, &context);
        offset += linop.size();
    }
    let (data, (rows, cols)) = tensor.reshape(offset);
    let shape = (offset * (context.var_length + 1), context.param_size_p_1);
    return Ok((
        data.into_pyarray(_py),
        (rows.into_pyarray(_py), cols.into_pyarray(_py)),
        shape,
    ));
}
#[pyfunction]
fn dummy_build_matrix<'a>(
    _py: Python<'a>,
    linops: Vec<linops::Linop<'a>>,
    param_size_p_1: u64, // NonZeroU64,
    id_to_column: HashMap<i64, u64>,
    param_to_size: HashMap<i64, u64>,
    param_to_column: HashMap<i64, u64>,
    var_length: u64,
) -> PyResult<(
    &'a PyArray<f64, Ix1>,
    (&'a PyArray<u64, Ix1>, &'a PyArray<u64, Ix1>),
    (u64, u64),
)>  {
    println!("{:?}", linops[0].ltype);
    panic!();
}
fn process_non_leaf_node_not_stack<'a>(
    linop: &linops::Linop<'a>,
    context: &'a Context,
    f: impl Fn(&mut view::TensorView) -> PyResult<()>,
) -> PyResult<view::TensorView<'a>> {
    assert!(!linop.args.is_empty());

    let mut retval = view::TensorView::new(context);
    for arg in &linop.args {
        let mut view = process_constraint(arg, context)?;
        f(&mut view)?;
        retval += view;
    }

    Ok(retval)
}

// fn get_constant_data<'a>(
//     linop: &linops::Linop<'a>,
//     view: &view::TensorView<'a>,
// ) -> sprs::CsMatViewI<'a, f64, u64> {
//     unimplemented!()
// }

fn process_constraint<'a>(
    linop: &linops::Linop<'a>,
    context: &'a Context,
) -> PyResult<view::TensorView<'a>> {
    let view = match &linop.ltype {
        linops::LinopType::Variable(id) => {
            view::TensorView::new_variable(context, *id, linop.size())
        }
        linops::LinopType::ScalarConst(x) => view::TensorView::new_scalar(context, *x),
        linops::LinopType::DenseConst(array) => {
            view::TensorView::new_dense(context, array.as_array())
        }
        linops::LinopType::SparseConst(mat) => {
            view::TensorView::new_sparse(context, mat.to_matrix()?)
        }
        linops::LinopType::Sum | linops::LinopType::Reshape => {
            process_non_leaf_node_not_stack(linop, context, |_| Ok(()))?
        }
        linops::LinopType::Mul(lhs) => process_non_leaf_node_not_stack(linop, context, |view| {
            // TODO(PTNobel, phschiele): Break up this function
            let const_view = process_constraint(lhs, context)?;
            assert!(!(!view.parameter_free && !const_view.parameter_free));
            assert!(const_view.variable_ids.is_empty());
            assert!(const_view.contains_constant_data);

            let lhs_shape = match lhs.shape {
                Shape::Zero => return Err(PyNotImplementedError::new_err("Implement me")),
                Shape::One(n) => (1, n),
                Shape::Two(m, n) => (m, n),
            };

            let reps: usize = view.rows() / (lhs_shape.1 as usize);

            let const_data = &const_view.tensor[&None];

            if const_view.parameter_free {
                let mat_reshaped = sparse_reshape(const_data[&None][0].view(), lhs_shape);

                let stacked_lhs =
                    sprs::kronecker_product(sprs::CsMatI::eye(reps).view(), mat_reshaped.view());

                for (_, param_map) in view.tensor.iter_mut() {
                    for (_, vec_of_rhs_mats) in param_map.iter_mut() {
                        for rhs_mat in vec_of_rhs_mats.iter_mut() {
                            *rhs_mat = &stacked_lhs * &rhs_mat.view();
                        }
                    }
                }
            } else {
                let mut stacked_lhs = HashMap::default();
                for (param_id, param_vec) in const_data {
                    for param_slice in param_vec {
                        let mat_reshaped = sparse_reshape(param_slice.view(), lhs_shape);
                        stacked_lhs.entry(*param_id).or_insert_with(Vec::new).push(
                            sprs::kronecker_product(
                                // TODO: can we prevent copy of kron()?
                                sprs::CsMatI::eye(reps).view(),
                                mat_reshaped.view(),
                            ),
                        );
                    }
                }

                let mut new_tensor = HashMap::default();
                for (variable_id, param_map) in &view.tensor {
                    new_tensor
                        .entry(*variable_id)
                        .or_insert_with(HashMap::default);

                    let rhs = &param_map[&None][0];

                    for (param_id, param_vec) in &stacked_lhs {
                        new_tensor
                            .get_mut(variable_id)
                            .unwrap()
                            .entry(*param_id)
                            .or_insert_with(Vec::new);

                        for param_slice in param_vec {
                            new_tensor
                                .get_mut(variable_id)
                                .unwrap()
                                .get_mut(param_id)
                                .unwrap()
                                .push(&rhs.view() * &param_slice.view());
                        }
                    }
                }

                view.tensor = new_tensor;
            }

            view.parameter_free &= const_view.parameter_free;

            Ok(())
        })?,
        linops::LinopType::Neg => process_non_leaf_node_not_stack(linop, context, |view| {
            for (_, param_map) in view.tensor.iter_mut() {
                for (_, vec_of_rhs_mats) in param_map.iter_mut() {
                    for rhs_mat in vec_of_rhs_mats.iter_mut() {
                        rhs_mat.scale(-1.0);
                    }
                }
            }
            Ok(())
        })?,
    };

    Ok(view)
}

/// Raises NotImplementedError
#[pyfunction]
fn not_impl_error() -> PyResult<String> {
    let message = "Rust backend isn't done yet..., and I was raised from Rust.";
    Err(PyNotImplementedError::new_err(message))
}

/// A Python module implemented in Rust.
#[pymodule]
fn cvxpy_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(not_impl_error, m)?)?;
    m.add_function(wrap_pyfunction!(dummy_build_matrix, m)?)?;
    Ok(())
}
