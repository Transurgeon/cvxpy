use numpy::ndarray::Array1;
use std::collections::hash_map::Entry;
//use std::collections::HashMap;
use rustc_hash::FxHashMap as HashMap;
//use std::collections::HashSet;
use rustc_hash::FxHashSet as HashSet;
use std::iter::zip;
use std::ops::AddAssign;

use crate::{Context, ParameterID, TensorRepresentation, VariableID};

pub struct TensorView<'a> {
    // Consider Optional on things?
    pub variable_ids: HashSet<VariableID>,
    pub contains_constant_data: bool, // if true, VariableID None is present
    pub tensor:
        HashMap<Option<VariableID>, HashMap<Option<ParameterID>, Vec<sprs::CsMatI<f64, u64>>>>,
    // constant_data: Vec<sprs::TriMatI<f64, u64>>,
    pub parameter_free: bool,
    pub context: &'a Context,
}

impl<'a> TensorView<'a> {
    #[inline]
    pub fn new(context: &'a Context) -> Self {
        TensorView::<'a> {
            context,
            variable_ids: HashSet::default(),
            contains_constant_data: false,
            tensor: HashMap::default(),
            // constant_data: Vec::new(),
            parameter_free: false,
        }
    }

    #[inline]
    pub fn new_variable(context: &'a Context, id: VariableID, size: u64) -> Self {
        TensorView::<'a> {
            context,
            variable_ids: HashSet::from_iter([id]),
            contains_constant_data: false,
            tensor: HashMap::from_iter([(
                Some(id),
                HashMap::from_iter([(None, vec![sprs::CsMatI::eye(size.try_into().unwrap())])]),
            )]),
            parameter_free: true,
        }
    }

    #[inline]
    pub fn new_scalar(context: &'a Context, scalar: f64) -> Self {
        TensorView::<'a> {
            context,
            variable_ids: HashSet::default(),
            contains_constant_data: true,
            tensor: HashMap::from_iter([(
                None,
                HashMap::from_iter([(
                    None,
                    vec![
                        sprs::TriMatI::from_triplets((1, 1), vec![0], vec![0], vec![scalar])
                            .to_csr(),
                    ],
                )]),

            )]),
            parameter_free: true,
        }
    }

    #[inline]
    pub fn new_dense(context: &'a Context, array: numpy::ndarray::ArrayView2<f64>) -> Self {
        let size = array.len();
        TensorView::<'a> {
            context,
            variable_ids: HashSet::default(),
            contains_constant_data: true,
            tensor: HashMap::from_iter([(
                None,
                HashMap::from_iter([(
                    None,
                    vec![sprs::CsMatI::csr_from_dense(
                        array
                            .to_shape(((size, 1), numpy::ndarray::Order::ColumnMajor))
                            .expect("TODO: fix this")
                            .view(),
                        f64::EPSILON,
                    )],
                )]),
            )]),
            parameter_free: true,
        }
    }

    #[inline]
    pub fn new_sparse<'b>(context: &'a Context, array: sprs::CsMatViewI<'b, f64, u64>) -> Self {
        // unimplemented!();
        let (m, n) = array.shape();
        let size = (m * n) as u64;

        TensorView::<'a> {
            context,
            variable_ids: HashSet::default(),
            contains_constant_data: true,
            tensor: HashMap::from_iter([(
                None,
                HashMap::from_iter([(
                    None,
                    vec![crate::linalg::sparse_reshape(array, (size, 1)).to_csr()],
                )]),
            )]), // COPY_DATA
            parameter_free: true,
        }
    }

    pub fn rows(&self) -> usize {
        (self.tensor.iter().next()
            .unwrap().1.iter().next()
            .unwrap().1
        )[0].rows()
    }

    pub fn get_tensor_repr(&self, offset: u64, context: &Context) -> TensorRepresentation {
        let mut tensor_repr = TensorRepresentation::new();
        for (var_id, param_mapping) in &self.tensor {
            let var_int = match var_id {
                Some(var_int) => var_int.0.try_into().unwrap(),
                &None => -1,
            };

            for (param_id, param_vec) in param_mapping {
                let param_int = match param_id {
                    Some(param_int) => param_int.0.try_into().unwrap(),
                    &None => -1,
                };
                for (param_offset, param_slice) in param_vec.iter().enumerate() {
                    let nnz = param_slice.nnz();
                    let mut new_v: Vec<f64> = Vec::with_capacity(nnz);
                    let mut new_i: Vec<u64> = Vec::with_capacity(nnz);
                    let mut new_j: Vec<u64> = Vec::with_capacity(nnz);
                    let param_val: u64 =
                        *(context.param_to_column.get(&param_int).unwrap()) + param_offset as u64;
                    let new_param_offset: Vec<u64> = vec![param_val; nnz];

                    for (v, (i, j)) in param_slice.view() {
                        new_v.push(*v);
                        new_i.push(i + offset);
                        new_j.push(j + context.id_to_column.get(&var_int).unwrap());
                    }
                    tensor_repr += TensorRepresentation::from_vecs(
                        Array1::from_vec(new_v),
                        Array1::from_vec(new_i),
                        Array1::from_vec(new_j),
                        Array1::from_vec(new_param_offset),
                    );
                }
            }
        }
        tensor_repr
    }
}

impl<'a> AddAssign for TensorView<'a> {
    fn add_assign(&mut self, rhs: Self) {
        self.variable_ids.extend(rhs.variable_ids);
        self.contains_constant_data |= rhs.contains_constant_data;
        self.parameter_free |= rhs.parameter_free;
        for (var_id, param_map) in rhs.tensor.into_iter() {
            match self.tensor.entry(var_id) {
                Entry::Vacant(ve) => {
                    ve.insert(param_map);
                }

                Entry::Occupied(mut outer_oe) => {
                    for (param_id, rhs_vec) in param_map.into_iter() {
                        match outer_oe.get_mut().entry(param_id) {
                            Entry::Vacant(ve) => {
                                ve.insert(rhs_vec);
                            }
                            Entry::Occupied(mut oe) => {
                                let lhs_vec = oe.get();
                                assert_eq!(rhs_vec.len(), lhs_vec.len());

                                let new_vec = zip(lhs_vec, rhs_vec)
                                    .map(
                                        |(a, b)| a + &b, // COPY_DATA
                                    )
                                    .collect();
                                oe.insert(new_vec);
                            }
                        }
                    }
                }
            }
        }
    }
}
