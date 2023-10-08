use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use sprs::CsMatViewI;

#[derive(Debug)]
pub struct Linop<'a> {
    pub ltype: LinopType<'a>,
    pub shape: Shape,
    pub args: Vec<Linop<'a>>,
}

#[derive(Debug)]
pub enum LinopType<'a> {
    Variable(crate::VariableID),
    ScalarConst(f64),
    DenseConst(PyReadonlyArray2<'a, f64>),
    SparseConst(PySprsMatrix<'a, f64>),
    Sum,
    Mul(Box<Linop<'a>>),
    Neg,
    Reshape,
}

impl<'a> Linop<'a> {
    #[inline]
    pub fn size(&self) -> u64 {
        match self.shape {
            Shape::Zero => 1,
            Shape::One(n) => n,
            Shape::Two(m, n) => m * n,
        }
    }
}

#[derive(Debug)]
enum Format {
    CSC,
    CSR,
}

#[derive(Debug)]
pub struct PySprsMatrix<'a, T: numpy::Element> {
    format: Format,
    shape: (usize, usize),
    indptr: PyReadonlyArray1<'a, u64>,
    indices: PyReadonlyArray1<'a, u64>,
    data: PyReadonlyArray1<'a, T>,
}

impl<'a, T: numpy::Element> PySprsMatrix<'a, T> {
    pub fn to_matrix(&'a self) -> PyResult<CsMatViewI<'a, T, u64>> {
        Ok(match self.format {
            Format::CSR => CsMatViewI::new(
                self.shape,
                self.indptr.as_slice()?,
                self.indices.as_slice()?,
                self.data.as_slice()?,
            ),
            Format::CSC => CsMatViewI::new_csc(
                self.shape,
                self.indptr.as_slice()?,
                self.indices.as_slice()?,
                self.data.as_slice()?,
            ),
        })
    }
}

#[derive(Debug)]
pub enum Shape {
    Zero,
    One(u64),
    Two(u64, u64),
}

impl<'source> FromPyObject<'source> for Linop<'source> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Check if Linop, if not raise TypeError
        let string_ltype = ob.getattr("type")?.downcast::<PyString>()?.to_str()?;
        let data = ob.getattr("data")?;

        let ltype = match string_ltype {
            "variable" => LinopType::Variable(data.extract()?),
            "mul" => LinopType::Mul(Box::new(data.extract()?)),
            "sum" => LinopType::Sum,
            "neg" => LinopType::Neg,
            "reshape" => LinopType::Reshape,
            "dense_const" => LinopType::DenseConst(data.extract()?),
            "scalar_const" => LinopType::ScalarConst(data.extract()?),
            "sparse_const" => LinopType::SparseConst(data.extract()?),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Illegal linop.type string",
                ))
            }
        };
        let shape = ob.getattr("shape")?;
        let shape = if shape.downcast::<PyTuple>()?.is_empty() {
            Shape::Zero
        } else if let Ok((n,)) = shape.extract() {
            Shape::One(n)
        } else if let Ok((n, m)) = shape.extract() {
            Shape::Two(n, m)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Illegal linop.shape",
            ));
        };

        Ok(Linop {
            ltype,
            shape,
            args: ob.getattr("args")?.extract()?,
        })
    }
}

impl<'source, T: numpy::Element> FromPyObject<'source> for PySprsMatrix<'source, T> {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let format = ob.getattr("format")?.downcast::<PyString>()?.to_str()?;
        let format = match format {
            "csc" => Format::CSC,
            "csr" => Format::CSR,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Matrix must be CSR or CSC",
                ))
            }
        };

        let shape: (usize, usize) = ob.getattr("shape")?.extract()?;
        let data: PyReadonlyArray1<'source, T> = ob.getattr("data")?.extract()?;
        let indices: PyReadonlyArray1<'source, u64> = ob.getattr("indices")?.extract()?;
        let indptr: PyReadonlyArray1<'source, u64> = ob.getattr("indptr")?.extract()?;
        let has_sorted_indices: bool = ob.getattr("has_sorted_indices")?.extract()?;
        if !has_sorted_indices {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "CSC/CSR Matrix must have sorted indicies",
            ));
        }

        Ok(Self {
            format,
            shape,
            data,
            indices,
            indptr,
        })
    }
}
