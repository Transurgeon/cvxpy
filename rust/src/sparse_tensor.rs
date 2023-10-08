use numpy::ndarray::{array, Array1, Axis};
use std::ops::AddAssign;

#[derive(Default)]
pub struct TensorRepresentation {
    pub data: Array1<f64>,
    pub row: Array1<u64>,
    pub col: Array1<u64>,
    pub parameter_offset: Array1<u64>,
}

impl TensorRepresentation {
    pub fn new() -> Self {
        TensorRepresentation {
            data: array![],
            row: array![],
            col: array![],
            parameter_offset: array![],
        }
    }

    pub fn from_vecs(
        data: Array1<f64>,
        row: Array1<u64>,
        col: Array1<u64>,
        parameter_offset: Array1<u64>,
    ) -> Self {
        TensorRepresentation {
            data,
            row,
            col,
            parameter_offset,
        }
    }

    pub fn reshape(self, total_rows: u64) -> (Array1<f64>, (Array1<u64>, Array1<u64>)) {
        let mut rows = self.col * total_rows;
        rows = rows + self.row;
        let cols = self.parameter_offset;
        (self.data, (rows, cols))
    }
}

impl AddAssign for TensorRepresentation {
    fn add_assign(&mut self, rhs: Self) {
        self.data.append(Axis(0), rhs.data.view()).unwrap();
        self.row.append(Axis(0), rhs.row.view()).unwrap();
        self.col.append(Axis(0), rhs.col.view()).unwrap();
        self.parameter_offset
            .append(Axis(0), rhs.parameter_offset.view())
            .unwrap();
    }
}
