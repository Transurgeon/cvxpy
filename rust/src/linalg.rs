#![allow(non_snake_case)] // A lot of linear algebra in this file where we want capital matrices

use sprs::{CsMatI, TriMatI};

pub fn sparse_reshape(A: sprs::CsMatViewI<'_, f64, u64>, (m, n): (u64, u64)) -> CsMatI<f64, u64> {
    //! Reshape A into (m,n) in Fortran (column-major) order.

    let oldn: u64 = A.shape().1 as u64;
    let mut i_idx: Vec<u64> = Vec::with_capacity(A.nnz());
    let mut j_idx: Vec<u64> = Vec::with_capacity(A.nnz());
    let mut values: Vec<f64> = Vec::with_capacity(A.nnz());

    for (v, (oldi, oldj)) in A {
        i_idx.push((oldj * oldn + oldi) % m);
        j_idx.push((oldj * oldn + oldi) / m);
        values.push(*v);
    }

    TriMatI::from_triplets(
        (m.try_into().unwrap(), n.try_into().unwrap()),
        i_idx,
        j_idx,
        values,
    )
    .to_csr()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sparse_reshape() {
        /*
        [[1, 2], <-> [[1],
         [3, 4]]      [3],
                      [2],
                      [4]]
        */

        let A: CsMatI<f64, u64> = TriMatI::from_triplets(
            (2, 2),
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            vec![1., 2., 3., 4.],
        )
        .to_csr();
        let A_reshaped = sparse_reshape(A.view(), (4, 1));
        let A_reshaped_back = sparse_reshape(A_reshaped.view(), (2, 2));

        let expected: CsMatI<f64, u64> =
            TriMatI::from_triplets((4, 1), vec![0, 1, 2, 3], vec![0; 4], vec![1., 3., 2., 4.])
                .to_csr();

        assert_eq!(A_reshaped, expected);
        assert_eq!(A_reshaped_back, A);
    }
}
