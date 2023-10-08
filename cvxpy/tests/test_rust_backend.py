import cvxpy_rust
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.lin_ops import LinOp


def test_example():
    # Problem data.
    m = 5
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    c = np.random.randn(n)

    # Construct the problem.
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(c @ x)
    constraints = [A @ x <= 0, x <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    prob.solve(canon_backend='RUST')
    print(prob.value)


def test_rust_linop_parsing():

    wrong_type = LinOp("wrong_type", tuple(), [], 1)
    with pytest.raises(ValueError, match="Illegal linop.type string"):
        cvxpy_rust.build_matrix([wrong_type], 1, {}, {}, {}, 1)

    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        var = LinOp("variable", tuple(), [], 1.)  # Note the '.'
        cvxpy_rust.build_matrix([var], 1, {}, {}, {}, 1)

    with pytest.raises(NotImplementedError, match="Rust backend isn't done yet"):
        var = LinOp("variable", tuple(), [], 1)
        cvxpy_rust.build_matrix([var], 1, {}, {}, {}, 1)
