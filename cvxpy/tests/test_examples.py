"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import unittest

import numpy as np

import cvxpy as cp
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest


class TestExamples(BaseTest):
    """ Unit tests using example problems. """

    # Find the largest Euclidean ball in the polyhedron.
    def test_chebyshev_center(self) -> None:
        # The goal is to find the largest Euclidean ball (i.e. its center and
        # radius) that lies in a polyhedron described by linear inequalities in this
        # fashion: P = {x : a_i'*x <= b_i, i=1,...,m} where x is in R^2

        # Generate the input data
        a1 = np.array([2, 1])
        a2 = np.array([2, -1])
        a3 = np.array([-1, 2])
        a4 = np.array([-1, -2])
        b = np.ones(4)

        # Create and solve the model
        r = cp.Variable(name='r')
        x_c = cp.Variable(2, name='x_c')
        obj = cp.Maximize(r)
        constraints = [  # TODO have atoms compute values for constants.
            a1.T @ x_c + np.linalg.norm(a1)*r <= b[0],
            a2.T @ x_c + np.linalg.norm(a2)*r <= b[1],
            a3.T @ x_c + np.linalg.norm(a3)*r <= b[2],
            a4.T @ x_c + np.linalg.norm(a4)*r <= b[3],
        ]

        p = cp.Problem(obj, constraints)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 0.447214)
        self.assertAlmostEqual(r.value, result)
        self.assertItemsAlmostEqual(x_c.value, [0, 0])

    # Test issue with numpy scalars.
    def test_numpy_scalars(self) -> None:
        n = 6
        eps = 1e-6
        np.random.seed(10)
        P0 = np.random.randn(n, n)
        eye = np.eye(n)
        P0 = P0.T.dot(P0) + eps * eye

        print(P0)

        P1 = np.random.randn(n, n)
        P1 = P1.T.dot(P1)
        P2 = np.random.randn(n, n)
        P2 = P2.T.dot(P2)
        P3 = np.random.randn(n, n)
        P3 = P3.T.dot(P3)

        q0 = np.random.randn(n, 1)
        q1 = np.random.randn(n, 1)
        q2 = np.random.randn(n, 1)
        q3 = np.random.randn(n, 1)

        r0 = np.random.randn(1, 1)
        r1 = np.random.randn(1, 1)
        r2 = np.random.randn(1, 1)
        r3 = np.random.randn(1, 1)

        slack = cp.Variable()
        # Form the problem
        x = cp.Variable(n)
        objective = cp.Minimize(0.5*cp.quad_form(x, P0) + q0.T @ x + r0 + slack)
        constraints = [0.5*cp.quad_form(x, P1) + q1.T @ x + r1 <= slack,
                       0.5*cp.quad_form(x, P2) + q2.T @ x + r2 <= slack,
                       0.5*cp.quad_form(x, P3) + q3.T @ x + r3 <= slack,
                       ]

        # We now find the primal result and compare it to the dual result
        # to check if strong duality holds i.e. the duality gap is effectively zero
        p = cp.Problem(objective, constraints)
        p.solve(solver=cp.SCS)

        # Note that since our data is random,
        # we may need to run this program multiple times to get a feasible primal
        # When feasible, we can print out the following values
        print(x.value)  # solution
        lam1 = constraints[0].dual_value
        lam2 = constraints[1].dual_value
        lam3 = constraints[2].dual_value
        print(type(lam1))

        P_lam = P0 + lam1*P1 + lam2*P2 + lam3*P3
        q_lam = q0 + lam1*q1 + lam2*q2 + lam3*q3
        r_lam = r0 + lam1*r1 + lam2*r2 + lam3*r3
        dual_result = -0.5*q_lam.T.dot(P_lam).dot(q_lam) + r_lam
        print(dual_result.shape)
        self.assertEqual(intf.shape(dual_result), (1, 1))

    # Tests examples from the README.
    def test_readme_examples(self) -> None:
        import numpy
        numpy.random.seed(1)
        # cp.Problem data.
        m = 30
        n = 20
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)

        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        constraints = [0 <= x, x <= 1]
        p = cp.Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        p.solve(solver=cp.SCS)
        # The optimal value for x is stored in x.value.
        print(x.value)
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print(constraints[0].dual_value)

        ####################################################

        # Scalar variable.
        a = cp.Variable()

        # Column vector variable of length 5.
        x = cp.Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = cp.Variable((4, 7))

        ####################################################

        # Positive scalar parameter.
        m = cp.Parameter(nonneg=True)

        # Column vector parameter with unknown sign (by default).
        cp.Parameter(5)

        # Matrix parameter with negative entries.
        G = cp.Parameter((4, 7), nonpos=True)

        # Assigns a constant value to G.
        G.value = -numpy.ones((4, 7))

        # Raises an error for assigning a value with invalid sign.
        with self.assertRaises(Exception) as cm:
            G.value = numpy.ones((4, 7))
        self.assertEqual(str(cm.exception), "Parameter value must be nonpositive.")

        ####################################################
        a = cp.Variable()
        x = cp.Variable(5)

        # expr is an Expression object after each assignment.
        expr = 2*x
        expr = expr - a
        expr = cp.sum(expr) + cp.norm(x, 2)

        ####################################################

        import numpy as np

        # cp.Problem data.
        n = 10
        m = 5
        A = np.random.randn(n, m)
        b = np.random.randn(n)
        gamma = cp.Parameter(nonneg=True)

        # Construct the problem.
        x = cp.Variable(m)
        objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma*cp.norm(x, 1))
        p = cp.Problem(objective)

        # Assign a value to gamma and find the optimal x.
        def get_x(gamma_value):
            gamma.value = gamma_value
            p.solve(solver=cp.SCS)
            return x.value

        gammas = np.logspace(-1, 2, num=2)
        # Serial computation.
        [get_x(value) for value in gammas]

        ####################################################
        n = 10

        mu = np.random.randn(1, n)
        sigma = np.random.randn(n, n)
        sigma = sigma.T.dot(sigma)
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 1
        x = cp.Variable(n)

        # Constants:
        # mu is the vector of expected returns.
        # sigma is the covariance matrix.
        # gamma is a cp.Parameter that trades off risk and return.

        # cp.Variables:
        # x is a vector of stock holdings as fractions of total assets.

        expected_return = mu @ x
        risk = cp.quad_form(x, sigma)

        objective = cp.Maximize(expected_return - gamma*risk)
        p = cp.Problem(objective, [cp.sum(x) == 1])
        p.solve(solver=cp.SCS)

        # The optimal expected return.
        print(expected_return.value)

        # The optimal risk.
        print(risk.value)

        ###########################################

        N = 50
        M = 40
        n = 10
        data = []
        for i in range(N):
            data += [(1, np.random.normal(loc=1.0, scale=2.0, size=n))]
        for i in range(M):
            data += [(-1, np.random.normal(loc=-1.0, scale=2.0, size=n))]

        # Construct problem.
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 0.1
        # 'a' is a variable constrained to have at most 6 non-zero entries.
        a = cp.Variable(n)  # mi.SparseVar(n, nonzeros=6)
        b = cp.Variable()

        slack = [cp.pos(1 - label*(sample.T @ a - b)) for (label, sample) in data]
        objective = cp.Minimize(cp.norm(a, 2) + gamma*sum(slack))
        p = cp.Problem(objective)
        # Extensions can attach new solve methods to the CVXPY cp.Problem class.
        # p.solve(method="admm")
        p.solve(solver=cp.SCS)

        # Count misclassifications.
        errors = 0
        for label, sample in data:
            if label*(sample.T @ a - b).value < 0:
                errors += 1

        print("%s misclassifications" % errors)
        print(a.value)
        print(b.value)

    def test_advanced1(self) -> None:
        """Code from the advanced tutorial.
        """
        # Solving a problem with different solvers.
        x = cp.Variable(2)
        obj = cp.Minimize(x[0] + cp.norm(x, 1))
        constraints = [x >= 2]
        prob = cp.Problem(obj, constraints)

        # Solve with ECOS.
        prob.solve(solver=cp.ECOS)
        print("optimal value with ECOS:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with CVXOPT.
        if cp.CVXOPT in cp.installed_solvers():
            prob.solve(solver=cp.CVXOPT)
            print("optimal value with CVXOPT:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        # Solve with SCS.
        prob.solve(solver=cp.SCS)
        print("optimal value with SCS:", prob.value)
        self.assertAlmostEqual(prob.value, 6, places=2)

        if cp.CPLEX in cp.installed_solvers():
            # Solve with CPLEX.
            prob.solve(solver=cp.CPLEX)
            print("optimal value with CPLEX:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        if cp.GLPK in cp.installed_solvers():
            # Solve with GLPK.
            prob.solve(solver=cp.GLPK)
            print("optimal value with GLPK:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

            # Solve with GLPK_MI.
            prob.solve(solver=cp.GLPK_MI)
            print("optimal value with GLPK_MI:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        if cp.GUROBI in cp.installed_solvers():
            # Solve with Gurobi.
            prob.solve(solver=cp.GUROBI)
            print("optimal value with GUROBI:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        if cp.XPRESS in cp.installed_solvers():
            # Solve with the Xpress Optimizer.
            prob.solve(solver=cp.XPRESS)
            print("optimal value with XPRESS:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        print(cp.installed_solvers())

    def test_log_det(self) -> None:
        # Generate data
        x = np.array([[0.55, 0.0],
                      [0.25, 0.35],
                      [-0.2, 0.2],
                      [-0.25, -0.1],
                      [-0.0, -0.3],
                      [0.4, -0.2]]).T
        (n, m) = x.shape

        # Create and solve the model
        A = cp.Variable((n, n))
        b = cp.Variable(n)
        obj = cp.Maximize(cp.log_det(A))
        constraints = []
        for i in range(m):
            constraints.append(cp.norm(A @ x[:, i] + b) <= 1)
        p = cp.Problem(obj, constraints)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1.9746, places=2)

    def test_portfolio_problem(self) -> None:
        """Test portfolio problem that caused dcp_attr errors.
        """
        import numpy as np
        import scipy.sparse as sp
        np.random.seed(5)
        n = 100  # 10000
        m = 10  # 100

        F = sp.rand(m, n, density=0.01)
        F.data = np.ones(len(F.data))
        D = sp.eye(n).tocoo()
        D.data = np.random.randn(len(D.data))**2
        Z = np.random.randn(m, 1)
        Z = Z.dot(Z.T)

        x = cp.Variable(n)
        y = F @ x
        # DCP attr causes error because not all the curvature
        # matrices are reduced to constants when an atom
        # is scalar.
        cp.square(cp.norm(D @ x)) + cp.square(Z @ y)

    def test_intro(self) -> None:
        """Test examples from cvxpy.org introduction.
        """
        import numpy

        # cp.Problem data.
        m = 30
        n = 20
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)

        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        prob.solve(solver=cp.SCS, eps=1e-6)
        # The optimal value for x is stored in x.value.
        print(x.value)
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print(constraints[0].dual_value)

        ########################################

        # Create two scalar variables.
        x = cp.Variable()
        y = cp.Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = cp.Minimize(cp.square(x - y))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

        ########################################

        # Create two scalar variables.
        x = cp.Variable()
        y = cp.Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = cp.Minimize(cp.square(x - y))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 0)

        ########################################

        # Replace the objective.
        prob = cp.Problem(cp.Maximize(x + y), prob.constraints)
        print("optimal value", prob.solve(solver=cp.SCS, eps=1e-6))

        self.assertAlmostEqual(prob.value, 1.0, places=3)

        # Replace the constraint (x + y == 1).
        constraints = prob.constraints
        constraints[0] = (x + y <= 3)
        prob = cp.Problem(prob.objective, constraints)
        print("optimal value", prob.solve(solver=cp.SCS, eps=1e-6))

        self.assertAlmostEqual(prob.value, 3.0, places=2)

        ########################################

        x = cp.Variable()

        # An infeasible problem.
        prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
        prob.solve(solver=cp.SCS, eps=1e-6)
        print("status:", prob.status)
        print("optimal value", prob.value)

        self.assertEqual(prob.status, cp.INFEASIBLE)
        self.assertAlmostEqual(prob.value, np.inf)

        # An unbounded problem.
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.ECOS)
        print("status:", prob.status)
        print("optimal value", prob.value)

        self.assertEqual(prob.status, cp.UNBOUNDED)
        self.assertAlmostEqual(prob.value, -np.inf)

        ########################################

        # A scalar variable.
        cp.Variable()

        # Column vector variable of length 5.
        x = cp.Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = cp.Variable((4, 7))

        ########################################
        import numpy

        # cp.Problem data.
        m = 10
        n = 5
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)

        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        print("Optimal value", prob.solve(solver=cp.SCS, eps=1e-6))
        print("Optimal var")
        print(x.value)  # A numpy matrix.

        self.assertAlmostEqual(prob.value, 4.14133859146)

        ########################################
        # Positive scalar parameter.
        m = cp.Parameter(nonneg=True)

        # Column vector parameter with unknown sign (by default).
        cp.Parameter(5)

        # Matrix parameter with negative entries.
        G = cp.Parameter((4, 7), nonpos=True)

        # Assigns a constant value to G.
        G.value = -numpy.ones((4, 7))
        ########################################

        # Create parameter, then assign value.
        rho = cp.Parameter(nonneg=True)
        rho.value = 2

        # Initialize parameter with a value.
        rho = cp.Parameter(nonneg=True, value=2)

        ########################################

        import numpy

        # cp.Problem data.
        n = 15
        m = 10
        numpy.random.seed(1)
        A = numpy.random.randn(n, m)
        b = numpy.random.randn(n)
        # gamma must be positive due to DCP rules.
        gamma = cp.Parameter(nonneg=True)

        # Construct the problem.
        x = cp.Variable(m)
        error = cp.sum_squares(A @ x - b)
        obj = cp.Minimize(error + gamma*cp.norm(x, 1))
        prob = cp.Problem(obj)

        # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
        sq_penalty = []
        l1_penalty = []
        x_values = []
        gamma_vals = numpy.logspace(-4, 6)
        for val in gamma_vals:
            gamma.value = val
            prob.solve(solver=cp.SCS, eps=1e-6)
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            sq_penalty.append(error.value)
            l1_penalty.append(cp.norm(x, 1).value)
            x_values.append(x.value)

        ########################################
        import numpy

        X = cp.Variable((5, 4))
        A = numpy.ones((3, 5))

        # Use expr.size to get the dimensions.
        print("dimensions of X:", X.size)
        print("dimensions of sum(X):", cp.sum(X).size)
        print("dimensions of A @ X:", (A @ X).size)

        # ValueError raised for invalid dimensions.
        try:
            A + X
        except ValueError as e:
            print(e)

    def test_inpainting(self) -> None:
        """Test image in-painting.
        """
        import numpy as np
        np.random.seed(1)
        rows, cols = 20, 20
        # Load the images.
        # Convert to arrays.
        Uorig = np.random.randint(0, 255, size=(rows, cols))

        rows, cols = Uorig.shape
        # Known is 1 if the pixel is known,
        # 0 if the pixel was corrupted.
        Known = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.random() > 0.7:
                    Known[i, j] = 1
        Ucorr = Known*Uorig
        # Recover the original image using total variation in-painting.
        U = cp.Variable((rows, cols))
        obj = cp.Minimize(cp.tv(U))
        constraints = [cp.multiply(Known, U) == cp.multiply(Known, Ucorr)]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)

    def test_advanced2(self) -> None:
        """Test code from the advanced section of the tutorial.
        """
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x)), [x == 2])
        # Get ECOS arguments.
        data, chain, inverse = prob.get_problem_data(cp.ECOS)

        # Get CVXOPT arguments.
        if cp.CVXOPT in cp.installed_solvers():
            data, chain, inverse = prob.get_problem_data(cp.CVXOPT)

        # Get SCS arguments.
        data, chain, inverse = prob.get_problem_data(cp.SCS)

        import ecos

        # Get ECOS arguments.
        data, chain, inverse = prob.get_problem_data(cp.ECOS)
        # Call ECOS solver.
        solution = ecos.solve(data["c"], data["G"], data["h"],
                              ecos_conif.dims_to_solver_dict(data["dims"]),
                              data["A"], data["b"])
        # Unpack raw solver output.
        prob.unpack_results(solution, chain, inverse)

    def test_log_sum_exp(self) -> None:
        """Test log_sum_exp function that failed in Github issue.
        """
        import numpy as np
        np.random.seed(1)
        m = 5
        n = 2
        X = np.ones((m, n))
        w = cp.Variable(n)

        expr2 = [cp.log_sum_exp(cp.hstack([0, X[i, :] @ w])) for i in range(m)]
        expr3 = sum(expr2)
        obj = cp.Minimize(expr3)
        p = cp.Problem(obj)
        p.solve(solver=cp.SCS, max_iters=1)


if __name__ == '__main__':
    unittest.main()
