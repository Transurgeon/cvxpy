"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The symbolic-parametric cone program of the DIFFENGINE backend: a
ParamConeProg subclass that owns a live diff-engine extractor and
re-evaluates the expression trees at the current parameter values on
every apply_parameters() call.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solvers.conic_solvers.conic_solver import build_restruct_mat_sparse
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.cone_stuffing import encode_cone_tensors
from cvxpy.reductions.utilities import ReducedMat


class _Matrices(NamedTuple):
    """The concrete cone matrices at one set of parameter values."""

    q: np.ndarray
    d: float
    A: sp.csc_array
    b: np.ndarray
    P: sp.csc_array | None


class _Tensors(NamedTuple):
    """ParamConeProg's coefficient tensors for one set of matrices."""

    q: sp.csr_array
    A: sp.csc_array
    P: sp.csc_array | None
    reduced_A: ReducedMat
    reduced_P: ReducedMat


class DiffengineParamConeProg(ParamConeProg):
    """A ParamConeProg whose matrices are re-extracted by the C diff engine.

    Parameters stay symbolic in the compiled engine program; each
    ``apply_parameters()`` pushes the current values and re-evaluates
    ``(q, d, A, b, P)`` directly, instead of multiplying parameter tensors.
    The instance always reaches the solver already ``formatted=True`` (the
    cone-restructuring matrix ``R`` pre-applied by ``ConeFormat``), so
    ``ConicSolver.format_constraints`` never replaces it with a stock
    program that could not re-extract.
    """

    def __init__(self, extractor, x, variables, var_id_to_col, constraints,
                 parameters, param_id_to_col, q, d, A, b, P,
                 formatted: bool = False, restruct_mat=None,
                 lower_bounds=None, upper_bounds=None) -> None:
        self.extractor = extractor
        self._restruct_mat = restruct_mat
        # The concrete matrices at the currently-pushed parameter values,
        # post-restructuring. Everything else here is derived from them.
        self._matrices = _Matrices(q, d, A, b, P)
        self._tensors: _Tensors | None = None
        # No tensors are passed: _store_tensors defers them until read.
        super().__init__(None, x, None, variables, var_id_to_col, constraints,
                         parameters, param_id_to_col, P=None,
                         formatted=formatted,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        # The parameter vector the stored matrices were extracted at.
        # Kept per instance, NOT on the shared extractor: a restructured copy
        # and its raw sibling can hold matrices from different parameter
        # vectors. Extraction-once: construction itself extracted at the
        # current values, so the first apply_parameters() short-circuits.
        self._extracted_param_vec = self._param_vec()

    def _store_tensors(self, q, A, P) -> None:
        """Build ParamConeProg's tensors only when something reads them.

        They are an encoding of ``self._matrices``, cost more to encode than
        the extraction that produced them, and nothing on the solve path reads
        them: the solver interfaces use apply_parameters' return values, and
        ask ``has_quad_obj`` rather than inspecting ``P``.
        """

    @property
    def q(self):
        return self._encode_tensors().q

    @property
    def A(self):
        return self._encode_tensors().A

    @property
    def P(self):
        return self._encode_tensors().P

    @property
    def reduced_A(self):
        return self._encode_tensors().reduced_A

    @property
    def reduced_P(self):
        return self._encode_tensors().reduced_P

    @property
    def has_quad_obj(self) -> bool:
        return self._matrices.P is not None

    def _encode_tensors(self) -> _Tensors:
        """Encode the current matrices into ParamConeProg's tensor layout."""
        if self._tensors is None:
            q_t, A_t, P_t = encode_cone_tensors(*self._matrices, self.x.size)
            self._tensors = _Tensors(
                q_t, A_t, P_t,
                ReducedMat(A_t, self.x.size),
                ReducedMat(P_t, self.x.size, quad_form=True))
        return self._tensors

    def _param_vec(self, id_to_param_value=None) -> np.ndarray:
        """Flatten and concatenate the parameter values, in the extractor's order.

        Reading ``p.value`` re-runs a CallbackParam's fold closure, so
        composite parametric coefficients are refreshed here as well.
        """
        return np.concatenate([
            np.asarray(p.value if id_to_param_value is None
                       else id_to_param_value[p.id],
                       dtype=np.float64).flatten(order='F')
            for p in self.parameters])

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Re-evaluate the engine program at the current parameter values."""
        if zero_offset or keep_zeros:
            # These flags belong to the DPP-tensor differentiation contract
            # (diffcp / problem.derivative), which the diff engine does not
            # implement: it re-evaluates a nonlinear map rather than applying
            # a stored linear one.
            raise NotImplementedError(
                "The DIFFENGINE backend does not support the parameter "
                "differentiation contract (zero_offset/keep_zeros); solve "
                "without requires_grad, or use a tensor canon backend.")
        theta = self._param_vec(id_to_param_value)
        stored = self._matrices
        if (np.array_equal(theta, self._extracted_param_vec)
                and (stored.P is not None or not quad_obj)):
            # The stored matrices already correspond to these values.
            if quad_obj:
                return stored.P, stored.q, stored.d, stored.A, stored.b
            return stored.q, stored.d, stored.A, stored.b
        self.extractor.update_parameters(theta)
        q, d, A, b, P = self.extractor.extract(quad_obj)
        if self._restruct_mat is not None:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()
        self._matrices = _Matrices(q, d, A, b, P)
        self._extracted_param_vec = theta
        self._tensors = None
        if quad_obj:
            return P, q, d, A, b
        return q, d, A, b

    def format_for(self, solver):
        """Apply the solver's cone row layout without losing re-extraction.

        The stock path rebuilds a plain ParamConeProg, which would drop the
        engine program. R is structural (constraint types and shapes only), so
        it is built once here and re-applied to every later extraction.
        """
        R = build_restruct_mat_sparse(self.constraints, solver.EXP_CONE_ORDER)
        q, d, A, b, P = self._matrices
        if R is not None:
            A = R @ A
            b = np.asarray(R @ b).flatten()
        formatted = DiffengineParamConeProg(
            self.extractor, self.x, self.variables, self.var_id_to_col,
            self.constraints, self.parameters, self.param_id_to_col,
            q, d, A, b, P, formatted=True, restruct_mat=R,
            lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        # The restructured matrices correspond to THIS instance's values.
        formatted._extracted_param_vec = self._extracted_param_vec
        return formatted

    def split_adjoint(self, del_vars=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")

    def apply_param_jac(self, delc, deld, delA, delb, active_params=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")
