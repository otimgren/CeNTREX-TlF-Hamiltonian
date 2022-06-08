from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

import numpy as np
import numpy.typing as npt
import scipy as sp
from centrex_TlF_hamiltonian.states import (
    ElectronicState,
    State,
    UncoupledBasisState,
    CoupledBasisState,
    find_exact_states,
    generate_uncoupled_states_ground,
)

from .basis_transformations import generate_transform_matrix
from .hamiltonian import (
    generate_uncoupled_hamiltonian_X,
    generate_uncoupled_hamiltonian_X_function,
    generate_coupled_hamiltonian_B,
    generate_coupled_hamiltonian_B_function,
)
from .utils import matrix_to_states, reduced_basis_hamiltonian, reorder_evecs

__all__ = [
    "generate_diagonalized_hamiltonian",
    "generate_reduced_X_hamiltonian",
    "generate_reduced_B_hamiltonian",
    "compose_reduced_hamiltonian",
    "generate_total_reduced_hamiltonian",
]


@dataclass
class HamiltonianDiagonalized:
    H: npt.NDArray[np.complex128]
    V: npt.NDArray[np.complex128]
    V_ref: Optional[npt.NDArray[np.complex128]] = None


def generate_diagonalized_hamiltonian(
    hamiltonian: npt.NDArray[np.complex128],
    keep_order: bool = True,
    return_V_ref: bool = False,
    rtol: Optional[float] = None,
) -> HamiltonianDiagonalized:
    _ = np.linalg.eigh(hamiltonian)
    D: npt.NDArray[np.complex128] = _[0]
    V: npt.NDArray[np.complex128] = _[1]
    if keep_order:
        V_ref = np.eye(V.shape[0], dtype=np.complex128)
        D, V = reorder_evecs(V, D, V_ref)

    hamiltonian_diagonalized = V.conj().T @ hamiltonian @ V
    if rtol:
        hamiltonian_diagonalized[
            np.abs(hamiltonian_diagonalized)
            < np.abs(hamiltonian_diagonalized).max() * rtol
        ] = 0
    if not return_V_ref or not keep_order:
        return HamiltonianDiagonalized(hamiltonian_diagonalized, V)
    else:
        return HamiltonianDiagonalized(hamiltonian_diagonalized, V, V_ref)


def generate_reduced_X_hamiltonian(
    ground_states_approx: Sequence[UncoupledBasisState],
    E: npt.NDArray[np.float64] = np.array([0, 0, 0]),
    B: npt.NDArray[np.float64] = np.array([0, 0, 0.001]),
    rtol: Optional[float] = None,
) -> Tuple[List[State], npt.NDArray[np.complex128]]:

    QN = generate_uncoupled_states_ground(
        np.unique([gs.J for gs in ground_states_approx])
    )
    H_X_uc = generate_uncoupled_hamiltonian_X(QN)
    H_X_uc_func = generate_uncoupled_hamiltonian_X_function(H_X_uc)
    S_transform = generate_transform_matrix(QN, ground_states_approx)

    H_X = S_transform.conj().T @ H_X_uc_func(E, B) @ S_transform
    if rtol:
        H_X[np.abs(H_X) < np.abs(H_X).max() * rtol] = 0

    # diagonalize the Hamiltonian
    H_diagonalized = generate_diagonalized_hamiltonian(
        H_X, keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_diag = matrix_to_states(H_diagonalized.V, ground_states_approx)

    ground_states = find_exact_states(
        [1 * gs for gs in ground_states_approx],
        H_diagonalized.H,
        QN_diag,
        V_ref=H_diagonalized.V_ref,
    )

    # ground_states = [gs.remove_small_components() for gs in ground_states]

    H_X_red = reduced_basis_hamiltonian(QN_diag, H_diagonalized.H, ground_states)

    return ground_states, H_X_red


def generate_reduced_B_hamiltonian(
    excited_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.float64] = np.array([0, 0, 0]),
    B: npt.NDArray[np.float64] = np.array([0, 0, 0.001]),
    rtol: Optional[float] = None,
    Jmin: int = 1,
    Jmax: int = 3,
) -> Tuple[List[State], npt.NDArray[np.complex128]]:
    # need to generate other states because excited states are mixed
    Ps = [-1, 1]
    I_F = 1 / 2
    I_Tl = 1 / 2
    QN_B = [
        CoupledBasisState(
            F, mF, F1, J, I_F, I_Tl, P=P, Omega=1, electronic_state=ElectronicState.B
        )
        for J in np.arange(Jmin, Jmax + 1)
        for F1 in np.arange(np.abs(J - I_F), J + I_F + 1)
        for F in np.arange(np.abs(F1 - I_Tl), F1 + I_Tl + 1)
        for mF in np.arange(-F, F + 1)
        for P in Ps
    ]

    for qn in excited_states_approx:
        assert qn.isCoupled, "supply list of CoupledBasisStates"
    H_B = generate_coupled_hamiltonian_B(QN_B)
    H_B_func = generate_coupled_hamiltonian_B_function(H_B)

    H_diagonalized = generate_diagonalized_hamiltonian(
        H_B_func(E, B), keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_B_diag = matrix_to_states(H_diagonalized.V, QN_B)

    excited_states = find_exact_states(
        [1 * e for e in excited_states_approx],
        H_diagonalized.H,
        QN_B_diag,
        V_ref=H_diagonalized.V,
    )

    H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_diagonalized.H, excited_states)
    return excited_states, H_B_red


def compose_reduced_hamiltonian(
    H_X_red: npt.NDArray[np.complex128],
    H_B_red: npt.NDArray[np.complex128],
    element_limit: float = 0.1,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    H_X_red[np.abs(H_X_red) < element_limit] = 0
    H_B_red[np.abs(H_B_red) < element_limit] = 0

    H_int: npt.NDArray[np.complex128] = sp.linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0], dtype=np.complex128)

    return H_int, V_ref_int


def generate_total_reduced_hamiltonian(
    ground_states_approx: Sequence[UncoupledBasisState],
    excited_states_approx: Sequence[CoupledBasisState],
    Jmin: Optional[int] = None,
    Jmax: Optional[int] = None,
    rtol: Optional[float] = None,
) -> Tuple[
    List[State],
    List[State],
    List[State],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    ground_states, H_X_red = generate_reduced_X_hamiltonian(
        ground_states_approx, rtol=rtol
    )

    # Js to include for rotational mixing in B state
    Jexc = np.unique([s.J for s in excited_states_approx])
    if not Jmin:
        Jmin_int = np.min(Jexc)
        Jmin_int = 1 if Jmin_int - 1 < 1 else Jmin_int - 1
    else:
        Jmin_int = Jmin
    if not Jmax:
        Jmax_int = np.max(Jexc) + 1
    else:
        Jmax_int = Jmax

    excited_states, H_B_red = generate_reduced_B_hamiltonian(
        excited_states_approx, Jmin=Jmin_int, Jmax=Jmax_int, rtol=rtol
    )

    H_int, V_ref_int = compose_reduced_hamiltonian(H_X_red, H_B_red)

    QN = ground_states.copy()
    QN.extend(excited_states)
    return ground_states, excited_states, QN, H_int, V_ref_int
