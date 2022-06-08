from dataclasses import dataclass
from typing import Callable, Sequence, Any, Union

import numpy as np
import numpy.typing as npt
from centrex_TlF_hamiltonian.states import CoupledBasisState, UncoupledBasisState

from . import B_coupled, X_uncoupled
from .coefficients import B, Coefficients, X

__all__ = [
    "Hamiltonian",
    "HamiltonianUncoupledX",
    "HamiltonianCoupledB",
    "generate_uncoupled_hamiltonian_X",
    "generate_coupled_hamiltonian_B",
    "generate_uncoupled_hamiltonian_X_function",
    "generate_coupled_hamiltonian_B_function",
]


def HMatElems(
    H: Callable,
    QN: Union[
        Sequence[UncoupledBasisState], Sequence[CoupledBasisState], npt.NDArray[Any]
    ],
    coefficients: Coefficients,
) -> npt.NDArray[np.complex128]:
    result = np.zeros((len(QN), len(QN)), dtype=complex)
    for i, a in enumerate(QN):
        for j in range(i, len(QN)):
            b = QN[j]
            val = (1 * a) @ H(b, coefficients)
            result[i, j] = val
            if i != j:
                result[j, i] = np.conjugate(val)
    return result


@dataclass
class Hamiltonian:
    None


@dataclass
class HamiltonianUncoupledX(Hamiltonian):
    Hff: npt.NDArray[np.complex128]
    HSx: npt.NDArray[np.complex128]
    HSy: npt.NDArray[np.complex128]
    HSz: npt.NDArray[np.complex128]
    HZx: npt.NDArray[np.complex128]
    HZy: npt.NDArray[np.complex128]
    HZz: npt.NDArray[np.complex128]


@dataclass
class HamiltonianCoupledB(Hamiltonian):
    Hrot: npt.NDArray[np.complex128]
    H_mhf_Tl: npt.NDArray[np.complex128]
    H_mhf_F: npt.NDArray[np.complex128]
    H_LD: npt.NDArray[np.complex128]
    H_cp1_Tl: npt.NDArray[np.complex128]
    H_c_Tl: npt.NDArray[np.complex128]
    HZz: npt.NDArray[np.complex128]


def generate_uncoupled_hamiltonian_X(
    QN: Union[
        Sequence[UncoupledBasisState], Sequence[CoupledBasisState], npt.NDArray[Any]
    ],
    coefficients: X = X(),
) -> HamiltonianUncoupledX:
    """
    Generate the uncoupled X state hamiltonian for the supplied set of
    basis states.
    Retrieved from a pre-calculated sqlite3 database

    Args:
        QN (array): array of UncoupledBasisStates

    Returns:
        HamiltonianUncoupledX: dataclass to hold uncoupled X hamiltonian terms
    """
    for qn in QN:
        assert qn.isUncoupled, "supply list with UncoupledBasisStates"

    return HamiltonianUncoupledX(
        HMatElems(X_uncoupled.Hff_alt, QN, coefficients),
        HMatElems(X_uncoupled.HSx, QN, coefficients),
        HMatElems(X_uncoupled.HSy, QN, coefficients),
        HMatElems(X_uncoupled.HSz, QN, coefficients),
        HMatElems(X_uncoupled.HZx, QN, coefficients),
        HMatElems(X_uncoupled.HZy, QN, coefficients),
        HMatElems(X_uncoupled.HZz, QN, coefficients),
    )


def generate_coupled_hamiltonian_B(
    QN: Union[Sequence[CoupledBasisState], npt.NDArray[Any]], coefficients: B = B()
) -> HamiltonianCoupledB:
    """Calculate the coupled B state hamiltonian for the supplied set of
    basis states.
    Retrieved from a pre-calculated sqlite3 database

    Args:
        QN (array): array of UncoupledBasisStates

    Returns:
        HamiltonianCoupledB: dataclass to hold coupled B hamiltonian terms
    """
    for qn in QN:
        assert qn.isCoupled, "supply list withCoupledBasisStates"

    return HamiltonianCoupledB(
        HMatElems(B_coupled.Hrot, QN, coefficients),
        HMatElems(B_coupled.H_mhf_Tl, QN, coefficients),
        HMatElems(B_coupled.H_mhf_F, QN, coefficients),
        HMatElems(B_coupled.H_LD, QN, coefficients),
        HMatElems(B_coupled.H_cp1_Tl, QN, coefficients),
        HMatElems(B_coupled.H_c_Tl, QN, coefficients),
        HMatElems(B_coupled.HZz, QN, coefficients),
    )


def generate_uncoupled_hamiltonian_X_function(H: HamiltonianUncoupledX) -> Callable:
    ham_func = (
        lambda E, B: 2
        * np.pi
        * (
            H.Hff
            + E[0] * H.HSx
            + E[1] * H.HSy
            + E[2] * H.HSz
            + B[0] * H.HZx
            + B[1] * H.HZy
            + B[2] * H.HZz
        )
    )
    return ham_func


def generate_coupled_hamiltonian_B_function(H: HamiltonianCoupledB) -> Callable:
    ham_func = (
        lambda E, B: 2
        * np.pi
        * (
            H.Hrot
            + H.H_mhf_Tl
            + H.H_mhf_F
            + H.H_LD
            + H.H_cp1_Tl
            + H.H_c_Tl
            + 0.01 * H.HZz
        )
    )
    return ham_func
