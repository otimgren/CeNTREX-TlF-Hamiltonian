from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, List, Optional, Sequence, Union, no_type_check

import numpy as np
import numpy.typing as npt

from .states import (
    CoupledBasisState,
    ElectronicState,
    State,
    UncoupledBasisState,
    BasisState,
)
from .utils import parity_X, reorder_evecs

__all__ = [
    "generate_uncoupled_states_ground",
    "generate_uncoupled_states_excited",
    "generate_coupled_states_ground",
    "find_state_idx_from_state",
    "find_exact_states",
    "find_closest_vector_idx",
    "check_approx_state_exact_state",
    "BasisStates_from_State",
    "get_indices_quantumnumbers_base",
    "get_indices_quantumnumbers",
    "get_unique_basisstates",
    "QuantumSelector",
]


def generate_uncoupled_states_ground(
    Js: Union[List[int], npt.NDArray[np.int_]], I_Tl: float = 1 / 2, I_F: float = 1 / 2
) -> npt.NDArray[Any]:
    # convert J to int(J); np.int with (-1)**J throws an exception for negative J
    QN = np.array(
        [
            UncoupledBasisState(
                int(J),
                mJ,
                I_Tl,
                m1,
                I_F,
                m2,
                Omega=0,
                P=parity_X(J),
                electronic_state=ElectronicState.X,
            )
            for J in Js
            for mJ in range(-J, J + 1)
            for m1 in np.arange(-I_Tl, I_Tl + 1)
            for m2 in np.arange(-I_F, I_F + 1)
        ]
    )
    return QN


def generate_uncoupled_states_excited(
    Js: Union[List[int], npt.NDArray[np.int_]],
    Ωs: List[int] = [-1, 1],
    I_Tl: float = 1 / 2,
    I_F: float = 1 / 2,
) -> npt.NDArray[Any]:
    QN = np.array(
        [
            UncoupledBasisState(
                J, mJ, I_Tl, m1, I_F, m2, Omega=Ω, electronic_state=ElectronicState.B
            )
            for Ω in Ωs
            for J in Js
            for mJ in range(-J, J + 1)
            for m1 in np.arange(-I_Tl, I_Tl + 1)
            for m2 in np.arange(-I_F, I_F + 1)
        ]
    )
    return QN


def generate_coupled_states_ground(
    Js: Union[List[int], npt.NDArray[np.int_]], I_Tl: float = 1 / 2, I_F: float = 1 / 2
) -> npt.NDArray[Any]:
    QN = np.array(
        [
            CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I_F,
                I_Tl,
                electronic_state=ElectronicState.X,
                P=parity_X(J),
                Omega=0,
            )
            for J in Js
            for F1 in np.arange(np.abs(J - I_F), J + I_F + 1)
            for F in np.arange(np.abs(F1 - I_Tl), F1 + I_Tl + 1)
            for mF in np.arange(-F, F + 1)
        ]
    )
    return QN


def find_state_idx_from_state(
    H: npt.NDArray[np.complex128],
    reference_state: State,
    QN: Sequence[State],
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> int:
    """Determine the index of the state vector most closely corresponding to an
    input state

    Args:
        H (np.ndarray): Hamiltonian to compare to
        reference_state (State): state to find closest state in H to
        QN (list): list of state objects defining the basis for H

    Returns:
        int: index of closest state vector of H corresponding to reference_state
    """

    # determine state vector of reference state
    reference_state_vec = reference_state.state_vector(QN)

    # find eigenvectors of the given Hamiltonian
    _ = np.linalg.eigh(H)
    E: npt.NDArray[np.complex128] = _[0]
    V: npt.NDArray[np.complex128] = _[1]

    if V_ref is not None:
        E, V = reorder_evecs(V, E, V_ref)

    overlaps = np.dot(np.conj(reference_state_vec), V)
    probabilities = overlaps * np.conj(overlaps)

    idx = int(np.argmax(probabilities))

    return idx


def find_closest_vector_idx(
    state_vec: npt.NDArray[np.complex128], vector_array: npt.NDArray[np.complex128]
) -> int:
    """ Function that finds the index of the vector in vector_array that most closely
    matches state_vec. vector_array is array where each column is a vector, typically
    corresponding to an eigenstate of some Hamiltonian.

    inputs:
    state_vec = Numpy array, 1D
    vector_array = Numpy array, 2D

    returns:
    idx = index that corresponds to closest matching vector
    """

    overlaps = np.abs(state_vec.conj().T @ vector_array)
    idx = int(np.argmax(overlaps))

    return idx


def check_approx_state_exact_state(approx: State, exact: State) -> None:
    """Check if the exact found states match the approximate states. The exact
    states are found from the eigenvectors of the hamiltonian and are often a
    superposition of various states.
    The approximate states are used in initial setup of the hamiltonian.

    Args:
        approx (State): approximate state
        exact (State): exact state
    """
    approx_largest = approx.find_largest_component()
    exact_largest = exact.find_largest_component()

    if not type(approx_largest) == type(exact_largest):
        raise TypeError(
            f"can't compare approx ({type(approx_largest)}) and exact "
            f"({type(exact_largest)}), not equal types"
        )

    if isinstance(approx_largest, CoupledBasisState) and isinstance(
        exact_largest, CoupledBasisState
    ):
        assert approx_largest.electronic_state == exact_largest.electronic_state, (
            f"mismatch in electronic state {approx_largest.electronic_state} != "
            f"{exact_largest.electronic_state}"
        )
        assert (
            approx_largest.J == exact_largest.J
        ), f"mismatch in J {approx_largest.J} != {exact_largest.J}"
        assert (
            approx_largest.F == exact_largest.F
        ), f"mismatch in F {approx_largest.F} != {exact_largest.F}"
        assert (
            approx_largest.F1 == exact_largest.F1
        ), f"mismatch in F1 {approx_largest.F1} != {exact_largest.F1}"
        assert (
            approx_largest.mF == exact_largest.mF
        ), f"mismatch in mF {approx_largest.mF} != {exact_largest.mF}"
    else:
        raise NotImplementedError


def find_exact_states(
    states_approx: Sequence[State],
    H: npt.NDArray[np.complex128],
    QN: Sequence[State],
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> List[State]:
    """Find closest approximate eigenstates corresponding to states_approx

    Args:
        states_approx (list): list of State objects
        H (np.ndarray): Hamiltonian, diagonal in basis QN
        QN (list): list of State objects defining the basis for H

    Returns:
        list: list of eigenstates of H closest to states_approx
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref)
        states.append(QN[i])

    return states


def BasisStates_from_State(
    states: Union[Sequence[State], npt.NDArray[Any]]
) -> npt.NDArray[Any]:
    unique = []
    for state in states:
        for amp, basisstate in state:
            if basisstate not in unique:
                unique.append(basisstate)
    return np.array(unique)


@no_type_check
def get_indices_quantumnumbers_base(
    qn_selector: QuantumSelector,
    QN: Union[Sequence[State], Sequence[CoupledBasisState], npt.NDArray[Any]],
    mode: str = "python",
) -> npt.NDArray[np.int_]:
    """Return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector.
    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found

    Args:
        qn_selector (QuantumSelector): QuantumSelector class containing the
                                        quantum numbers to find corresponding
                                        indices for
        QN (Union[list, np.ndarray]): list or array of states

    Raises:
        AssertionError: only supports State and CoupledBasisState types the States list
        or array

    Returns:
        np.ndarray: indices corresponding to the quantum numbers
    """
    assert isinstance(
        qn_selector, QuantumSelector
    ), "supply a QuantumSelector object to select states"
    if isinstance(QN[0], State):
        Js = np.array([s.find_largest_component().J for s in QN])
        F1s = np.array([s.find_largest_component().F1 for s in QN])
        Fs = np.array([s.find_largest_component().F for s in QN])
        mFs = np.array([s.find_largest_component().mF for s in QN])
        estates = np.array([s.find_largest_component().electronic_state for s in QN])
    elif isinstance(QN[0], CoupledBasisState):
        Js = np.array([s.J for s in QN])
        F1s = np.array([s.F1 for s in QN])
        Fs = np.array([s.F for s in QN])
        mFs = np.array([s.mF for s in QN])
        estates = np.array([s.electronic_state for s in QN])
    else:
        raise AssertionError(
            "get_indices_quantumnumbers_base() only supports State and "
            "CoupledBasisState types the States list or array"
        )

    J = qn_selector.J
    F1 = qn_selector.F1
    F = qn_selector.F
    mF = qn_selector.mF
    estate = qn_selector.electronic
    assert estate is not None, "supply the electronic state to select states"

    # generate all combinations
    fields = []
    for par in ["J", "F1", "F", "mF", "electronic"]:
        par = getattr(qn_selector, par)
        fields.append([par] if not isinstance(par, (list, tuple, np.ndarray)) else par)
    combinations = product(*fields)

    mask = np.zeros(len(QN), dtype=bool)
    mask_all = np.ones(len(QN), dtype=bool)
    for J, F1, F, mF, estate in combinations:
        # generate the masks for states in QN where the conditions are met
        mask_J = Js == J if J is not None else mask_all
        mask_F1 = F1s == F1 if F1 is not None else mask_all
        mask_F = Fs == F if F is not None else mask_all
        mask_mF = mFs == mF if mF is not None else mask_all
        mask_es = (
            estates == estate if estate is not None else np.zeros(len(QN), dtype=bool)
        )
        # get the indices of the states in QN to compact
        mask = mask | (mask_J & mask_F1 & mask_F & mask_mF & mask_es)

    if mode == "python":
        return np.where(mask)[0]
    elif mode == "julia":
        return np.where(mask)[0] + 1
    else:
        raise TypeError("mode != python or julia")


def get_indices_quantumnumbers(
    qn_selector: Union[QuantumSelector, Sequence[QuantumSelector], npt.NDArray[Any]],
    QN: Union[Sequence[State], Sequence[CoupledBasisState], npt.NDArray[Any]],
) -> npt.NDArray[np.int_]:
    """return the indices corresponding to all states in QN that correspond to
    the quantum numbers in QuantumSelector or a list of QuantumSelector objects.
    Entries in QuantumSelector quantum numbers can be either single numbers or
    lists/arrays. States with all possible combinations of quantum numbers in
    QuantumSelector are found

    Args:
        qn_selector (Union[QuantumSelector, list, np.ndarray]):
                    QuantumSelector class or list/array of QuantumSelectors
                    containing the quantum numbers to find corresponding indices

        QN (Union[list, np.ndarray]): list or array of states

    Raises:
        AssertionError: only supports State and CoupledBasisState types the States list
        or array

    Returns:
        np.ndarray: indices corresponding to the quantum numbers
    """
    if isinstance(qn_selector, QuantumSelector):
        return get_indices_quantumnumbers_base(qn_selector, QN)
    elif isinstance(qn_selector, (list, np.ndarray)):
        return np.unique(
            np.concatenate(
                [get_indices_quantumnumbers_base(qns, QN) for qns in qn_selector]
            )
        )
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )


def get_unique_basisstates(
    states: Union[Sequence[BasisState], npt.NDArray[Any]]
) -> Union[Sequence[BasisState], npt.NDArray[Any]]:
    """get a list/array of unique BasisStates in the states list/array

    Args:
        states (Union[list, np.ndarray]): list/array of BasisStates

    Returns:
        Union[list, np.ndarray]: list/array of unique BasisStates
    """
    states_unique = []
    for state in states:
        if state not in states_unique:
            states_unique.append(state)

    if isinstance(states, np.ndarray):
        return np.asarray(states_unique)
    else:
        return states_unique


@dataclass
class QuantumSelector:
    """Class for setting quantum numbers for selecting a subset of states
    from a larger set of states

    Args:
        J (Union[NumberType, list, np.ndarray]): rotational quantum number
        F1 (Union[NumberType, list, np.ndarray]):
        F (Union[NumberType, list, np.ndarray]):
        mF (Union[NumberType, list, np.ndarray]):
        electronic (Union[str, list, np.ndarray]): electronic state
    """

    J: Optional[Union[Sequence[int], npt.NDArray[np.int_]]] = None
    F1: Optional[Union[Sequence[float], npt.NDArray[np.float_]]] = None
    F: Optional[Union[Sequence[int], npt.NDArray[np.int_]]] = None
    mF: Optional[Union[Sequence[int], npt.NDArray[np.int_]]] = None
    electronic: Optional[Union[Sequence[ElectronicState], npt.NDArray[Any]]] = None
    P: Optional[Union[Sequence[int], npt.NDArray[np.int_]]] = None
    Ω: Optional[Union[Sequence[int], npt.NDArray[np.int_]]] = None

    def get_indices(
        self,
        QN: Union[Sequence[State], Sequence[CoupledBasisState], npt.NDArray[Any]],
        mode: str = "python",
    ) -> npt.NDArray[np.int_]:
        return get_indices_quantumnumbers_base(self, QN, mode)
