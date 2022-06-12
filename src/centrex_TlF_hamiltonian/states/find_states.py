from dataclasses import dataclass
from itertools import product
from typing import Any, List, Optional, Sequence, Union, no_type_check, Callable

import numpy as np
import numpy.typing as npt

from .states import BasisState, CoupledBasisState, ElectronicState, State
from .utils import reorder_evecs


__all__ = [
    "QuantumSelector",
    "find_state_idx_from_state",
    "find_exact_states_indices",
    "find_exact_states",
    "find_closest_vector_idx",
    "check_approx_state_exact_state",
    "get_indices_quantumnumbers_base",
    "get_indices_quantumnumbers",
    "get_unique_basisstates",
]


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

    J: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None
    F1: Optional[Union[Sequence[float], npt.NDArray[np.float_], float]] = None
    F: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None
    mF: Optional[Union[Sequence[int], npt.NDArray[np.int_], float]] = None
    electronic: Optional[
        Union[Sequence[ElectronicState], npt.NDArray[Any], ElectronicState]
    ] = None
    P: Optional[Union[Callable, Sequence[int], npt.NDArray[np.int_], int]] = None
    Ω: Optional[Union[Sequence[int], npt.NDArray[np.int_], int]] = None

    def get_indices(
        self,
        QN: Union[Sequence[State], Sequence[CoupledBasisState], npt.NDArray[Any]],
        mode: str = "python",
    ) -> npt.NDArray[np.int_]:
        return get_indices_quantumnumbers_base(self, QN, mode)


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


def find_exact_states_indices(
    states_approx: Sequence[State],
    QN: Sequence[State],
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> npt.NDArray[np.int_]:
    """
    Find the indices for the closest approximate eigenstates corresponding to
    states_approx for a Hamiltonian constructed from the quantum states QN_approx.

    Args:
        states_approx (Sequence[State]): approximate states to find the indices for
        QN (Sequence[State]): states from which H was constructed
        H (Optional[npt.NDArray[np.complex128]], optional): Hamiltonian. Defaults to None.
        V (Optional[npt.NDArray[np.complex128]], optional): Eigenvectors. Defaults to None.
        V_ref (Optional[npt.NDArray[np.complex128]], optional): Eigenvector order.
                                                                Defaults to None.

    Returns:
        npt.NDArray[np.int_]: _description_
    """
    # generating the state vectors for states_approx in the basis of QN, which is the
    # basis H was generated from. Note that this is not the actual basis of H, which
    # can be generated with QN and V, with the matrix_to_states function in .utils
    state_vecs = np.asarray([s.state_vector(QN) for s in states_approx])

    # generate the eigenvectors if they were not provided
    if V is None:
        assert H is not None, "Need to supply H if V is None"
        _V = np.linalg.eigh(H)[1]
    else:
        _V = V
    if V_ref is not None:
        _, _V = reorder_evecs(_V, np.ones(len(QN)), V_ref)

    # calculating the overlaps between the state vectors of states_approx in the basis
    # from which H was generated with the eigenvectors of H
    overlaps = np.dot(np.conj(state_vecs), _V) ** 2
    indices = np.argmax(overlaps, axis=1)

    # check if all maximal overlaps are unique
    if np.unique(indices).size != len(indices):
        raise AssertionError(f"duplicate indices found: {list(indices)}")
    return indices


def find_exact_states(
    states_approx: Sequence[State],
    QN_approx: Sequence[State],
    QN: Sequence[State],
    H: Optional[npt.NDArray[np.complex128]] = None,
    V: Optional[npt.NDArray[np.complex128]] = None,
    V_ref: Optional[npt.NDArray[np.complex128]] = None,
) -> List[State]:
    """Find closest approximate eigenstates corresponding to states_approx

    Args:
        states_approx (list): list of State objects to find the closest match to
        QN_approx (list): list of State objects from which H was constructed
        QN (list): list of State objects defining the basis for H
        H (np.ndarray): Hamiltonian, diagonal in basis QN
        V (np.ndarray): eigenvectors in basis QN

    Returns:
        list: list of eigenstates of H closest to states_approx
    """

    indices = find_exact_states_indices(states_approx, QN_approx, H, V, V_ref)
    return [QN[idx] for idx in indices]


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
