from __future__ import annotations

import copy
from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .constants import TlFNuclearSpins

from .states import CoupledBasisState, ElectronicState, UncoupledBasisState
from .utils import parity_X
from .find_states import QuantumSelector, get_unique_basisstates

__all__ = [
    "generate_uncoupled_states_ground",
    "generate_uncoupled_states_excited",
    "generate_coupled_states_ground",
    "generate_coupled_states_excited",
    "states.generate_coupled_states_ground_X",
    "states.generate_coupled_states_excited_B",
]


def generate_uncoupled_states_ground(
    Js: Union[List[int], npt.NDArray[np.int_]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
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
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
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
    Js: Union[List[int], npt.NDArray[np.int_]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
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


def generate_coupled_states_excited(
    Js: Union[List[int], npt.NDArray[np.int_]],
    Ps: Union[int, List[int], Tuple[int]] = 1,
    Omegas: Union[int, List[int], Tuple[int]] = 1,
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F
    if not isinstance(Ps, (list, tuple)):
        _Ps = [Ps]
    else:
        _Ps = list(Ps)
    if not isinstance(Omegas, (list, tuple)):
        _Omegas = [Omegas]
    else:
        _Omegas = list(Omegas)
    QN = np.array(
        [
            CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I_F,
                I_Tl,
                electronic_state=ElectronicState.B,
                P=P,
                Omega=Omega,
            )
            for J in Js
            for F1 in np.arange(np.abs(J - I_F), J + I_F + 1)
            for F in np.arange(np.abs(F1 - I_Tl), F1 + I_Tl + 1)
            for mF in np.arange(-F, F + 1)
            for P in _Ps
            for Omega in _Omegas
        ]
    )
    return QN


def generate_coupled_states_base(
    qn_selector: QuantumSelector, nuclear_spins: TlFNuclearSpins = TlFNuclearSpins()
) -> npt.NDArray[Any]:
    """generate CoupledBasisStates for the quantum numbers given by qn_selector

    Args:
        qn_selector (QuantumSelector): quantum numbers to use to generate the
                                        CoupledBasisStates

    Returns:
        np.ndarray: array of CoupledBasisStates for the excited state
    """
    assert qn_selector.P is not None, "function requires a parity to be set"
    assert (
        qn_selector.J is not None
    ), "function requires a rotational quantum number to be set"
    assert (
        qn_selector.electronic is not None
    ), "function requires electronic state to be set"
    assert qn_selector.Ω is not None, "function requires Ω to be set"

    # generate all combinations
    quantum_numbers = []
    for par in ["J", "F1", "F", "mF", "electronic", "P", "Ω"]:
        par_val = getattr(qn_selector, par)
        quantum_numbers.append(
            [par_val] if not isinstance(par_val, (list, tuple, np.ndarray)) else par_val
        )

    I_Tl = nuclear_spins.I_Tl
    I_F = nuclear_spins.I_F

    QN = []
    # the worst nested loops I've ever created
    Js, F1s, Fs, mFs, estates, Ps, Ωs = quantum_numbers
    for estate in estates:
        for J in Js:
            F1_allowed = np.arange(np.abs(J - I_F), J + I_F + 1)
            F1sl = F1s if F1s[0] is not None else F1_allowed
            for F1 in F1sl:
                if F1 not in F1_allowed:
                    continue
                Fs_allowed = np.arange(np.abs(F1 - I_Tl), F1 + I_Tl + 1)
                Fsl = Fs if Fs[0] is not None else Fs_allowed
                for F in Fsl:
                    if F not in Fs_allowed:
                        continue
                    mF_allowed = np.arange(-F, F + 1)
                    mFsl = mFs if mFs[0] is not None else mF_allowed
                    for mF in mFsl:
                        if mF not in mF_allowed:
                            continue
                        for P in Ps:
                            P = P if not callable(P) else P(J)
                            for Ω in Ωs:
                                QN.append(
                                    CoupledBasisState(
                                        F,
                                        mF,
                                        F1,
                                        J,
                                        I_F,
                                        I_Tl,
                                        electronic_state=estate,
                                        P=P,
                                        Ω=Ω,
                                    )
                                )
    return np.asarray(QN)


def generate_coupled_states_X(
    qn_selector: Union[QuantumSelector, List[QuantumSelector], npt.NDArray[Any]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    """generate ground X state CoupledBasisStates for the quantum numbers given
    by qn_selector

    Args:
        qn_selector (Union[QuantumSelector, list, np.ndarray]): QuantumSelector
            or list/array of QuantumSelectors to use for generating the
            CoupledBasisStates

    Returns:
        np.ndarray: array of CoupledBasisStates for the excited state
    """
    if isinstance(qn_selector, QuantumSelector):
        qns = copy.copy(qn_selector)
        qns.Ω = 0
        qns.P = parity_X
        qns.electronic = ElectronicState.X
        return generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
    elif isinstance(qn_selector, (list, np.ndarray)):
        coupled_states = []
        for qns in qn_selector:
            qns = copy.copy(qns)
            qns.Ω = 0
            qns.P = parity_X
            qns.electronic = ElectronicState.X
            coupled_states.append(
                generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
            )
        return np.asarray(get_unique_basisstates(np.concatenate(coupled_states)))
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )


def generate_coupled_states_B(
    qn_selector: Union[QuantumSelector, List[QuantumSelector], npt.NDArray[Any]],
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> npt.NDArray[Any]:
    """generate excited B state CoupledBasisStates for the quantum numbers given
    by qn_selector

    Args:
        qn_selector (Union[QuantumSelector, list, np.ndarray]): QuantumSelector
            or list/array of QuantumSelectors to use for generating the
            CoupledBasisStates

    Returns:
        np.ndarray: array of CoupledBasisStates for the excited state
    """
    if isinstance(qn_selector, QuantumSelector):
        qns = copy.copy(qn_selector)
        qns.Ω = 1 if qns.Ω is None else qns.Ω
        qns.electronic = ElectronicState.B
        return generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
    elif isinstance(qn_selector, (list, np.ndarray)):
        coupled_states = []
        for qns in qn_selector:
            qns = copy.copy(qns)
            qns.Ω = 1 if qns.Ω is None else qns.Ω
            qns.electronic = ElectronicState.B
            coupled_states.append(
                generate_coupled_states_base(qns, nuclear_spins=nuclear_spins)
            )
        return np.asarray(get_unique_basisstates(np.concatenate(coupled_states)))
    else:
        raise AssertionError(
            "qn_selector required to be of type QuantumSelector, list or np.ndarray"
        )
