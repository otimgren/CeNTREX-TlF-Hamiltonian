from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

# this causes a dependency on couplings, which itself has a dependency on states and
# hamiltonians
import centrex_TlF_couplings as couplings
import numpy as np
import numpy.typing as npt
from centrex_TlF_hamiltonian.states import (
    CoupledBasisState,
    ElectronicState,
    QuantumSelector,
    State,
    find_exact_states,
    generate_uncoupled_states_ground,
)
from centrex_TlF_hamiltonian.states.constants import TlFNuclearSpins
from centrex_TlF_hamiltonian.states.find_states import QuantumSelector
from centrex_TlF_hamiltonian.states.generate_states import (
    generate_coupled_states_B,
    generate_coupled_states_ground,
    generate_coupled_states_X,
)
from scipy import linalg

from .basis_transformations import generate_transform_matrix
from .constants import BConstants, XConstants
from .generate_hamiltonian import (
    generate_coupled_hamiltonian_B,
    generate_coupled_hamiltonian_B_function,
    generate_uncoupled_hamiltonian_X,
    generate_uncoupled_hamiltonian_X_function,
)
from .utils import matrix_to_states, reduced_basis_hamiltonian, reorder_evecs

__all__ = [
    "generate_diagonalized_hamiltonian",
    "generate_reduced_X_hamiltonian",
    "generate_reduced_B_hamiltonian",
    "compose_reduced_hamiltonian",
    "generate_total_reduced_hamiltonian",
    "generate_reduced_hamiltonian_transitions",
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
    X_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1e-3]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin: Optional[int] = None,
    Jmax: Optional[int] = None,
    constants: XConstants = XConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    transform: npt.NDArray[np.complex_] = None,
    H_func: Optional[Callable] = None,
) -> Tuple[List[State], npt.NDArray[np.complex128]]:
    """
    Generate the reduced X state hamiltonian.
    Generates the Hamiltonian for all X states from Jmin to Jmax, if provided. Otherwise
    uses the min/max J found in X_states_approx. Then selects the part of the
    hamiltonian that corresponds to X_states_approx.
    The Hamiltonian is diagonal and the returned states are the states corresponding to
    X_states_approx in the basis of the Hamiltonian.

    Args:
        ground_states_approx (Sequence[CoupledBasisState]): States
        E (npt.NDArray[np.float64], optional): Electric field in V/cm. Defaults to
                                                            np.array([0.0, 0.0, 0.0]).
        B (npt.NDArray[np.float64], optional): Magnetic field in G. Defaults to
                                                            np.array([0.0, 0.0, 1e-3]).
        rtol (Optional[float], optional): Remove components smaller than rtol in the
                                                        hamiltonian. Defaults to None.
        stol: (float): Remove superpositions with amplitude smaller than stol from each
                        state. Defaults to 1e-3.
        Jmin (Optional[int], optional): Minimum J to include in the Hamiltonian.
                                        Defaults to None.
        Jmax (Optional[int], optional): Maximum J to include in the Hamiltonian.
                                        Defaults to None.
        constants (XConstants, optional): X state constants. Defaults to XConstants().
        nuclear_spins (TlFNuclearSpins, optional): TlF nuclear spins. Defaults to TlFNuclearSpins().
        transform (npt.NDArray[np.complex_], optional): Transformation matrix from
                                                        uncoupled to coupled for J
                                                        states from Jmin to Jmax.
                                                        Defaults to None.
        H_func (Optional[Callable], optional): Function to generate the Hamiltonian
                                                depending on E and B. Defaults to None.

    Returns:
        Tuple[List[State], npt.NDArray[np.complex128]]: States and Hamiltonian
    """

    # need to generate the other states in case of mixing
    _Jmin = min([gs.J for gs in X_states_approx]) if Jmin is None else Jmin
    _Jmax = max([gs.J for gs in X_states_approx]) if Jmax is None else Jmax

    QN = generate_uncoupled_states_ground(
        Js=np.arange(_Jmin, _Jmax + 1), nuclear_spins=nuclear_spins
    )
    QNc = generate_coupled_states_ground(
        Js=np.arange(_Jmin, _Jmax + 1), nuclear_spins=nuclear_spins
    )
    if H_func is None:
        H_X_uc = generate_uncoupled_hamiltonian_X(QN, constants=constants)
        H_X_uc_func = generate_uncoupled_hamiltonian_X_function(H_X_uc)
    else:
        H_X_uc_func = H_func

    if transform is None:
        S_transform = generate_transform_matrix(QN, QNc)
    else:
        assert transform.shape[0] == len(QN), (
            f"shape of transform incorrect; requires {len(QN), len(QN)}, "
            f"not {transform.shape}",
        )
        S_transform = transform

    H_X = S_transform.conj().T @ H_X_uc_func(E, B) @ S_transform
    if rtol:
        H_X[np.abs(H_X) < np.abs(H_X).max() * rtol] = 0

    # diagonalize the Hamiltonian
    H_diagonalized = generate_diagonalized_hamiltonian(
        H_X, keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_diag = matrix_to_states(H_diagonalized.V, list(QNc))

    ground_states = find_exact_states(
        [1 * gs for gs in X_states_approx],
        list(QNc),
        QN_diag,
        V=H_diagonalized.V,
        # V_ref=H_diagonalized.V_ref,
    )
    ground_states = [gs.remove_small_components(stol) for gs in ground_states]

    # ground_states = [gs.remove_small_components() for gs in ground_states]

    H_X_red = reduced_basis_hamiltonian(QN_diag, H_diagonalized.H, ground_states)

    return ground_states, H_X_red


def generate_reduced_B_hamiltonian(
    B_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1e-6]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin: Optional[int] = None,
    Jmax: Optional[int] = None,
    constants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    H_func: Optional[Callable] = None,
) -> Tuple[List[State], npt.NDArray[np.complex128]]:
    """
    Generate the reduced B state hamiltonian.
    Generates the Hamiltonian for all B states from Jmin to Jmax, if provided. Otherwise
    uses Jmin = 1 and for Jmax the maximum J found in B_states_approx with 2 added.
    Then selects the part of the hamiltonian that corresponds to B_states_approx.
    The Hamiltonian is diagonal and the returned states are the states corresponding to
    B_states_approx in the basis of the Hamiltonian.

    Args:
        B_states_approx (Sequence[CoupledBasisState]): States
        E (npt.NDArray[np.float64], optional): Electric field in V/cm. Defaults to
                                                            np.array([0.0, 0.0, 0.0]).
        B (npt.NDArray[np.float64], optional): Magnetic field in G. Defaults to
                                                            np.array([0.0, 0.0, 1e-3]).
        rtol (Optional[float], optional): Remove components smaller than rtol in the
                                                        hamiltonian. Defaults to None.
        stol: (float): Remove superpositions with amplitude smaller than stol from each
                        state. Defaults to 1e-3.
        Jmin (Optional[int], optional): Minimum J to include in the Hamiltonian.
                                        Defaults to None.
        Jmax (Optional[int], optional): Maximum J to include in the Hamiltonian.
                                        Defaults to None.
        constants (BConstants, optional): B state constants. Defaults to BConstants().
        nuclear_spins (TlFNuclearSpins, optional): TlF nuclear spins. Defaults to TlFNuclearSpins().
        H_func (Optional[Callable], optional): Function to generate the Hamiltonian
                                                depending on E and B. Defaults to None.

    Returns:
        Tuple[List[State], npt.NDArray[np.complex128]]: States and Hamiltonian
    """
    # need to generate the other states in case of mixing
    _Jmin = 1 if Jmin is None else Jmin
    _Jmax = max([gs.J for gs in B_states_approx]) + 2 if Jmax is None else Jmax
    # crude check to see if on Omega basis or P basis
    if any([qn.P is not None for qn in B_states_approx]):
        qn_select = QuantumSelector(J=np.arange(_Jmin, _Jmax + 1), P=[-1, 1], Ω=1)
    else:
        qn_select = QuantumSelector(J=np.arange(_Jmin, _Jmax + 1), P=None, Ω=[-1, 1])
    QN_B = list(generate_coupled_states_B(qn_select, nuclear_spins=nuclear_spins))

    for qn in B_states_approx:
        assert qn.isCoupled, "supply a Sequence of CoupledBasisStates"

    if H_func is None:
        H_B = generate_coupled_hamiltonian_B(QN_B, constants=constants)
        H_B_func = generate_coupled_hamiltonian_B_function(H_B)
    else:
        H_B_func = H_func

    H_diagonalized = generate_diagonalized_hamiltonian(
        H_B_func(E, B), keep_order=True, return_V_ref=True, rtol=rtol
    )

    # new set of quantum numbers:
    QN_B_diag = matrix_to_states(H_diagonalized.V, QN_B)

    excited_states = find_exact_states(
        [1 * e for e in B_states_approx], QN_B, QN_B_diag, V=H_diagonalized.V
    )
    excited_states = [es.remove_small_components(stol) for es in excited_states]

    H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_diagonalized.H, excited_states)
    return excited_states, H_B_red


def compose_reduced_hamiltonian(
    H_X_red: npt.NDArray[np.complex128],
    H_B_red: npt.NDArray[np.complex128],
    element_limit: float = 0.1,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    H_X_red[np.abs(H_X_red) < element_limit] = 0
    H_B_red[np.abs(H_B_red) < element_limit] = 0

    H_int: npt.NDArray[np.complex128] = linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0], dtype=np.complex128)

    return H_int, V_ref_int


@dataclass
class ReducedHamiltonian:
    X_states: List[State]
    B_states: List[State]
    QN: List[State]
    H_int: npt.NDArray[np.complex_]
    V_ref_int: npt.NDArray[np.complex_]
    X_states_basis: Optional[List[State]] = None
    B_states_basis: Optional[List[State]] = None
    QN_basis: Optional[List[State]] = None


def generate_total_reduced_hamiltonian(
    X_states_approx: Sequence[CoupledBasisState],
    B_states_approx: Sequence[CoupledBasisState],
    E: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1e-6]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    X_constants: XConstants = XConstants(),
    B_constants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
    transform: npt.NDArray[np.complex128] = None,
    H_func_X: Optional[Callable] = None,
    H_func_B: Optional[Callable] = None,
) -> ReducedHamiltonian:
    """
    Generate the total reduced hamiltonian for all X and B states in X_states_approx and
    B_states_approx, from an X state hamiltonian for all states from Jmin_X to Jmax_X
    and a B state Hamiltonian for all states from Jmin_B to Jmax_B.
    Returns the X_states, B_states, total states, Hamiltonian and V_reference_int which
    keeps track of the state ordering.

    Args:
        X_states_approx (Sequence[CoupledBasisState]): X_states_approx to generate the

        B_states_approx (Sequence[CoupledBasisState]): _description_
        E (npt.NDArray[np.float64], optional): Electric field. Defaults to
                                                np.array([0.0, 0.0, 0.0]).
        B (npt.NDArray[np.float64], optional): Magnetic field. Defaults to
                                                np.array([0.0, 0.0, 1e-6]).
        rtol (Optional[float], optional): Tolerance for the Hamiltonian. Defaults to
                                            None.
        stol: (float): Remove superpositions with amplitude smaller than stol from each
                        state. Defaults to 1e-3.
        Jmin_X (Optional[int], optional): Jmin for the X state Hamiltonian. Defaults to
                                            None.
        Jmax_X (Optional[int], optional): Jmax for the X state Hamiltonian. Defaults to
                                            None.
        Jmin_B (Optional[int], optional): Jmin for the B state Hamiltonian. Defaults to
                                            None.
        Jmax_B (Optional[int], optional): Jmax for the B state Hamiltonian. Defaults to
                                            None.
        X_constants (XConstants, optional): X state constants. Defaults to XConstants().
        B_constants (BConstants, optional): B state constants. Defaults to BConstants().
        nuclear_spins (TlFNuclearSpins, optional): TlF nuclear spins. Defaults to
                                                    TlFNuclearSpins().
        transform (npt.NDArray[np.complex128], optional): transformation matrix to
                                                            transform the uncoupled X
                                                            state Hamiltonian to
                                                            coupled. Defaults to None.
        H_func_X (Optional[Callable], optional): Function to generate the X state
                                                    Hamiltonian for E and B. Defaults
                                                    to None.
        H_func_B (Optional[Callable], optional): Function to generate the B state
                                                    Hamiltonian for E and B. Defaults
                                                    to None.

    Returns:
        ReducedHamiltonian: Dataclass holding the X states, B states, total states,
                            Hamiltonian and reference eigenvectors
    """
    ground_states, H_X_red = generate_reduced_X_hamiltonian(
        X_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin=Jmin_X,
        Jmax=Jmax_X,
        constants=X_constants,
        nuclear_spins=nuclear_spins,
        H_func=H_func_X,
        transform=transform,
    )

    excited_states, H_B_red = generate_reduced_B_hamiltonian(
        B_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin=Jmin_B,
        Jmax=Jmax_B,
        constants=B_constants,
        nuclear_spins=nuclear_spins,
        H_func=H_func_B,
    )

    H_int, V_ref_int = compose_reduced_hamiltonian(H_X_red, H_B_red)

    QN = ground_states.copy()
    QN.extend(excited_states)
    return ReducedHamiltonian(ground_states, excited_states, QN, H_int, V_ref_int)


def generate_reduced_hamiltonian_transitions(
    transitions: Sequence[
        Union[couplings.OpticalTransition, couplings.MicrowaveTransition]
    ],
    E: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    B: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1e-6]),
    rtol: Optional[float] = None,
    stol: float = 1e-3,
    Jmin_X: Optional[int] = None,
    Jmax_X: Optional[int] = None,
    Jmin_B: Optional[int] = None,
    Jmax_B: Optional[int] = None,
    Xconstants: XConstants = XConstants(),
    Bconstants: BConstants = BConstants(),
    nuclear_spins: TlFNuclearSpins = TlFNuclearSpins(),
) -> ReducedHamiltonian:
    _J_ground: List[List[int]] = []
    excited_states_selectors = []

    for transition in transitions:
        if isinstance(transition, couplings.OpticalTransition):
            excited_states_approx_qn_select = transition.qn_select_excited
            excited_states_approx = list(
                generate_coupled_states_B(excited_states_approx_qn_select)
            )
            excited_states, excited_hamiltonian = generate_reduced_B_hamiltonian(
                B_states_approx=excited_states_approx,
                E=E,
                B=B,
                rtol=rtol,
                stol=stol,
                Jmin=Jmin_B,
                Jmax=Jmax_B,
                constants=Bconstants,
                nuclear_spins=nuclear_spins,
            )

            # figure out which ground states are involved
            excited_states = [s.remove_small_components(1e-3) for s in excited_states]
            # only couplings to opposite parities are allowed
            Js_excited: npt.NDArray[np.int_] = np.unique(
                [s.J for es in excited_states for a, s in es]
            )
            Js_ground = list(np.arange(Js_excited.min() - 1, Js_excited.max() + 2))
            Js_ground = [J for J in Js_ground if (-1) ** J == transition.P_ground]

            _J_ground.extend(Js_ground)
            excited_states_selectors.append(excited_states_approx_qn_select)

        if isinstance(transition, couplings.MicrowaveTransition):
            J_ground.extend([transition.J_ground, transition.J_excited])

    # removing duplicates
    J_ground: List[int] = list(np.unique(_J_ground))

    ground_states_approx_qn_select = QuantumSelector(
        J=J_ground, electronic=ElectronicState.X, Ω=0
    )
    ground_states_approx = list(
        generate_coupled_states_X(ground_states_approx_qn_select)
    )

    ground_states, ground_hamiltonian = generate_reduced_X_hamiltonian(
        X_states_approx=ground_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        stol=stol,
        Jmin=Jmin_X,
        Jmax=Jmax_X,
        constants=Xconstants,
        nuclear_spins=nuclear_spins,
    )

    excited_states_approx = list(generate_coupled_states_B(excited_states_selectors))
    excited_states, excited_hamiltonian = generate_reduced_B_hamiltonian(
        B_states_approx=excited_states_approx,
        E=E,
        B=B,
        rtol=rtol,
        Jmin=Jmin_B,
        Jmax=Jmax_B,
        constants=Bconstants,
        nuclear_spins=nuclear_spins,
    )

    H_total, V_ref_total = compose_reduced_hamiltonian(
        ground_hamiltonian, excited_hamiltonian
    )

    ground_states_approx = [1 * s for s in ground_states_approx]
    excited_states_approx = [1 * s for s in excited_states_approx]

    return ReducedHamiltonian(
        X_states=ground_states,
        B_states=excited_states,
        QN=ground_states + excited_states,
        H_int=H_total,
        V_ref_int=V_ref_total,
        X_states_basis=ground_states_approx,
        B_states_basis=excited_states_approx,
        QN_basis=ground_states_approx + excited_states_approx,
    )

