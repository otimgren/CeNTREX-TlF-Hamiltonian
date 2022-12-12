from .constants import HamiltonianConstants
from .quantum_operators import J2
from centrex_tlf_hamiltonian.states import State, UncoupledBasisState

__all__ = ["Hrot"]

########################################################
# Rotational Term
########################################################


def Hrot(psi: UncoupledBasisState, coefficients: HamiltonianConstants) -> State:
    return coefficients.B_rot * J2(psi)
