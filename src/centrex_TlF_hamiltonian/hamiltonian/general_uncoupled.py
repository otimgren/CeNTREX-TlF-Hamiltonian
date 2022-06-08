from .coefficients import Coefficients
from .quantum_operators import J2
from centrex_TlF_hamiltonian.states import State, UncoupledBasisState

__all__ = ["Hrot"]

########################################################
# Rotational Term
########################################################


def Hrot(psi: UncoupledBasisState, coefficients: Coefficients) -> State:
    return coefficients.B_rot * J2(psi)
