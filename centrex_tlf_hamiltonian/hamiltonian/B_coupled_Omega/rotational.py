from centrex_tlf_hamiltonian.states import CoupledBasisState, State

from ..constants import BConstants
from ..quantum_operators import J2, J4, J6


def Hrot(psi: CoupledBasisState, constants: BConstants) -> State:
    """
    Rotational Hamiltonian for the B-state.
    """
    return (
        constants.B_rot * J2(psi)
        + constants.D_rot * J4(psi)
        + constants.H_const * J6(psi)
    )

