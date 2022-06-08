import numpy as np
from centrex_TlF_hamiltonian.states import CoupledBasisState, State

from .coefficients import B
from .quantum_operators import (
    J2,
    J4,
    J6,
)
from .wigner import threej_f, sixj_f

__all__ = ["Hrot", "H_LD", "H_mhf_Tl", "H_mhf_F", "H_c_Tl", "H_cp1_Tl", "HZz"]


def Hrot(psi: CoupledBasisState, coefficients: B) -> State:
    return (
        coefficients.B_rot * J2(psi)
        - coefficients.D_rot * J4(psi)
        + coefficients.H_const * J6(psi)
    )


###################################################
# Λ doubling term
###################################################


def H_LD(psi: CoupledBasisState, coefficients: B) -> State:
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    P = psi.P
    S = 0

    data = []

    def ME(J, Jprime, Omega, Omegaprime):
        amp = (
            coefficients.q
            * (-1) ** (J - Omegaprime)
            / (2 * np.sqrt(6))
            * threej_f(J, 2, J, -Omegaprime, Omegaprime - Omega, Omega)
            * np.sqrt((2 * J - 1) * 2 * J * (2 * J + 1) * (2 * J + 2) * (2 * J + 3))
        )

        return amp

    for Pprime in [-1, 1]:
        amp = (
            P * (-1) ** (J - S) * ME(J, J, 1, -1)
            + Pprime * (-1) ** (J - S) * ME(J, J, -1, 1)
        ) / 2
        ket = CoupledBasisState(
            F,
            mF,
            F1,
            J,
            I1,
            I2,
            Omega=psi.Omega,
            P=Pprime,
            electronic_state=psi.electronic_state,
        )

        # If matrix element is non-zero, add to list
        if amp != 0:
            data.append((amp, ket))

    return State(data)


###################################################
# Electron Magnetic Hyperfine Operator
###################################################


def H_mhf_Tl(psi: CoupledBasisState, coefficients: B) -> State:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = psi.Omega
    P = psi.P

    # I1, I2, F1 and F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    F1prime = F1
    mFprime = mF
    Fprime = F

    # Container for the states and amplitudes
    data = []

    # Loop over possible values of Jprime
    for Jprime in np.arange(np.abs(J - 1), J + 2):

        # Check that the Jprime and Fprime values are physical
        if np.abs(Fprime - Jprime) <= (I1 + I2):
            # Calculate matrix element
            try:
                amp = coefficients.h1_Tl * (
                    (-1) ** (J + Jprime + F1 + I1 - Omega)
                    * sixj_f(I1, Jprime, F1, J, I1, 1)
                    * threej_f(J, 1, Jprime, -Omega, 0, Omega)
                    * np.sqrt(
                        (2 * J + 1) * (2 * Jprime + 1) * I1 * (I1 + 1) * (2 * I1 + 1)
                    )
                )

            except ValueError:
                amp = 0

            basis_state = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                Omega=psi.Omega,
                P=P,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, basis_state))

    return State(data)


def H_mhf_F(psi: CoupledBasisState, coefficients: B) -> State:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    Omega = psi.Omega
    P = psi.P

    # I1, I2, F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    Fprime = F
    mFprime = mF

    # Initialize container for storing states and matrix elements
    data = []

    # Loop over the possible values of quantum numbers for which the matrix
    # element can be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in np.arange(np.abs(J - 1), J + 2):

        # Loop over possible values of F1prime
        for F1prime in np.arange(np.abs(Jprime - I1), Jprime + I1 + 1):
            try:
                amp = coefficients.h1_F * (
                    (-1) ** (2 * F1prime + F + 2 * J + 1 + I1 + I2 - Omega)
                    * sixj_f(I2, F1prime, F, F1, I2, 1)
                    * sixj_f(Jprime, F1prime, I1, F1, J, 1)
                    * threej_f(J, 1, Jprime, -Omega, 0, Omega)
                    * np.sqrt(
                        (2 * F1 + 1)
                        * (2 * F1prime + 1)
                        * (2 * J + 1)
                        * (2 * Jprime + 1)
                        * I2
                        * (I2 + 1)
                        * (2 * I2 + 1)
                    )
                )

            except ValueError:
                amp = 0

            basis_state = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                P=P,
                Omega=psi.Omega,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, basis_state))

    return State(data)


###################################################
# C(Tl) - term
###################################################


def H_c_Tl(psi: CoupledBasisState, coefficients: B) -> State:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF

    # I1, I2, F and mF are the same for both states
    Jprime = J
    I1prime = I1
    I2prime = I2
    Fprime = F
    F1prime = F1
    mFprime = mF

    # Initialize container for storing states and matrix elements
    data = []

    # Calculate matrix element
    amp = (
        coefficients.c_Tl
        * (-1) ** (J + F1 + I1)
        * sixj_f(I1, J, F1, J, I1, 1)
        * np.sqrt(J * (J + 1) * (2 * J + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
    )

    basis_state = CoupledBasisState(
        Fprime,
        mFprime,
        F1prime,
        Jprime,
        I1prime,
        I2prime,
        Omega=psi.Omega,
        P=psi.P,
        electronic_state=psi.electronic_state,
    )

    # If matrix element is non-zero, add to list
    if amp != 0:
        data.append((amp, basis_state))

    return State(data)


def H_cp1_Tl(psi: CoupledBasisState, coefficients: B) -> State:
    # Find the quantum numbers of the input state
    J = psi.J
    I1 = psi.I1
    I2 = psi.I2
    F1 = psi.F1
    F = psi.F
    mF = psi.mF
    # Omega = psi.Omega
    P = psi.P

    # I1, I2, F and mF are the same for both states
    I1prime = I1
    I2prime = I2
    Fprime = F
    F1prime = F1
    mFprime = mF

    # Total spin is 1
    S = 0

    # Omegaprime is negative of Omega
    # Omegaprime = -Omega

    # Calculate the correct value of q
    # q = Omegaprime

    # Initialize container for storing states and matrix elements
    data = []

    def ME(J, Jprime, Omega, Omegaprime):
        q = Omegaprime
        amp = (
            -0.5
            * coefficients.c1p_Tl
            * (-1) ** (-J + Jprime - Omegaprime + F1 + I1)
            * np.sqrt((2 * Jprime + 1) * (2 * J + 1) * I1 * (I1 + 1) * (2 * I1 + 1))
            * sixj_f(I1, J, F1, Jprime, I1, 1)
            * (
                (-1) ** (J)
                * threej_f(Jprime, 1, J, -Omegaprime, q, 0)
                * threej_f(J, 1, J, 0, q, Omega)
                * np.sqrt(J * (J + 1) * (2 * J + 1))
                + (
                    (-1) ** (Jprime)
                    * threej_f(Jprime, 1, Jprime, -Omegaprime, q, 0)
                    * threej_f(Jprime, 1, J, 0, q, Omega)
                    * np.sqrt(Jprime * (Jprime + 1) * (2 * Jprime + 1))
                )
            )
        )

        return amp

    # Loop over the possible values of quantum numbers for which the matrix
    # element can be non-zero
    # Need Jprime = J+1 ... |J-1|
    for Jprime in range(np.abs(J - 1), J + 2):
        for Pprime in [-1, 1]:
            amp = (
                (
                    P * (-1) ** (J - S) * ME(J, Jprime, 1, -1)
                    + Pprime * (-1) ** (Jprime - S) * ME(J, Jprime, -1, 1)
                )
                * (-1) ** float((J - Jprime) != 0)
                / 2
            )

            ket = CoupledBasisState(
                Fprime,
                mFprime,
                F1prime,
                Jprime,
                I1prime,
                I2prime,
                Omega=psi.Omega,
                P=Pprime,
                electronic_state=psi.electronic_state,
            )

            # If matrix element is non-zero, add to list
            if amp != 0:
                data.append((amp, ket))

    return State(data)


def HZz(psi: CoupledBasisState, coefficients: B) -> State:
    return coefficients.μ_B * psi.mF * psi
