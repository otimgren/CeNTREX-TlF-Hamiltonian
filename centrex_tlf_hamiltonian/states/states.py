from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import sympy as sp

from .utils import CGc

__all__ = [
    "ElectronicState",
    "BasisState",
    "CoupledBasisState",
    "UncoupledBasisState",
    "State",
    "BasisStates_from_State",
    "Basis",
]


class ElectronicState(Enum):
    X = auto()
    B = auto()


class Basis(Enum):
    Uncoupled = auto()
    CoupledP = auto()
    CoupledΩ = auto()


class Parity(Enum):
    Pos = +1
    Neg = -1


@dataclass
class BasisState(abc.ABC):
    J: int
    electronic_state: Optional[ElectronicState]
    isCoupled: bool
    isUncoupled: bool
    basis: Optional[Basis]

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]) -> State:
        raise NotImplementedError

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> State:
        raise NotImplementedError

    def __matmul__(self, other: BasisState) -> Union[int, float, complex]:
        raise NotImplementedError

    def __add__(self, other: BasisState) -> State:
        raise NotImplementedError


class CoupledBasisState(BasisState):
    # constructor
    def __init__(
        self,
        F: int,
        mF: int,
        F1: float,
        J: int,
        I1: float,
        I2: float,
        Omega: int = 0,
        P: Optional[int] = None,
        electronic_state: Optional[ElectronicState] = None,
        energy: Optional[float] = None,
        Ω: Optional[int] = None,
        v: Optional[int] = None,
        basis: Optional[Basis] = None,
    ):
        self.F, self.mF = F, mF
        self.F1 = F1
        self.J = J
        self.I1 = I1
        self.I2 = I2

        if Ω is not None:
            self.Ω: int = Ω
            self.Omega = self.Ω
        elif Omega is not None:
            self.Omega = Omega
            self.Ω = self.Omega
        else:
            raise AssertionError("need to supply either Omega or Ω")
        self.P = P
        self.electronic_state = electronic_state
        self.energy = energy
        self.isCoupled = True
        self.isUncoupled = False

        # determine which basis we are in
        if basis is not None:
            self.basis = basis
        elif (self.P is None) and (self.electronic_state == ElectronicState.B):
            self.basis = Basis.CoupledΩ
        elif self.electronic_state == ElectronicState.B:
            self.basis = Basis.CoupledP
        elif self.electronic_state == ElectronicState.X:
            self.basis = Basis.CoupledP
        else:
            self.basis = None
        self.v = v

    # equality testing
    def __eq__(self, other: object) -> bool:
        # return self.__hash__() == other.__hash__()
        if not isinstance(other, CoupledBasisState):
            return False
        else:
            return (
                self.F == other.F
                and self.mF == other.mF
                and self.I1 == other.I1
                and self.I2 == other.I2
                and self.F1 == other.F1
                and self.J == other.J
                and self.Omega == other.Omega
                and self.P == other.P
                and self.electronic_state == other.electronic_state
                and self.v == other.v
            )

    # inner product
    def __matmul__(self, other: BasisState) -> Union[int, float, complex]:
        if not isinstance(other, BasisState):
            raise TypeError(
                "can only matmul CoupledBasisState with CoupledBasisState or "
                f"UncoupledBasisState (not {type(other)})"
            )
        if other.isCoupled:
            if self == other:
                return 1
            else:
                return 0
        else:
            return State([(1, other)]) @ self.transform_to_uncoupled()

    # superposition: addition
    def __add__(self, other: BasisState) -> State:
        if self == other:
            return State([(2, self)])
        elif isinstance(other, CoupledBasisState):
            return State([(1, self), (1, other)])
        else:
            raise TypeError(
                f"can only add CoupledBasisState (not {type(other)}) "
                "to CoupledBasisState"
            )

    # superposition: subtraction
    def __sub__(self, other: BasisState):
        if self == other:
            return 0
        elif isinstance(other, CoupledBasisState):
            return State([(1, self), (-1, other)])
        else:
            raise TypeError(
                f"can only subtract CoupledBasisState (not {type(other)}) "
                "from CoupledBasisState"
            )

    # scalar product (psi * a)
    def __mul__(self, a: Union[complex, float, int]) -> State:
        return State([(a, self)])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> State:
        return self * a

    def __hash__(self) -> int:
        if self.P is not None:
            P = self.P
        else:
            P = 0
        ev = self.electronic_state.value if self.electronic_state is not None else 0
        quantum_numbers = (
            int(self.J),
            float(self.F1),
            int(self.F),
            int(self.mF),
            float(self.I1),
            float(self.I2),
            int(P),
            int(self.Omega),
            ev,
        )
        return int(hashlib.md5(json.dumps(quantum_numbers).encode()).hexdigest(), 16)

    def __repr__(self) -> str:
        return self.state_string()

    def state_string(self) -> str:
        F = sp.S(str(self.F), rational=True)
        mF = sp.S(str(self.mF), rational=True)
        F1 = sp.S(str(self.F1), rational=True)
        J = sp.S(str(self.J), rational=True)
        I1 = sp.S(str(self.I1), rational=True)
        I2 = sp.S(str(self.I2), rational=True)
        if self.P is not None:
            if self.P == 1:
                P = "+"
            elif self.P == -1:
                P = "-"
        Omega = self.Omega
        v = self.v

        string = f"J = {J}, F₁ = {F1}, F = {F}, mF = {mF}, I₁ = {I1}, I₂ = {I2}"

        if self.electronic_state is not None:
            string = f"{self.electronic_state.name}, {string}"
        if self.P is not None:
            string = f"{string}, P = {P}"
        if Omega is not None:
            string = f"{string}, Ω = {Omega}"
        if v is not None:
            string = f"{string}, v = {v}"
        return "|" + string + ">"

    def state_string_custom(self, to_print: List[str]) -> str:
        F = sp.S(str(self.F), rational=True)
        mF = sp.S(str(self.mF), rational=True)
        F1 = sp.S(str(self.F1), rational=True)
        J = sp.S(str(self.J), rational=True)
        I1 = sp.S(str(self.I1), rational=True)
        I2 = sp.S(str(self.I2), rational=True)
        if self.P is not None:
            if self.P == 1:
                P = "+"
            elif self.P == -1:
                P = "-"
        Omega = self.Omega
        v = self.v

        string = ""
        for val in to_print:
            if val == "J":
                string += f"J = {J}, "
            elif val == "F1":
                string += f"F₁ = {F1}, "
            elif val == "F":
                string += f"F = {F}, "
            elif val == "mF":
                string += f"mF = {mF}, "
            elif val == "I1":
                string += f"I₁ = {I1}, "
            elif val == "I2":
                string += f"I₂ = {I2}, "
            elif val == "electronic":
                if self.electronic_state is not None:
                    string = f"{self.electronic_state.name}, {string}"
            elif val == "P":
                if P is not None:
                    string += f"P = {P}, "
            elif val == "Ω":
                if Omega is not None:
                    string += f"Ω = {Omega}, "
            elif val == "v":
                if v is not None:
                    string += f"v = {v}, "
        string = string.strip(", ")
        return "|" + string + ">"

    def print_quantum_numbers(self, printing: bool = False) -> str:
        if printing:
            print(self.state_string())
        return self.state_string()

    # A method to transform from coupled to uncoupled basis
    def transform_to_uncoupled(self) -> State:
        F = self.F
        mF = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        electronic_state = self.electronic_state
        P = self.P
        Omega = self.Omega

        mF1s = np.arange(-F1, F1 + 1, 1)
        mJs = np.arange(-J, J + 1, 1)
        m1s = np.arange(-I1, I1 + 1, 1)
        m2s = np.arange(-I2, I2 + 1, 1)

        uncoupled_state = State()

        for mF1 in mF1s:
            for mJ in mJs:
                for m1 in m1s:
                    for m2 in m2s:
                        amp = CGc(J, mJ, I1, m1, F1, mF1) * CGc(F1, mF1, I2, m2, F, mF)
                        basis_state = UncoupledBasisState(
                            J,
                            mJ,
                            I1,
                            m1,
                            I2,
                            m2,
                            P=P,
                            Omega=Omega,
                            electronic_state=electronic_state,
                        )
                        uncoupled_state = uncoupled_state + State([(amp, basis_state)])

        return uncoupled_state.normalize()

    # Method for transforming parity eigenstate to Omega eigenstate basis
    def transform_to_omega_basis(self) -> State:
        F = self.F
        mF = self.mF
        F1 = self.F1
        J = self.J
        I1 = self.I1
        I2 = self.I2
        electronic_state = self.electronic_state
        P = self.P
        Omega = self.Omega

        assert self.basis is not None, "Unknown basis state, can't transform to Ω basis"

        # Check that not already in omega basis
        if self.basis == Basis.CoupledP and self.electronic_state == ElectronicState.B:
            state_minus = 1 * CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=-1 * Omega,
                P=None,
                electronic_state=electronic_state,
                basis=Basis.CoupledΩ,
                v=self.v,
            )
            state_plus = 1 * CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=1 * Omega,
                P=None,
                electronic_state=electronic_state,
                basis=Basis.CoupledΩ,
                v=self.v,
            )

            state = 1 / np.sqrt(2) * (state_plus + P * (-1) ** (J) * state_minus)
        elif (
            self.basis == Basis.CoupledP and self.electronic_state == ElectronicState.X
        ):
            state = 1 * CoupledBasisState(
                F,
                mF,
                F1,
                J,
                I1,
                I2,
                Omega=Omega,
                P=None,
                electronic_state=electronic_state,
                basis=Basis.CoupledΩ,
                v=self.v,
            )

        return state

    # Find energy of state given a list of energies and eigenvectors and basis QN
    def find_energy(
        self,
        energies: npt.NDArray[np.complex128],
        V: npt.NDArray[np.complex128],
        QN: Sequence[BasisState],
    ) -> npt.NDArray[np.complex128]:

        # Convert state to uncoupled basis
        state = self.transform_to_uncoupled()

        # Convert to a vector that can be multiplied by the evecs to determine overlap
        state_vec = np.zeros((1, len(QN)))
        for i, basis_state in enumerate(QN):
            amp = State([(1, basis_state)]) @ state
            state_vec[0, i] = amp

        coeffs = np.multiply(np.dot(state_vec, V), np.conjugate(np.dot(state_vec, V)))
        energy = np.dot(coeffs, energies)

        self.energy = energy
        return energy


# Class for uncoupled basis states
class UncoupledBasisState(BasisState):
    # constructor
    def __init__(
        self,
        J: int,
        mJ: int,
        I1: float,
        m1: float,
        I2: float,
        m2: float,
        Omega: Optional[int] = None,
        P: Optional[int] = None,
        electronic_state: Optional[ElectronicState] = None,
        energy: Optional[float] = None,
        basis: Optional[Basis] = None,
    ):
        self.J, self.mJ = J, mJ
        self.I1, self.m1 = I1, m1
        self.I2, self.m2 = I2, m2
        self.Omega = Omega
        self.P = P
        self.electronic_state = electronic_state
        self.isCoupled = False
        self.isUncoupled = True

        if basis is not None:
            self.basis = basis
        else:
            self.basis = Basis.Uncoupled

        self.energy = energy

    # equality testing
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UncoupledBasisState):
            return False
        else:
            return (
                self.J == other.J
                and self.mJ == other.mJ
                and self.I1 == other.I1
                and self.I2 == other.I2
                and self.m1 == other.m1
                and self.m2 == other.m2
                and self.Omega == other.Omega
                and self.P == other.P
                and self.electronic_state == other.electronic_state
            )

    # inner product
    def __matmul__(self, other: BasisState) -> Union[int, float, complex]:
        if other.isUncoupled:
            if self == other:
                return 1
            else:
                return 0
        elif isinstance(other, CoupledBasisState):
            return State([(1, self)]) @ other.transform_to_uncoupled()
        else:
            raise TypeError(
                "can only multiply UncoupledBasisState with UncoupledBasisState or "
                f"CoupledBasisState (not {type(other)}"
            )

    # superposition: addition
    def __add__(self, other: BasisState) -> State:
        if self == other:
            return State([(2, self)])
        elif isinstance(other, UncoupledBasisState):
            return State([(1, self), (1, other)])
        else:
            raise TypeError(
                f"can only add UncoupledBasisState (not {type(other)}) "
                "to UncoupledBasisState"
            )

    # superposition: subtraction
    def __sub__(self, other: BasisState) -> Union[int, State]:
        if self == other:
            return 0
        elif isinstance(other, UncoupledBasisState):
            return State([(1, self), (-1, other)])
        else:
            raise TypeError(
                f"can only subtract UncoupledBasisState (not {type(other)}) "
                "from UncoupledBasisState"
            )

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]) -> State:
        return State([(a, self)])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> State:
        return self * a

    def __hash__(self):
        quantum_numbers = (
            int(self.J),
            int(self.mJ),
            float(self.I1),
            float(self.m1),
            float(self.I2),
            float(self.m2),
            int(self.P),
            int(self.Omega),
            self.electronic_state.value,
        )
        return int(hashlib.md5(json.dumps(quantum_numbers).encode()).hexdigest(), 16)

    def __repr__(self) -> str:
        return self.state_string()

    def state_string(self) -> str:
        J, mJ = sp.S(str(self.J), rational=True), sp.S(str(self.mJ), rational=True)
        I1 = sp.S(str(self.I1), rational=True)
        m1 = sp.S(str(self.m1), rational=True)
        I2 = sp.S(str(self.I2), rational=True)
        m2 = sp.S(str(self.m2), rational=True)
        if self.P == 1:
            P = "+"
        elif self.P == -1:
            P = "-"
        Omega = self.Omega

        string = f"J = {J}, mJ = {mJ}, I₁ = {I1}, m₁ = {m1}, I₂ = {I2}, m₂ = {m2}"

        if self.electronic_state is not None:
            string = f"{self.electronic_state.name}, {string}"
        if self.P is not None:
            string = f"{string}, P = {P}"
        if Omega is not None:
            string = f"{string}, Ω = {Omega}"
        return "|" + string + ">"

    def print_quantum_numbers(self, printing: bool = False) -> str:
        if printing:
            print(self.state_string())
        return self.state_string()

    # Method for converting to coupled basis
    def transform_to_coupled(self) -> State:
        # Determine quantum numbers
        J = self.J
        mJ = self.mJ
        I1 = self.I1
        m1 = self.m1
        I2 = self.I1
        m2 = self.m2
        Omega = self.Omega
        electronic_state = self.electronic_state
        P = self.P
        Omega = 0 if self.Omega is None else self.Omega

        # Determine what mF has to be
        mF = int(mJ + m1 + m2)

        uncoupled_state = self

        data = []

        # Loop over possible values of F1, F and m_F
        for F1 in np.arange(J - I1, J + I1 + 1):
            for F in np.arange(F1 - I2, F1 + I2 + 1):
                if np.abs(mF) <= F:
                    coupled_state = CoupledBasisState(
                        F,
                        mF,
                        F1,
                        J,
                        I1,
                        I2,
                        Omega=Omega,
                        P=P,
                        electronic_state=electronic_state,
                    )
                    amp = uncoupled_state @ coupled_state
                    data.append((amp, coupled_state))

        return State(data)

    # Method for transforming parity eigenstate to Omega eigenstate basis
    def transform_to_omega_basis(self) -> State:
        # Determine quantum numbers
        J = self.J
        mJ = self.mJ
        I1 = self.I1
        m1 = self.m1
        I2 = self.I1
        m2 = self.m2
        Omega = self.Omega
        electronic_state = self.electronic_state
        P = self.P
        Omega = 0 if self.Omega is None else self.Omega

        # Check that not already in omega basis
        if P is not None and not electronic_state == "X":
            state_minus = 1 * UncoupledBasisState(
                J,
                mJ,
                I1,
                m1,
                I2,
                m2,
                P=P,
                Omega=-1 * Omega,
                electronic_state=electronic_state,
            )
            state_plus = 1 * UncoupledBasisState(
                J,
                mJ,
                I1,
                m1,
                I2,
                m2,
                P=P,
                Omega=Omega,
                electronic_state=electronic_state,
            )

            state = 1 / np.sqrt(2) * (state_plus + P * (-1) ** (J - 1) * state_minus)
        else:
            state = 1 * self

        return state


# Define a class for superposition states
class State:
    # constructor
    def __init__(
        self,
        data: List[Tuple[Union[int, float, complex], Any]] = [],
        remove_zero_amp_cpts: bool = True,
        name: Optional[str] = None,
        energy: float = 0.0,
    ):
        # remove check, takes a lot of time, doesn't seem to get called anyway
        # check for duplicates
        # for i in range(len(data)):
        #     _, cpt1 = data[i][0], data[i][1]
        #     for amp2, cpt2 in data[i + 1 :]:
        #         if cpt1 == cpt2:
        #             raise AssertionError("duplicate components!")
        # remove components with zero amplitudes
        if remove_zero_amp_cpts:
            self.data = [(amp, cpt) for amp, cpt in data if amp != 0]
        else:
            self.data = data
        # for iteration over the State
        self.index = len(self.data)
        # Store energy of state
        self.energy = energy
        # Give the state a name if desired
        self.name = name

    # superposition: addition
    # (highly inefficient and ugly but should work)
    def __add__(self, other: State) -> State:
        data = []
        # add components that are in self but not in other
        for amp1, cpt1 in self.data:
            only_in_self = True
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    only_in_self = False
                    break
            if only_in_self:
                data.append((amp1, cpt1))
        # add components that are in other but not in self
        for amp1, cpt1 in other.data:
            only_in_other = True
            for amp2, cpt2 in self.data:
                if cpt2 == cpt1:
                    only_in_other = False
                    break
            if only_in_other:
                data.append((amp1, cpt1))
        # add components that are both in self and in other
        for amp1, cpt1 in self.data:
            for amp2, cpt2 in other.data:
                if cpt2 == cpt1:
                    data.append((amp1 + amp2, cpt1))
                    break
        return State(data)

    # superposition: subtraction
    def __sub__(self, other: State) -> State:
        return self + -1 * other

    # scalar product (psi * a)
    def __mul__(self, a: Union[float, complex, int]) -> State:
        return State([(a * amp, psi) for amp, psi in self.data])

    # scalar product (a * psi)
    def __rmul__(self, a: Union[float, complex, int]) -> State:
        return self * a

    # scalar division (psi / a)
    def __truediv__(self, a: Union[float, complex, int]) -> State:
        return self * (1 / a)

    # negation
    def __neg__(self) -> State:
        return -1 * self

    # inner product
    def __matmul__(self, other: State) -> Union[int, float, complex]:
        result = 0
        for amp1, psi1 in self:
            for amp2, psi2 in other:
                result += amp1.conjugate() * amp2 * (psi1 @ psi2)
        return result

    # iterator methods
    def __iter__(self):
        return ((amp, state) for amp, state in self.data)

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

    def __eq__(self, other: object) -> bool:
        """
        __eq__ method
        iterates through all amplitudes and states, can be slow

        Args:
            other (object): object to compare to

        Returns:
            bool: True if equal, False if not
        """
        if not isinstance(other, State):
            return False
        else:
            S1 = self.order_by_amp()
            S2 = other.order_by_amp()
            for (a1, s1), (a2, s2) in zip(S1, S2):
                if a1 != a2:
                    return False
                elif s1 != s2:
                    return False
            return True

    # direct access to a component
    def __getitem__(self, i: int) -> Tuple[Union[complex, float, int], BasisState]:
        return self.data[i]

    # this breaks the code, havent figured out why yet, has something to do with the
    # __add__ function I think
    # def __len__(self):
    #     return len(self.data)

    
    def state_string(self, digits: int = 2):
        """
        Returns a string representing the state.

        inputs:
        digits : number of significant digits in amplitudes

        outputs:
        string : string representing self
        """
        ordered = self.order_by_amp()
        idx = 0
        string = ""
        amp_max = np.max(np.abs(list(zip(*ordered))[0]))
        for amp, state in ordered:
            if np.abs(amp) < amp_max * 1e-3:
                continue
            string += f"{amp:.{digits}f} x {state}"
            idx += 1
            if (idx > 5) or (idx == len(ordered.data)):
                break
            string += "\n"
        if idx == 0:
            return ""
        else:
            return string

    def __repr__(self) -> str:
        self.state_string()

    @property
    def largest(self) -> BasisState:
        return self.find_largest_component()

    # Some utility functions
    # Function for normalizing states
    def normalize(self) -> State:
        data = []
        N = np.sqrt(self @ self)
        for amp, basis_state in self.data:
            data.append((amp / N, basis_state))

        return State(data)

    # Function that displays the state as a sum of the basis states
    def print_state(self, tol: float = 0.1, probabilities: bool = False):
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                if probabilities:
                    amp = np.abs(amp) ** 2
                if np.real(complex(amp)) > 0:  # type: ignore
                    print("+", end="")
                string = basis_state.print_quantum_numbers(printing=False)
                string = "{:.4f}".format(complex(amp)) + " x " + string  # type: ignore
                print(string)

    # Function that returns state vector in given basis
    def state_vector(
        self, QN: Union[Sequence[BasisState], Sequence[State]]
    ) -> npt.NDArray[np.complex128]:
        state_vector = [1 * state @ self for state in QN]
        return np.array(state_vector, dtype=complex)

    # Method that generates a density matrix from state
    def density_matrix(
        self, QN: Union[Sequence[BasisState], Sequence[State]]
    ) -> npt.NDArray[np.complex128]:
        # Get state vector
        state_vec = self.state_vector(QN)

        # Generate density matrix from state vector
        density_matrix = np.tensordot(state_vec.conj(), state_vec, axes=0)

        return density_matrix

    # Method that removes components that are smaller than tolerance from the state
    def remove_small_components(self, tol: float = 1e-3) -> State:
        purged_data = []
        for amp, basis_state in self.data:
            if np.abs(amp) > tol:
                purged_data.append((amp, basis_state))

        return State(purged_data, energy=self.energy)

    # Method for ordering states in descending order of amp^2
    def order_by_amp(self) -> State:
        data = self.data
        amp_array = np.zeros(len(data))

        # Make an numpy array of the amplitudes
        for i, d in enumerate(data):
            amp_array[i] = np.abs((data[i][0])) ** 2

        # Find ordering of array in descending order
        index = np.argsort(-1 * amp_array)

        # Reorder data
        reordered_data = data
        reordered_data = [reordered_data[i] for i in index]

        return State(reordered_data)

    # Method for printing largest component basis states
    def print_largest_components(self, n: int = 1) -> str:
        # Order the state by amplitude
        state = self.order_by_amp()

        # Initialize an empty string
        string = ""

        for i in range(0, n):
            basis_state = state.data[i][1]
            basis_state.print_quantum_numbers()

        return string

    def find_largest_component(self) -> BasisState:
        # Order the state by amplitude
        state = self.order_by_amp()

        return state.data[0][1]

    # Method for converting the state into the coupled basis
    def transform_to_coupled(self) -> State:
        # Loop over the basis states, check if they are already in uncoupled
        # basis and if not convert to coupled basis, output state in new basis
        state_in_coupled_basis = State()

        for amp, basis_state in self.data:
            if basis_state.isCoupled:
                state_in_coupled_basis += State([(amp, basis_state)])
            if basis_state.isUncoupled:
                state_in_coupled_basis += amp * basis_state.transform_to_coupled()

        return state_in_coupled_basis

    # Method for converting the state into the uncoupled basis
    def transform_to_uncoupled(self) -> State:
        # Loop over the basis states, check if they are already in uncoupled
        # basis and if not convert to uncoupled basis, output state in new basis
        state_in_uncoupled_basis = State()

        for amp, basis_state in self.data:
            if basis_state.isUncoupled:
                state_in_uncoupled_basis += State([(amp, basis_state)])
            if basis_state.isCoupled:
                state_in_uncoupled_basis += amp * basis_state.transform_to_uncoupled()

        return state_in_uncoupled_basis

    # Method for transforming all basis states to omega basis
    def transform_to_omega_basis(self) -> State:
        state = State()
        for amp, basis_state in self.data:
            state += amp * basis_state.transform_to_omega_basis()

        return state

    # Method for transforming all basis states to parity basis
    def transform_to_parity_basis(self) -> State:
        state = State()
        for amp, basis_state in self.data:
            state += amp * basis_state.transform_to_parity_basis()

        return state

    # Method for obtaining time-reversed version of state (i.e. just reverse all
    # projection quantum numbers)
    def time_reversed(self) -> State:
        new_data: List[Tuple[Union[int, float, complex], BasisState]] = []

        for amp, basis_state in self.data:
            electronic_state = basis_state.electronic_state
            P = basis_state.P
            Omega = basis_state.Omega
            if basis_state.isCoupled:
                F = basis_state.F
                mF = -basis_state.mF
                F1 = basis_state.F1
                J = basis_state.J
                I1 = basis_state.I1
                I2 = basis_state.I2

                new_data.append(
                    (
                        amp,
                        CoupledBasisState(
                            F,
                            mF,
                            F1,
                            J,
                            I1,
                            I2,
                            electronic_state=electronic_state,
                            P=P,
                            Omega=Omega,
                        ),
                    )
                )

            elif basis_state.isUncoupled:
                J = basis_state.J
                mJ = -basis_state.mJ
                I1 = basis_state.I1
                m1 = -basis_state.m1
                I2 = basis_state.I2
                m2 = -basis_state.m2

                new_data.append(
                    (
                        amp,
                        UncoupledBasisState(
                            J,
                            mJ,
                            I1,
                            m1,
                            I2,
                            m2,
                            electronic_state=electronic_state,
                            P=P,
                            Omega=Omega,
                        ),
                    )
                )

        return State(new_data)

    # Method for making the largest coefficient real
    def make_real(self) -> State:
        reordered_state = self.order_by_amp()
        a = reordered_state.data[0][0]
        arg = np.arctan(np.imag(a) / np.real(a))

        return self * np.exp(-1j * arg * np.sign(np.imag(a)))


def BasisStates_from_State(
    states: Union[Sequence[State], npt.NDArray[Any]]
) -> npt.NDArray[Any]:
    unique = []
    for state in states:
        for amp, basisstate in state:
            if basisstate not in unique:
                unique.append(basisstate)
    return np.array(unique)
