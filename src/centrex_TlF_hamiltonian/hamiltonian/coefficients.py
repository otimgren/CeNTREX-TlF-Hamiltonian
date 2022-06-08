from dataclasses import dataclass

__all__ = ["Coefficients", "X", "B"]


@dataclass
class Coefficients:
    B_rot: float


# Values for rotational constant are from "Microwave Spectral tables:
# Diatomic molecules" by Lovas & Tiemann (1974).
# Note that Brot differs from the one given by Ramsey by about 30 MHz.

B_ϵ = 6.689873e9
α = 45.0843e6


@dataclass(unsafe_hash=True)
class X(Coefficients):
    B_rot: float = B_ϵ - α / 2
    c1: float = 126030.0
    c2: float = 17890.0
    c3: float = 700.0
    c4: float = -13300.0
    μ_J: float = 35.0  # Hz/G
    μ_Tl: float = 1240.5  # Hz/G
    μ_F: float = 2003.63  # Hz/G
    D_TlF: float = 4.2282 * 0.393430307 * 5.291772e-9 / 4.135667e-15  # Hz/(V/cm)


@dataclass(unsafe_hash=True)
class B(Coefficients):
    # Constants in MHz
    B_rot: float = 6687.879e6
    D_rot: float = 0.010869e6
    H_const: float = -8.1e-2
    h1_Tl: float = 28789e6
    h1_F: float = 861e6
    q: float = 2.423e6
    c_Tl: float = -7.83e6
    c1p_Tl: float = 11.17e6
    μ_B: float = 100
    gL: float = 1
    gS: float = 2
