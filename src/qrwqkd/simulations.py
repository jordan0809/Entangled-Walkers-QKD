from geqo.core import Sequence
from geqo.gates import PauliX, Ry, Rz, CNOT, Hadamard
from geqo.operations import QuantumControl, Measure
from geqo.initialization import SetQubits
from geqo.simulators import ensembleSimulatorCuPy
from geqo.utils import bin2num
import numpy as np
import math


class EntangledWalkersQKD:
    def __init__(self, steps: int, coin_params: dict = {}):
        self.steps = steps
        self.a = coin_params.get("a", 0.85)
        self.phi = coin_params.get("phi", np.pi / 4)
        self.theta = coin_params.get("theta", 0.635)
        self.lam = coin_params.get("lam", 0)

    def __str__(self):
        return (
            f"{self.steps}-step entangled walkers QKD protocol\n"
            f"coin parameters:\n"
            f"-----------\n"
            f"Initial coin state: {self.a}|00> + {(1 - self.a**2) ** 0.5 * np.exp(1j * self.phi)}|11>\n"
            f"Coin-flip operator U(θ,λ) = U({self.theta}, {self.lam}) = \n {self.coin_flip_operator()}"
        )

    def coin_flip_operator(self):
        return np.array(
            [
                [
                    np.cos(self.theta / 2),
                    -np.exp(1j * self.lam) * np.sin(self.theta / 2),
                ],
                [
                    np.sin(self.theta / 2),
                    np.exp(1j * self.lam) * np.cos(self.theta / 2),
                ],
            ]
        )

    def quantum_circuit(self):
        steps = self.steps
        a = self.a
        phi = self.phi
        theta = self.theta
        lam = self.lam

        # Store coin parameters for simulation
        params = {"a": 2 * np.arccos(a), "φ": phi, "θ": theta, "λ": lam}

        # Number of qubits needed to represent the position state
        num_pos = math.floor(math.log2(2 * steps)) + 1

        # Initialize position and coin quantum registers
        pos_A = [f"x_{i}" for i in range(num_pos)]
        pos_B = [f"y_{i}" for i in range(num_pos)]
        coin_A = ["coin_A"]
        coin_B = ["coin_B"]
        qubits = pos_A + coin_A + coin_B + pos_B

        # Initialize classical registers for coin and position measurements
        preg_A = [f"pA_{i}" for i in range(num_pos)]  #  Alice's position measurement
        preg_B = [f"pB_{i}" for i in range(num_pos)]  # Bob's position measurement
        creg_A = ["cA"]  # Alice's coin measurement
        creg_B = ["cB"]  # Bob's coin measurement
        cbits = creg_A + creg_B + preg_A + preg_B

        op = []
        # Initialize position states = 2**(num_pos-1)
        op.extend(
            [
                (SetQubits("init", num_pos), pos_A, []),
                (SetQubits("init", num_pos), pos_B, []),
            ]
        )
        params.update({"init": [1] + [0] * (num_pos - 1)})

        # Initialize entangled coins a|00> + (1-a**2)**0.5 * e**(1j* phi)|11>
        op.extend(
            [
                (Ry("a"), coin_A, []),
                (Rz("φ"), coin_A, []),
                (CNOT(), coin_A + coin_B, []),
            ]
        )

        # Implement quantum random walk
        for _ in range(steps):
            ## Alice's walk
            # Apply coin-flip operator U(θ,λ)
            op.extend([(Rz("λ"), coin_A, []), (Ry("θ"), coin_A, [])])

            # Addition (position + 1)
            for i in range(num_pos):
                targets = coin_A + pos_A[::-1][:-i] if i != 0 else coin_A + pos_A[::-1]
                op.append((QuantumControl([1] * (num_pos - i), PauliX()), targets, []))

            # Subtraction (position - 1)
            for i in range(num_pos):
                op.append(
                    (
                        QuantumControl([0] + [1] * i, PauliX()),
                        coin_A + pos_A[::-1][: i + 1],
                        [],
                    )
                )

            ## Bob's walk
            op.extend([(Rz("λ"), coin_B, []), (Ry("θ"), coin_B, [])])
            for i in range(num_pos):
                targets = coin_B + pos_B[::-1][:-i] if i != 0 else coin_B + pos_B[::-1]
                op.append((QuantumControl([1] * (num_pos - i), PauliX()), targets, []))
            for i in range(num_pos):
                op.append(
                    (
                        QuantumControl([0] + [1] * i, PauliX()),
                        coin_B + pos_B[::-1][: i + 1],
                        [],
                    )
                )

        # Bell state measurement (BSM) on entangled coins
        op.extend(
            [
                (CNOT(), coin_A + coin_B, []),
                (Hadamard(), coin_A, []),
                (Measure(2), coin_A + coin_B, creg_A + creg_B),
            ]
        )

        # Position measurement
        op.append((Measure(num_pos), pos_A, preg_A))
        op.append((Measure(num_pos), pos_B, preg_B))

        qrwqkd = Sequence(qubits, cbits, op)

        return qrwqkd, params

    def evaluate_circuit(
        self, circuit: Sequence, params: dict, normalize: bool = False
    ):
        num_pos = math.floor(math.log2(2 * self.steps)) + 1
        sim = ensembleSimulatorCuPy(2 * num_pos + 2, 2 * num_pos + 2)
        sim.values = params
        sim.apply(circuit, [*range(2 * num_pos + 2)], [*range(2 * num_pos + 2)])

        bell00 = {}
        bell01 = {}
        bell10 = {}
        bell11 = {}
        probs = [0, 0, 0, 0]
        bound = 2 + num_pos  # index of Bob's first position qubit
        x0 = 2 ** (num_pos - 1)  # initial position state
        for key, item in sim.ensemble.items():
            key = list(key)
            if key[0] == 0 and key[1] == 0:
                bell00[(bin2num(key[2:bound]) - x0, bin2num(key[bound:]) - x0)] = item[
                    0
                ]
                probs[0] += item[0]
            if key[0] == 0 and key[1] == 1:
                bell01[(bin2num(key[2:bound]) - x0, bin2num(key[bound:]) - x0)] = item[
                    0
                ]
                probs[1] += item[0]
            if key[0] == 1 and key[1] == 0:
                bell10[(bin2num(key[2:bound]) - x0, bin2num(key[bound:]) - x0)] = item[
                    0
                ]
                probs[2] += item[0]
            if key[0] == 1 and key[1] == 1:
                bell11[(bin2num(key[2:bound]) - x0, bin2num(key[bound:]) - x0)] = item[
                    0
                ]
                probs[3] += item[0]

        ensemble = {
            "00": (probs[0], bell00),
            "01": (probs[1], bell01),
            "10": (probs[2], bell10),
            "11": (probs[3], bell11),
        }

        if normalize:
            for key, item in ensemble.items():
                total = np.sum([v for v in item[1].values()])
                norm_dist = {pos: value / total for pos, value in item[1].items()}
                ensemble[key] = (item[0], norm_dist)

        return ensemble
