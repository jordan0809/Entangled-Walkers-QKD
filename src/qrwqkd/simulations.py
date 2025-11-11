from geqo.core import Sequence
from geqo.gates import PauliX, Ry, Rz, CNOT, Hadamard
from geqo.operations import QuantumControl, Measure
from geqo.initialization import SetQubits
from geqo.simulators import ensembleSimulatorCuPy
from geqo.utils import bin2num
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


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

    def recursive_simulate(self, normalize: bool = False):
        """
        We adopt the convention that the row indices of the amplitude tensor (from top to bottom)
        correspond to positions ranging from -steps to +steps, while the column indices (from left to right)
        correspond to the same range of positions.
        """
        steps = self.steps
        a = self.a
        phi = self.phi
        theta = self.theta
        lam = self.lam

        def shift_matrices(n, dx, dy):
            """dx = 1 shift 1 step to the left (position -1). dy = 1 shift 1 step upward (position -1)"""
            R = np.zeros((n, n), dtype=np.complex128)  # top-down shift
            C = np.zeros((n, n), dtype=np.complex128)  # left-right shift
            if dy >= 0:
                R[: -dy or None, dy:] = np.eye(n - dy)
            else:
                R[-dy:, : dy or None] = np.eye(n + dy)
            if dx >= 0:
                C[dx:, : -dx or None] = np.eye(n - dx)
            else:
                C[: dx or None, -dx:] = np.eye(n + dx)
            return R, C

        b = (1 - a**2) ** 0.5 * np.exp(1j * phi)
        u = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [np.sin(theta / 2), np.exp(1j * lam) * np.cos(theta / 2)],
            ]
        )
        U = np.kron(u, u)  # shape (4, 4)

        # Initialize amplitude tensors
        n = 2 * steps + 1
        A00 = np.zeros((n, n), dtype=np.complex128)
        A11 = np.zeros((n, n), dtype=np.complex128)
        A00[steps, steps] = a
        A11[steps, steps] = b

        # Wrap all tensors in a column vector
        A_vec = np.zeros((4, n, n), dtype=np.complex128)
        A_vec[0] = A00
        A_vec[3] = A11

        # Recursive walks
        for _ in range(steps):
            # shifts correspond to coin states |00>, |01>, |10>, |11>
            shifts = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

            # Get shift matrices
            Rs_Cs = [shift_matrices(n, dx, dy) for dx, dy in shifts]
            Rs = np.stack(
                [R for R, _ in Rs_Cs]
            )  # shape (4, n, n)  # four different types of top-down shift
            Cs = np.stack(
                [C for _, C in Rs_Cs]
            )  # shape (4, n, n)  # four different types of left-right shift

            # Apply U
            UA_vec = np.einsum(
                "ij,jkl->ikl", U, A_vec
            )  # shape (4, n, n) each row represents a linear combination of amplitude tensors from different coin states
            # Apply shift matrices
            A_vec = np.einsum(
                "ijk,ikl,ilm->ijm", Rs, UA_vec, Cs
            )  # shape (4, n, n) shift the amplitude tensor of each coin state

        # Bell state measurement (BSM)
        bell_tensor = np.zeros((4, n, n))
        bell_tensor[0] = (1 / 2) * np.abs(A_vec[0] + A_vec[3]) ** 2
        bell_tensor[1] = (1 / 2) * np.abs(A_vec[1] + A_vec[2]) ** 2
        bell_tensor[2] = (1 / 2) * np.abs(A_vec[0] - A_vec[3]) ** 2
        bell_tensor[3] = (1 / 2) * np.abs(A_vec[1] - A_vec[2]) ** 2
        # take only -s, -s+2, .., s-2, s as indices
        bsm_probs = [
            bell_tensor[i][
                np.ix_(np.arange(0, 2 * steps + 1, 2), np.arange(0, 2 * steps + 1, 2))
            ]
            for i in range(4)
        ]

        ensemble = {
            f"{bin(i)[2:].zfill(2)}": (np.sum(bsm_probs[i]), bsm_probs[i])
            for i in range(4)
        }
        if normalize:
            ensemble = {
                key: (item[0], item[1] / item[0]) for key, item in ensemble.items()
            }

        # aggregate probability distributions over all BSM outcomes
        prob = np.array([bell_tensor[i] for i in range(4)])
        sum_prob = np.einsum("ijk->jk", prob)
        full_prob = sum_prob[
            np.ix_(np.arange(0, 2 * steps + 1, 2), np.arange(0, 2 * steps + 1, 2))
        ]

        return ensemble, full_prob

    def distribute_key(self, nrounds: int = 10000):
        """Simulate multiple rounds of key distribution."""
        bsm_outcomes, all_probs = self.recursive_simulate(normalize=True)

        bsm_probs = [v[0] for v in bsm_outcomes.values()]
        bsm_dist = [v[1] for v in bsm_outcomes.values()]
        steps = self.steps

        success = []  # track whether each round establishes a shared key or not
        sifted_keys = []  # store shared keys
        bell_samples = random.choices(
            [0, 1, 2, 3], weights=bsm_probs, k=nrounds
        )  # sample from the probability distribution of BSM
        pos_samples_count = np.zeros(
            (steps + 1, steps + 1)
        )  # store joint distribution counts of measured positons

        for b in bell_samples:
            joint_probs = bsm_dist[b].flatten()
            pairs = [
                (i, j)
                for i in range(-steps, steps + 1, 2)
                for j in range(-steps, steps + 1, 2)
            ]
            pos_sample = random.choices(pairs, weights=joint_probs, k=1)[
                0
            ]  # sample Alice and Bob's position measurement

            alice_bit = pos_sample[0]
            bob_bit = pos_sample[1]
            pos_samples_count[
                int((alice_bit + steps) / 2), int((bob_bit + steps) / 2)
            ] += 1

            if b % 2 == 0:  # coin state = |00> or |10>
                if bob_bit == steps:
                    if alice_bit == steps:
                        success.append(1)
                        sifted_keys.append(steps)
                    elif alice_bit not in [
                        -steps,
                        steps,
                    ]:  # no shared key (used for security check)
                        success.append(0)
                    else:  # alice bit = -steps (invalid)
                        print(
                            f"Key sharing failed. Impossible scenario encountered. Alice: {alice_bit}, Bob : {bob_bit}, Bell: {b}"
                        )
                        break
                elif bob_bit not in [-steps, steps]:  # intermediate positions
                    success.append(0)
                else:  # bob_bit = -steps
                    if alice_bit == -steps:
                        success.append(1)
                        sifted_keys.append(-steps)
                    elif alice_bit not in [-steps, steps]:
                        success.append(0)
                    else:
                        print(
                            f"Key sharing failed. Impossible scenario encountered. Alice: {alice_bit}, Bob : {bob_bit}, Bell: {b}"
                        )
                        break
            else:  # coin state = |01> or |11>
                if bob_bit == steps:
                    if alice_bit == -steps:
                        success.append(1)
                        sifted_keys.append(steps)
                    elif alice_bit not in [-steps, steps]:
                        success.append(0)
                    else:
                        print(
                            f"Key sharing failed. Impossible scenario encountered. Alice: {alice_bit}, Bob : {bob_bit}, Bell: {b}"
                        )
                        break
                elif bob_bit not in [-steps, steps]:
                    success.append(0)
                else:  # bob_bit = -steps
                    if alice_bit == steps:
                        success.append(1)
                        sifted_keys.append(-steps)
                    elif alice_bit not in [-steps, steps]:
                        success.append(0)
                    else:
                        print(
                            f"Key sharing failed. Impossible scenario encountered. Alice: {alice_bit}, Bob : {bob_bit}, Bell: {b}"
                        )
                        break

        # Calculate Shannon entropy of sifted keys
        p = np.sum([1 for k in sifted_keys if k == 2]) / len(sifted_keys)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        print("Shared key rate:", np.sum(success) / nrounds)
        print(f"First 100 sifted keys:{sifted_keys[:100]}")
        print(f"Shannon entropy of sifted keys: {entropy}")

        # Calculate probability distributions for security check
        pos_samples_prob = pos_samples_count / np.sum(pos_samples_count)
        bsm_samples_prob = np.array([bell_samples.count(i) for i in range(4)])
        bsm_samples_prob = bsm_samples_prob / np.sum(bsm_samples_prob)

        # Compute total variation distance (TVD)
        pos_tvd = np.sum(np.abs(pos_samples_prob - all_probs)) / 2
        bsm_tvd = np.sum(np.abs(bsm_samples_prob - np.array(bsm_probs))) / 2
        print("-" * 50)
        print("Security Check \n" + "-" * 50)
        print("Total variation distance (TVD) for position measurements:", pos_tvd)
        print("Total variation distance (TVD) for BSM:", bsm_tvd)

        return sifted_keys


def plot_heatmap(data: list[list[float]], **kwargs):
    if data.shape[0] != data.shape[1]:
        raise ValueError("Input distribution has unequal numbers of rows and columns.")

    plt.rcParams["font.size"] = kwargs.get("fontsize", 12)
    fig, ax = plt.subplots()

    custom_colors = ["royalblue", "steelblue", "dodgerblue", "lightseagreen"]
    colors = kwargs.get("cmap", custom_colors)
    colormap = LinearSegmentedColormap.from_list("cmap", colors)

    im = ax.imshow(data, cmap=colormap, norm=LogNorm(vmin=0.001, vmax=0.6))

    steps = len(data) - 1

    # Add grid lines
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    x_labels = np.arange(-steps, steps + 1, 2)
    y_labels = np.arange(-steps, steps + 1, 2)

    # Set ticks at integer positions
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    # Set custom labels
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2e}", ha="center", va="center", color="white")

    ax.set_xlabel("Bob")
    ax.set_ylabel("Alice")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
