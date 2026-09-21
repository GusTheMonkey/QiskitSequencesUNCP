from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import math
from scipy.special import gammaincc


# ============================================================
# SETTINGS
# ============================================================

num_qubits = 1
shots = 1000

gate = "H"

# Used by parameterized gates such as RX, RY, RZ, and P
angle = math.pi / 2

# Statistical significance level
ALPHA = 0.01


# ============================================================
# CREATE QUANTUM CIRCUIT
# ============================================================

qc = QuantumCircuit(num_qubits)

if gate == "H":
    qc.h(0)

elif gate == "X":
    qc.x(0)

elif gate == "Y":
    qc.y(0)

elif gate == "Z":
    qc.z(0)

elif gate == "RX":
    qc.rx(angle, 0)

elif gate == "RY":
    qc.ry(angle, 0)

elif gate == "RZ":
    qc.rz(angle, 0)

elif gate == "P":
    qc.p(angle, 0)

else:
    print("Unknown gate:", gate)
    exit()


# ============================================================
# ADD MEASUREMENT
# ============================================================

qc.measure_all()


# ============================================================
# RUN QRNG
# ============================================================

simulator = AerSimulator()

job = simulator.run(
    qc,
    shots=shots,
    memory=True
)

result = job.result()


# ============================================================
# GET RANDOM BITS
# ============================================================

bits = result.get_memory()

random_string = ''.join(bits)


# ============================================================
# BASIC COUNTS
# ============================================================

zeros = random_string.count("0")
ones = random_string.count("1")
total = len(random_string)

zero_probability = zeros / total
one_probability = ones / total

zero_percentage = zero_probability * 100
one_percentage = one_probability * 100


# ============================================================
# SHANNON ENTROPY
# ============================================================

def shannon_entropy(p0, p1):

    entropy = 0

    if p0 > 0:
        entropy -= p0 * math.log2(p0)

    if p1 > 0:
        entropy -= p1 * math.log2(p1)

    return entropy


shannon = shannon_entropy(
    zero_probability,
    one_probability
)


# ============================================================
# MIN-ENTROPY
# ============================================================

maximum_probability = max(
    zero_probability,
    one_probability
)

min_entropy = -math.log2(maximum_probability)


# ============================================================
# RUN ANALYSIS
# ============================================================

def count_runs(bit_string):

    if len(bit_string) == 0:
        return 0

    runs = 1

    for i in range(1, len(bit_string)):

        if bit_string[i] != bit_string[i - 1]:
            runs += 1

    return runs


runs = count_runs(random_string)


# ============================================================
# LONGEST RUN
# ============================================================

def longest_run(bit_string):

    if len(bit_string) == 0:
        return 0

    longest = 1
    current = 1

    for i in range(1, len(bit_string)):

        if bit_string[i] == bit_string[i - 1]:

            current += 1

            if current > longest:
                longest = current

        else:

            current = 1

    return longest


longest = longest_run(random_string)


# ============================================================
# TWO-BIT PATTERNS
# ============================================================

pattern_counts = {
    "00": 0,
    "01": 0,
    "10": 0,
    "11": 0
}

for i in range(len(random_string) - 1):

    pattern = random_string[i:i + 2]

    pattern_counts[pattern] += 1


pattern_total = sum(pattern_counts.values())


# ============================================================
# AUTOCORRELATION
# ============================================================

def autocorrelation(bit_string, lag):

    if len(bit_string) <= lag:
        return 0

    matches = 0
    comparisons = len(bit_string) - lag

    for i in range(comparisons):

        if bit_string[i] == bit_string[i + lag]:
            matches += 1

    return matches / comparisons


autocorrelation_1 = autocorrelation(random_string, 1)
autocorrelation_2 = autocorrelation(random_string, 2)
autocorrelation_3 = autocorrelation(random_string, 3)


# ============================================================
# NIST FREQUENCY TEST
# ============================================================

def frequency_test(bit_string):

    n = len(bit_string)

    if n == 0:
        return 0

    # Convert 0 -> -1 and 1 -> +1
    s = 0

    for bit in bit_string:

        if bit == "1":
            s += 1
        else:
            s -= 1

    test_statistic = abs(s) / math.sqrt(n)

    p_value = math.erfc(
        test_statistic / math.sqrt(2)
    )

    return p_value


frequency_p = frequency_test(random_string)


# ============================================================
# NIST RUNS TEST
# ============================================================

def runs_test(bit_string):

    n = len(bit_string)

    if n < 2:
        return 0

    pi = bit_string.count("1") / n

    # Runs test requires approximately balanced bits
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0

    runs = count_runs(bit_string)

    numerator = abs(
        runs -
        (2 * n * pi * (1 - pi))
    )

    denominator = (
        2 *
        math.sqrt(2 * n) *
        pi *
        (1 - pi)
    )

    if denominator == 0:
        return 0

    p_value = math.erfc(
        numerator / denominator
    )

    return p_value


runs_p = runs_test(random_string)


# ============================================================
# BLOCK FREQUENCY TEST
# ============================================================

def block_frequency_test(bit_string, block_size=10):

    n = len(bit_string)

    number_of_blocks = n // block_size

    if number_of_blocks == 0:
        return 0

    chi_square = 0

    for i in range(number_of_blocks):

        block = bit_string[
            i * block_size:
            (i + 1) * block_size
        ]

        ones_in_block = block.count("1")

        proportion = ones_in_block / block_size

        chi_square += (
            4 *
            block_size *
            (proportion - 0.5) ** 2
        )

    p_value = gammaincc(
        number_of_blocks / 2,
        chi_square / 2
    )

    return p_value


block_frequency_p = block_frequency_test(
    random_string,
    block_size=10
)


# ============================================================
# APPROXIMATE ENTROPY TEST
# ============================================================

def approximate_entropy_test(bit_string, m=2):

    n = len(bit_string)

    if n < 100:
        return None

    def phi(block_size):

        extended = bit_string + bit_string[:block_size]

        counts = {}

        for i in range(n):

            block = extended[
                i:i + block_size
            ]

            counts[block] = counts.get(block, 0) + 1

        result = 0

        for count in counts.values():

            probability = count / n

            result += probability * math.log(
                probability
            )

        return result

    phi_m = phi(m)
    phi_m_plus_1 = phi(m + 1)

    approximate_entropy = phi_m - phi_m_plus_1

    chi_square = (
        2 *
        n *
        (
            math.log(2)
            - approximate_entropy
        )
    )

    degrees_of_freedom = 2 ** (m - 1)

    p_value = gammaincc(
        degrees_of_freedom / 2,
        chi_square / 2
    )

    return p_value


approximate_entropy_p = approximate_entropy_test(
    random_string
)


# ============================================================
# CUMULATIVE SUMS TEST
# ============================================================

def cumulative_sums_test(bit_string):

    n = len(bit_string)

    if n == 0:
        return 0

    # Convert bits to +1 and -1
    values = []

    for bit in bit_string:

        if bit == "1":
            values.append(1)
        else:
            values.append(-1)

    # Calculate cumulative sums
    cumulative = 0
    maximum = 0

    for value in values:

        cumulative += value

        maximum = max(
            maximum,
            abs(cumulative)
        )

    # Approximation of the NIST cumulative sums
    if maximum == 0:
        return 1.0

    z = maximum / math.sqrt(n)

    p_value = math.erfc(
        z / math.sqrt(2)
    )

    return p_value


cumulative_sums_p = cumulative_sums_test(
    random_string
)


# ============================================================
# PASS / FAIL HELPER
# ============================================================

def pass_fail(p_value):

    if p_value is None:
        return "N/A"

    if p_value >= ALPHA:
        return "PASS"

    return "FAIL"


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n================================================")
print("                 QRNG RESULTS")
print("================================================")

print("\nQuantum Circuit:")
print(qc)


print("\n------------------------------------------------")
print("RANDOM BIT STREAM")
print("------------------------------------------------")

print("Total bits:", total)

print("\nRandom bit string:")

# Print
print(random_string)


print("\n------------------------------------------------")
print("FREQUENCY")
print("------------------------------------------------")

print("Zeros:", zeros)
print("Ones:", ones)

print(f"0 frequency: {zero_percentage:.2f}%")
print(f"1 frequency: {one_percentage:.2f}%")


print("\n------------------------------------------------")
print("ENTROPY")
print("------------------------------------------------")

print(f"Shannon entropy: {shannon:.6f} bits/bit")
print(f"Min-entropy:     {min_entropy:.6f} bits/bit")


print("\n------------------------------------------------")
print("RUN ANALYSIS")
print("------------------------------------------------")

print("Number of runs:", runs)
print("Longest run:", longest)


print("\n------------------------------------------------")
print("TWO-BIT PATTERNS")
print("------------------------------------------------")

for pattern in ["00", "01", "10", "11"]:

    count = pattern_counts[pattern]

    percentage = (
        count /
        pattern_total *
        100
    )

    print(
        f"{pattern}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


print("\n------------------------------------------------")
print("AUTOCORRELATION")
print("------------------------------------------------")

print(f"Lag 1: {autocorrelation_1:.4f}")
print(f"Lag 2: {autocorrelation_2:.4f}")
print(f"Lag 3: {autocorrelation_3:.4f}")


print("\n------------------------------------------------")
print("STATISTICAL TESTS")
print("------------------------------------------------")

print(f"Significance level: α = {ALPHA}")

print(
    f"\nFrequency test: "
    f"p = {frequency_p:.6f} "
    f"-> {pass_fail(frequency_p)}"
)

print(
    f"Runs test: "
    f"p = {runs_p:.6f} "
    f"-> {pass_fail(runs_p)}"
)

print(
    f"Block frequency test: "
    f"p = {block_frequency_p:.6f} "
    f"-> {pass_fail(block_frequency_p)}"
)

if approximate_entropy_p is not None:

    print(
        f"Approximate entropy test: "
        f"p = {approximate_entropy_p:.6f} "
        f"-> {pass_fail(approximate_entropy_p)}"
    )

else:

    print(
        "Approximate entropy test: "
        "N/A"
    )

print(
    f"Cumulative sums test: "
    f"p = {cumulative_sums_p:.6f} "
    f"-> {pass_fail(cumulative_sums_p)}"
)


print("\n================================================")
print("                 END ANALYSIS")
print("================================================")
