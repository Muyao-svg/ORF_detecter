# constants_frame.py

import numpy as np

DATA_DIR= "/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Data/Benchmark"
OUTPUT_DIR = "/stockage/plateformes/I2BC/CBNG_Muyao-thesis/Ribokast_hmm/Segmentation_signal/Outputs/Benchmark"

#GENE_NAME = "RPS20"
OFFSET = 12
LAMBDA_CANONICAL_START = 200
LAMBDA_NON_CANONICAL_START = 100
LAMBDA_NO_START = 500
LAMBDA_STOP = 100
LAMBDA_NO_STOP = 500

BETA0 = -7.186061
BETA1 = 4.105736
BETA2 = 7.199081

MAX_WORKERS = 4

# =========================================================
# Unified-frame HMM with explicit Start/C/Stop states
#
# States:
#   B
#   Start0, C0, Stop0
#   Start1, C1, Stop1
#   Start2, C2, Stop2
#
# Notes:
# - global frame identity is encoded in state names
# - frame_run is NOT a state; it will be used in emission/transition
# - Start/Stop gating by frame_run will be handled later outside this file
# =========================================================

STATE_NAMES = [
    "B",
    "Start0", "C0", "Stop0",
    "Start1", "C1", "Stop1",
    "Start2", "C2", "Stop2",
]

STATE_TO_INDEX = {s: i for i, s in enumerate(STATE_NAMES)}

BACKGROUND_STATE = "B"
START_STATES = ["Start0", "Start1", "Start2"]
CODING_STATES = ["C0", "C1", "C2"]
STOP_STATES = ["Stop0", "Stop1", "Stop2"]

FRAME_OF_STATE = {
    "Start0": 0, "C0": 0, "Stop0": 0,
    "Start1": 1, "C1": 1, "Stop1": 1,
    "Start2": 2, "C2": 2, "Stop2": 2,
}

# =========================================================
# Initial NB parameters
#
# Keep coding states sharing the same initial scale.
# The frame difference is expressed mainly by multinomial emission
# after rotation with frame_run.
# =========================================================

MU_B_INIT = 20.0
R_B_INIT = 20.0

MU_START_INIT = 120.0
R_START_INIT = 20.0

MU_C_INIT = 120.0
R_C_INIT = 20.0

MU_STOP_INIT = 80.0
R_STOP_INIT = 20.0

# =========================================================
# Initial multinomial templates
#
# Background:
#   near uniform
#
# Coding base template:
#   x1-dominant in canonical local coordinate
#
# Start / Stop can initially share the same base template.
# Later, if needed, you can let Start / C / Stop use different pi_base.
# =========================================================

RHO_INIT = 0.80
OTHER_INIT = (1.0 - RHO_INIT) / 2.0

PI_B_INIT = np.array([1/3, 1/3, 1/3], dtype=float)
PI_C_BASE_INIT = np.array([RHO_INIT, OTHER_INIT, OTHER_INIT], dtype=float)
PI_START_BASE_INIT = PI_C_BASE_INIT.copy()
PI_STOP_BASE_INIT = PI_C_BASE_INIT.copy()

# =========================================================
# State parameter dictionary
#
# For coding-like states:
#   type        : "start" / "coding" / "stop"
#   global_frame: 0 / 1 / 2
#   pi_base     : canonical local template before rotation
#
# For B:
#   type = "background"
# =========================================================

STATE_PARAMS = {
    "B": {
        "mu": MU_B_INIT,
        "r": R_B_INIT,
        "pi": PI_B_INIT.copy(),
        "type": "background",
    },

    "Start0": {
        "mu": MU_START_INIT,
        "r": R_START_INIT,
        "pi_base": PI_START_BASE_INIT.copy(),
        "global_frame": 0,
        "type": "start",
    },
    "C0": {
        "mu": MU_C_INIT,
        "r": R_C_INIT,
        "pi_base": PI_C_BASE_INIT.copy(),
        "global_frame": 0,
        "type": "coding",
    },
    "Stop0": {
        "mu": MU_STOP_INIT,
        "r": R_STOP_INIT,
        "pi_base": PI_STOP_BASE_INIT.copy(),
        "global_frame": 0,
        "type": "stop",
    },

    "Start1": {
        "mu": MU_START_INIT,
        "r": R_START_INIT,
        "pi_base": PI_START_BASE_INIT.copy(),
        "global_frame": 1,
        "type": "start",
    },
    "C1": {
        "mu": MU_C_INIT,
        "r": R_C_INIT,
        "pi_base": PI_C_BASE_INIT.copy(),
        "global_frame": 1,
        "type": "coding",
    },
    "Stop1": {
        "mu": MU_STOP_INIT,
        "r": R_STOP_INIT,
        "pi_base": PI_STOP_BASE_INIT.copy(),
        "global_frame": 1,
        "type": "stop",
    },

    "Start2": {
        "mu": MU_START_INIT,
        "r": R_START_INIT,
        "pi_base": PI_START_BASE_INIT.copy(),
        "global_frame": 2,
        "type": "start",
    },
    "C2": {
        "mu": MU_C_INIT,
        "r": R_C_INIT,
        "pi_base": PI_C_BASE_INIT.copy(),
        "global_frame": 2,
        "type": "coding",
    },
    "Stop2": {
        "mu": MU_STOP_INIT,
        "r": R_STOP_INIT,
        "pi_base": PI_STOP_BASE_INIT.copy(),
        "global_frame": 2,
        "type": "stop",
    },
}

# =========================================================
# Static HMM transition skeleton
#
# B -> B or Start0/1/2
# Startg -> Cg
# Cg -> Cg or Stopg
# Stopg -> B
#
# Start gating by p_start and frame_run will be added later
# in transition_frame.py / transition logic, not here.
# =========================================================

S = len(STATE_NAMES)
A_INIT_HMM = np.zeros((S, S), dtype=float)

iB = STATE_TO_INDEX["B"]

# B row
A_INIT_HMM[iB, iB] = 0.97
A_INIT_HMM[iB, STATE_TO_INDEX["Start0"]] = 0.01
A_INIT_HMM[iB, STATE_TO_INDEX["Start1"]] = 0.01
A_INIT_HMM[iB, STATE_TO_INDEX["Start2"]] = 0.01

# Frame 0 chain
A_INIT_HMM[STATE_TO_INDEX["Start0"], STATE_TO_INDEX["C0"]] = 1.00
A_INIT_HMM[STATE_TO_INDEX["C0"], STATE_TO_INDEX["C0"]] = 0.95
A_INIT_HMM[STATE_TO_INDEX["C0"], STATE_TO_INDEX["Stop0"]] = 0.05
A_INIT_HMM[STATE_TO_INDEX["Stop0"], iB] = 1.00

# Frame 1 chain
A_INIT_HMM[STATE_TO_INDEX["Start1"], STATE_TO_INDEX["C1"]] = 1.00
A_INIT_HMM[STATE_TO_INDEX["C1"], STATE_TO_INDEX["C1"]] = 0.95
A_INIT_HMM[STATE_TO_INDEX["C1"], STATE_TO_INDEX["Stop1"]] = 0.05
A_INIT_HMM[STATE_TO_INDEX["Stop1"], iB] = 1.00

# Frame 2 chain
A_INIT_HMM[STATE_TO_INDEX["Start2"], STATE_TO_INDEX["C2"]] = 1.00
A_INIT_HMM[STATE_TO_INDEX["C2"], STATE_TO_INDEX["C2"]] = 0.95
A_INIT_HMM[STATE_TO_INDEX["C2"], STATE_TO_INDEX["Stop2"]] = 0.05
A_INIT_HMM[STATE_TO_INDEX["Stop2"], iB] = 1.00

#ALLOWED_MASK_HMM = (A_INIT_HMM > 0).astype(float)
ALLOWED_MASK = np.zeros((S, S), dtype=float)

idx = {s:i for i, s in enumerate(STATE_NAMES)}
# B can stay in B or open any Start frame
ALLOWED_MASK[idx["B"], idx["B"]] = 1
ALLOWED_MASK[idx["B"], idx["Start0"]] = 1
ALLOWED_MASK[idx["B"], idx["Start1"]] = 1
ALLOWED_MASK[idx["B"], idx["Start2"]] = 1

# frame 0
ALLOWED_MASK[idx["Start0"], idx["C0"]] = 1
ALLOWED_MASK[idx["C0"], idx["C0"]] = 1
ALLOWED_MASK[idx["C0"], idx["Stop0"]] = 1
ALLOWED_MASK[idx["Stop0"], idx["B"]] = 1

# frame 1
ALLOWED_MASK[idx["Start1"], idx["C1"]] = 1
ALLOWED_MASK[idx["C1"], idx["C1"]] = 1
ALLOWED_MASK[idx["C1"], idx["Stop1"]] = 1
ALLOWED_MASK[idx["Stop1"], idx["B"]] = 1

# frame 2
ALLOWED_MASK[idx["Start2"], idx["C2"]] = 1
ALLOWED_MASK[idx["C2"], idx["C2"]] = 1
ALLOWED_MASK[idx["C2"], idx["Stop2"]] = 1
ALLOWED_MASK[idx["Stop2"], idx["B"]] = 1

PI0_INIT_HMM = np.zeros(S, dtype=float)
PI0_INIT_HMM[iB] = 1.0

EPS = 1e-12