from __future__ import annotations

ROWS = 4
COLS = 4
BYTES_PER_ELEM = 4
TILE_ELEMS = ROWS * COLS
NUM_VREGS = 16

VPU_OPS = {"ADD": 0, "MUL": 1, "MAX": 3, "SUM_REDUCE": 4, "CMPLT": 5, "CMPNE": 6, "SUB": 7, "CMPEQ": 8, "MAX_REDUCE": 9, "SHL": 10, "SHR": 11, "MIN": 12, "MIN_REDUCE": 13, "DIV": 14, "AND": 15, "OR": 16, "XOR": 17,
             "FADD": 18, "FMUL": 19, "FSUB": 20, "FMAX": 21, "FCMPLT": 22, "FRECIP": 23, "I2F": 24, "F2I": 25, "NOT": 26, "SELECT": 27, "COPY": 28,
             "SUM_REDUCE_COL": 29, "MAX_REDUCE_COL": 30, "MIN_REDUCE_COL": 31,
             "SUM_REDUCE_TILE": 32, "MAX_REDUCE_TILE": 33, "MIN_REDUCE_TILE": 34,
             "MUL_REDUCE": 35, "MUL_REDUCE_COL": 36, "MUL_REDUCE_TILE": 37,
             "FSUM_REDUCE_TILE": 38, "FMAX_REDUCE_TILE": 39, "FMIN_REDUCE_TILE": 40,
             "FMIN": 41,
             "FSUM_REDUCE": 42, "FMAX_REDUCE": 43, "FMIN_REDUCE": 44,
             "FSUM_REDUCE_COL": 45, "FMAX_REDUCE_COL": 46, "FMIN_REDUCE_COL": 47,
             "FPROD_REDUCE_TILE": 48, "FPROD_REDUCE": 49, "FPROD_REDUCE_COL": 50,
             "EXP2": 51, "LOG2": 52, "SIN": 53, "COS": 54,
             "PACKED_I8_ADD": 55, "PACKED_I8_SUB": 56,
             "PACKED_I8_MAX": 57, "PACKED_I8_MIN": 58,
             "PACKED_I8_NEG": 59, "PACKED_I8_RELU": 60,
             "PACKED_I8_CMPLT": 61, "PACKED_I8_CMPEQ": 62,
             "PACKED_I8_MUL_LOW": 63, "PACKED_I8_MUL_HIGH": 64,
             "PACKED_I8_ABS": 65, "SIGN": 66, "PACKED_I8_SIGN": 67,
             "FSIGN": 68, "ARGMIN": 69, "ARGMAX": 70,
             "CLZ": 71, "POPCOUNT": 72, "CTZ": 73, "BYTE_REVERSE": 74,
             "SAT_ADD_I32": 75, "SAT_SUB_I32": 76,
             "ABS_DIFF_I32": 77, "PACKED_I8_ABS_DIFF": 78,
             "FABS": 79, "ROTL": 80, "ROTR": 81,
             "MIN_U32": 82, "MAX_U32": 83, "PAIR_ROTATE": 84}
VPU_BOOL_OPS = {VPU_OPS["CMPLT"], VPU_OPS["CMPNE"], VPU_OPS["CMPEQ"]}

SXU_OPS = {"LOAD_VREG": 0, "STORE_VREG": 1, "DISPATCH_VPU": 2, "DISPATCH_XLU_BROADCAST": 3, "DISPATCH_MXU": 4, "WAIT_MXU": 5, "LOAD_MXU_RESULT": 6, "HALT": 7, "DISPATCH_SELECT": 8, "BROADCAST_SCALAR": 9, "BROADCAST_ROW": 10, "BROADCAST_COL": 11, "DISPATCH_XLU_TRANSPOSE": 12, "LOAD_VPU_RESULT": 13, "LOAD_XLU_RESULT": 14, "PSUM_WRITE": 15, "PSUM_ACCUMULATE": 16, "PSUM_READ": 17,
             "DISPATCH_MXU_EPILOGUE": 42, "LOAD_EPILOGUE_STAT": 43,
             "SET_REQUANT_CONFIG": 44, "DISPATCH_MXU_REQUANT": 45,
             # Generic-VPU MXU epilogue (CODA-style composable epilogue).
             # Stub: enum value reserved, hardware execution lands in a
             # later slice. See doc/plan-mxu-epilogue-generic-vpu.md.
             "DISPATCH_MXU_VPU_EPILOGUE": 46}
