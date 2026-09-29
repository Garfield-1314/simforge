"""System identification of SISO LTI systems (input u -> output y).

典型用法 / typical usage::

    from simforge.id import load_system_csv, run_identification, write_report

    data = load_system_csv("data.csv")              # t_s,u,y + metadata
    res = run_identification(data, max_order=3)     # automatic order selection
    tf = res.arx.to_tf()
    print(tf.g_s_str(), tf.char_eq_s())
    write_report(res, "output/run1")

电机预设 / motor preset::

    from simforge.id import load_motor_csv
    data = load_motor_csv("run1.csv")               # pwm_pct/enc_delta layout
"""

from .io import SystemData, MotorData, load_motor_csv, load_system_csv, \
    save_motor_csv, save_system_csv, check_data
from .arx import (ArxResult, identify_arx, identify_arx_auto,
                  identify_arx_structure, fit_structure_auto)
from .stepfit import (PemResult, identify_pem, select_structure_pem,
                      StructureSelection, simulate_oe)
from .tf import TransferFunction, first_order_from_arx
from .pipeline import IdentificationResult, run_identification
from .validate import (CheckResult, Thresholds, vaf, rmse, ljung_box,
                       acceptance_checks, estimate_frf, residual_psd)
from .report import write_report, build_figures
from .preprocess import find_segments, holdout_indices, Segment
from . import excitation, simulator

__all__ = [
    "SystemData", "MotorData",
    "load_motor_csv", "load_system_csv", "save_motor_csv", "save_system_csv",
    "check_data",
    "ArxResult", "identify_arx", "identify_arx_auto", "identify_arx_structure",
    "fit_structure_auto",
    "PemResult", "identify_pem", "select_structure_pem", "StructureSelection",
    "simulate_oe",
    "TransferFunction", "first_order_from_arx",
    "IdentificationResult", "run_identification",
    "CheckResult", "Thresholds", "vaf", "rmse", "ljung_box",
    "acceptance_checks", "estimate_frf", "residual_psd",
    "write_report", "build_figures",
    "find_segments", "holdout_indices", "Segment",
    "excitation", "simulator",
]
