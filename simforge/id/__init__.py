"""System identification of a brushed DC motor speed loop (PWM -> rpm).

典型用法 / typical usage::

    from simforge.id import load_motor_csv, run_identification, write_report

    data = load_motor_csv("data/motor_run1.csv")
    res = run_identification(data, holdout_duty=50.0)
    write_report(res, "output/run1")
"""

from .io import SystemData, MotorData, load_motor_csv, load_system_csv, \
    save_motor_csv, save_system_csv, check_data
from .arx import ArxResult, identify_arx
from .stepfit import PemResult, identify_pem
from .pipeline import IdentificationResult, run_identification
from .validate import CheckResult, vaf, rmse, ljung_box, acceptance_checks
from .report import write_report, build_figures
from .preprocess import find_segments, holdout_indices, Segment
from . import excitation, simulator

__all__ = [
    "SystemData", "MotorData",
    "load_motor_csv", "load_system_csv", "save_motor_csv", "save_system_csv",
    "check_data",
    "ArxResult", "identify_arx",
    "PemResult", "identify_pem",
    "IdentificationResult", "run_identification",
    "CheckResult", "vaf", "rmse", "ljung_box", "acceptance_checks",
    "write_report", "build_figures",
    "find_segments", "holdout_indices", "Segment",
    "excitation", "simulator",
]
