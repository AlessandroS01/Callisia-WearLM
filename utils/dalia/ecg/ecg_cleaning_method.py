from enum import Enum


class ECGCleaningMethod(Enum):
    """
    Provides a comprehensive list of the methods by which ECG cleaning is performed in neurokit2.
    """
    NEUROKIT = "neurokit"
    BIOSPPY = "biosppy"
    PANTOMPKINS1985 = "pantompkins1985"
    HAMILTON2002 = "hamilton2002"
    ELGENDI2010 = "elgendi2010"
    ENGZEEMOD2012 = "engzeemod2012"
    VG = "vg"
    TEMPLATECONVOLUTION = "templateconvolution"

class ECGProcessMethod(Enum):
    """
    Provides a comprehensive list of the methods by which ECG process is performed in neurokit2.
    """
    NEUROKIT = "neurokit"
    PANTOMPKINS1985 = "pantompkins1985"
    HAMILTON2002 = "hamilton2002"
    ELGENDI2010 = "elgendi2010"
    ENGZEEMOD2012 = "engzeemod2012"