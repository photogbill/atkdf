"""atkdf — the Analyst Toolkit's direction-finding and geolocation engine.

Coherent multi-channel samples -> bearings -> fixes. Pure Python (numpy /
scipy), no Qt, no imports from ATK, nothing GPL. Frames come from atkdaq.exe,
a .atkq recording, a SigMF clip, or the simulator -- all one FrameSource port.

The maths that moved from ATK's siga package (geodesy, ephemeris, fusion,
geoloc, the SigMF/BLUE DF reader) keeps its own names here; the engine's own
modules (manifold, calibration, doa, records, ports, sim, engine, atkq) are
built on top. ATK reaches this package through re-export shims, so nothing in
ATK imports it by a path that would create a GPL combined work with atkdaq.
"""

from __future__ import annotations

__version__ = "0.3.0"
API_VERSION = 1

# The moved maths (same names as in ATK's siga)
from . import bluefile, df_file, dossier, ephemeris, fusion, geodesy, geoloc
# The engine
from . import (atkq, calibration, doa, engine, exports, fakes, heading, identity, manifold,
               motion, ports, posterior, recio, recorder, records, references, registry,
               roster, scheduler, sim, site, tracker)
# 2026-10-29: burst integration, the P25 path tag by frequency, sync-aided
# pilots (pilots imports atkdsp only when a PilotStage is built without an
# injected correlator — importing it here costs nothing)
from . import bandplan, integrate, pilots
# 2026-10-30: the phased-array option (plan §11.4) — array profiles (schema
# v2), the four-architecture dispatcher, the two-table calibration store with
# its persistence policy, channel health, timing tiers, and 2-D DoA. All are
# configuration and estimators; no new hardware dependency.
from . import architectures, arrayprofile, caltables, doa2d, health, timing
from . import channel_grid
from .dossier import Measurement
from .engine import Engine, PlatformMode
from .heading import HeadingFilter
from .arrayprofile import ArrayProfile
from .architectures import ArrayDF
from .manifold import (BeamspaceManifold, Geometry, IdealManifold, MeasuredManifold,
                       PerturbedManifold, ElementPattern, uca, ula, ura, l_array, multi_face)
from .records import BearingRecord, Emitter, EmitterKey, Passport
from .registry import Registry
from .site import FixedSite

__all__ = [
    "Engine", "PlatformMode", "Geometry", "IdealManifold", "MeasuredManifold",
    "PerturbedManifold", "BeamspaceManifold", "ElementPattern",
    "uca", "ula", "ura", "l_array", "multi_face",
    "ArrayProfile", "ArrayDF",
    "architectures", "arrayprofile", "caltables", "channel_grid", "doa2d", "health", "timing",
    "BearingRecord", "Emitter", "EmitterKey", "Passport", "Measurement",
    "Registry", "HeadingFilter", "FixedSite", "site",
    "atkq", "calibration", "doa", "manifold", "records", "registry", "recorder",
    "heading", "identity", "motion", "posterior", "references", "roster", "scheduler",
    "tracker", "exports", "bandplan", "integrate", "pilots",
    "ports", "fakes", "sim", "recio", "geodesy",
    "ephemeris", "fusion", "geoloc", "df_file", "dossier", "bluefile",
]
