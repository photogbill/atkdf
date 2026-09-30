"""Recording a clip, and reprocessing it — the live and file paths are one.

A clip is the atkdaq stream written verbatim (`.atkq`) — which keeps every
per-frame header: the delays, the weights, the fractional delays, the
calibration state and the flags, none of which a plain SigMF file carries —
plus a sidecar JSON with everything the engine needs to reprocess and
everything an analyst needs to read it later: the array geometry and connector
map, the passport(s) seen, the manifold id, the GPS track and heading
estimates for the window, the references seen, the platform mode, and the
decoder events and bearing records for the window (umbrella §11.1).

Reprocessing runs the SAME engine over the clip, so a drive is re-run
afterwards with hindsight — a better manifold, a learned error model, global
fusion — and the reprocessed fix beats the live one for free once the file
path exists (§11.2). This module is that file path.

For interoperability with ATK's Geolocation tab, `to_sigmf` also writes a
5-channel SigMF pair (`core:num_channels`, per-channel `core:sample_rate` —
the trap `recio`/ATK's `recording` both document) from the clip; the sidecar
carries the calibration the SigMF cannot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from . import atkq
from .records import BearingRecord

SIDECAR_VERSION = 1


def _plain(o):
    if is_dataclass(o):
        return {k: _plain(v) for k, v in asdict(o).items()}
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class ClipRecorder:
    """Writes an `.atkq` clip and its sidecar. Feed it the frames as they go
    by (`add_frame`) and the records/events as they are produced; `close`
    writes the sidecar. It never blocks the stream: it appends bytes."""

    def __init__(self, path, geometry=None, manifold_id: str = "ideal",
                 platform_mode: str = "static", site_id: str = "",
                 connector_map=None, extra=None):
        self.path = Path(path)
        self.data = self.path.with_suffix(".atkq")
        self.sidecar = self.path.with_suffix(".sidecar.json")
        self._f = open(self.data, "wb")
        self.meta = {
            "sidecar_version": SIDECAR_VERSION,
            "clip": self.data.name,
            "site_id": site_id,
            "platform_mode": platform_mode,
            "manifold_id": manifold_id,
            "geometry": (geometry.to_dict() if geometry is not None else None),
            "connector_map": list(connector_map) if connector_map is not None
            else (list(geometry.channel_of_element) if geometry is not None else None),
            "passports": {},          # uid -> passport dict
            "track": [],              # positioning fixes for the window
            "headings": [],           # heading estimates for the window
            "references": [],         # references seen (key, residual)
            "events": [],             # decoder burst events
            "records": [],            # bearing records (JSON strings)
            "n_frames": 0, "fc_hz": None, "fs_hz": None, "n_channels": None,
            "stream_sample0_first": None, "stream_sample0_last": None,
        }
        for k, v in (extra or {}).items():
            self.meta[k] = _plain(v)

    def add_frame(self, frame) -> None:
        h = frame.header
        self._f.write(atkq.pack_header(h))
        self._f.write(bytes(frame.payload))
        m = self.meta
        m["n_frames"] += 1
        if m["fs_hz"] is None:
            m["fs_hz"] = int(h.fs_hz)
            m["n_channels"] = int(h.n_channels)
            m["stream_sample0_first"] = int(h.stream_sample0)
        m["fc_hz"] = int(h.fc_hz)
        m["stream_sample0_last"] = int(h.stream_sample0 + h.samples_per_ch)

    def add_passport(self, passport) -> None:
        if passport is not None:
            self.meta["passports"][passport.uid] = _plain(passport)

    def add_record(self, rec: BearingRecord) -> None:
        self.meta["records"].append(rec.to_json())

    def add_event(self, event) -> None:
        self.meta["events"].append(_plain(event))

    def add_fix(self, fix) -> None:
        self.meta["track"].append(_plain(fix))

    def add_heading(self, heading) -> None:
        self.meta["headings"].append(_plain(heading))

    def add_reference(self, ref) -> None:
        self.meta["references"].append(_plain(ref))

    def close(self) -> tuple:
        self._f.close()
        self.sidecar.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        return str(self.data), str(self.sidecar)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_sidecar(path) -> dict:
    """The sidecar beside a clip, whether given the .atkq or the .sidecar.json."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    sc = p.with_suffix(".sidecar.json")
    return json.loads(sc.read_text(encoding="utf-8")) if sc.is_file() else {}


def records_of(path) -> list:
    """The bearing records saved in a clip's sidecar."""
    return [BearingRecord.from_json(s) for s in read_sidecar(path).get("records", [])]


def reprocess(clip_path, engine, events=None):
    """Re-run `engine` over a clip's frames and return the fresh records.

    The SAME engine loop as live (`Engine.process`), so a reprocessed fix is
    at least as good as the live one and usually better (a measured manifold,
    a learned error model, global fusion). `events` overrides the sidecar's
    saved decoder events when a better decode is available on the second pass;
    otherwise the saved events are used.
    """
    from .atkq import FileSource
    data = Path(clip_path)
    if data.suffix == ".json":
        data = data.with_suffix("")
        if not data.suffix:
            data = Path(str(data) + ".atkq")
    if data.suffix != ".atkq":
        data = data.with_suffix(".atkq")

    if events is None:
        from .ports import BurstEvent
        saved = read_sidecar(clip_path).get("events", [])
        events = [BurstEvent(**{k: v for k, v in e.items()
                                if k in BurstEvent.__dataclass_fields__}) for e in saved]
    by_start: dict = {}
    for e in events:
        if e.n0 is not None:
            by_start.setdefault(int(e.n0), []).append(e)

    out = []
    for fr in FileSource(data).frames():
        s0 = fr.header.stream_sample0
        # events whose burst overlaps this frame
        fr_events = []
        for n0, evs in by_start.items():
            if s0 <= n0 < s0 + fr.header.samples_per_ch:
                fr_events.extend(evs)
        out += engine.process(fr, events=fr_events)
    return out


def to_sigmf(clip_path, out_base, calibrated: bool = False) -> tuple:
    """Write a 5-channel SigMF pair from a clip, for ATK's Geolocation tab.

    `core:num_channels` and a per-channel `core:sample_rate`, samples
    interleaved element-by-element (what both `recio` and ATK's `recording`
    de-interleave). With `calibrated=True` the weights+slope are applied so a
    tool that cannot read the sidecar still sees an aligned array; the default
    is the raw payload, and the sidecar carries the calibration. Returns
    (data_path, meta_path)."""
    from .atkq import FileSource
    from .calibration import calibrate
    out_base = Path(out_base)
    data_p = Path(str(out_base) + ".sigmf-data")
    meta_p = Path(str(out_base) + ".sigmf-meta")
    fs = fc = nch = None
    t0 = None
    with data_p.open("wb") as df:
        for fr in FileSource(clip_path if str(clip_path).endswith(".atkq")
                             else str(Path(clip_path).with_suffix(".atkq"))).frames():
            h = fr.header
            fs, fc, nch = int(h.fs_hz), int(h.fc_hz), int(h.n_channels)
            if t0 is None:
                t0 = h.t_utc
            x = calibrate(fr) if calibrated else fr.iq()      # (n_ch, N) complex64
            # interleave element-by-element, cf32
            inter = x.T.reshape(-1).astype(np.complex64)
            df.write(inter.view(np.float32).tobytes())
    import datetime
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": fs,
                       "core:num_channels": nch, "core:version": "1.0.0",
                       "atk:clip_sidecar": Path(clip_path).with_suffix(".sidecar.json").name,
                       "atk:calibrated": bool(calibrated)},
            "captures": [{"core:sample_start": 0, "core:frequency": fc,
                          "core:datetime": (datetime.datetime.utcfromtimestamp(t0 or 0)
                                            .strftime("%Y-%m-%dT%H:%M:%SZ"))}],
            "annotations": []}
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return str(data_p), str(meta_p)
