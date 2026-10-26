"""Read Windows camera control ranges without changing settings or starting capture."""

import ctypes
from contextlib import contextmanager
import hashlib
import sys
import uuid

import cv2


# DirectShow property IDs (IAMVideoProcAmp and IAMCameraControl).
PROPERTIES = {
    "brightness": ("proc", 0),
    "contrast": ("proc", 1),
    "saturation": ("proc", 3),
    "gain": ("proc", 9),
    "exposure": ("camera", 4),
    "focus": ("camera", 6),
}
INTERFACES = {
    "proc": "C6E13360-30AC-11D0-A18C-00A0C9118956",
    "camera": "C6E13370-30AC-11D0-A18C-00A0C9118956",
}


def _guid(value):
    return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


def _call(pointer, slot, *types):
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *types)(table[slot])


def _release(pointer):
    if pointer:
        _call(pointer, 2)(pointer)


def _check(result):
    if result < 0:
        raise OSError(f"Windows camera query failed (0x{result & 0xFFFFFFFF:08X}).")


@contextmanager
def _device(path):
    ole = ctypes.WinDLL("ole32")
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    initialized = ole.CoInitializeEx(None, 2)
    # RPC_E_CHANGED_MODE: the request thread already uses the other COM apartment.
    if initialized < 0 and initialized != -2147417850:
        _check(initialized)
    context, moniker, device = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        ole.CreateBindCtx.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
        _check(ole.CreateBindCtx(0, ctypes.byref(context)))
        consumed = ctypes.c_ulong()
        ole.MkParseDisplayName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _check(
            ole.MkParseDisplayName(
                context, "@device:pnp:" + path, ctypes.byref(consumed), ctypes.byref(moniker)
            )
        )
        interface = _guid("56A86895-0AD4-11CE-B03A-0020AF0BA770")
        _check(
            _call(
                moniker,
                8,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(moniker, context, None, interface, ctypes.byref(device))
        )
        yield device
    finally:
        for pointer in (device, moniker, context):
            _release(pointer)
        if initialized >= 0:
            ole.CoUninitialize()


def _query(device):
    controls = {}
    for group, identifier in INTERFACES.items():
        interface, pointer = _guid(identifier), ctypes.c_void_p()
        result = _call(device, 0, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            device, interface, ctypes.byref(pointer)
        )
        if result < 0:
            continue
        try:
            for name, (owner, prop) in PROPERTIES.items():
                if owner != group:
                    continue
                values = [ctypes.c_long() for _ in range(5)]
                result = _call(pointer, 3, ctypes.c_long, *[ctypes.POINTER(ctypes.c_long)] * 5)(
                    pointer, prop, *[ctypes.byref(value) for value in values]
                )
                low, high, step, default, flags = [value.value for value in values]
                if result < 0 or high < low or step <= 0 or not low <= default <= high:
                    continue
                current, mode = ctypes.c_long(), ctypes.c_long()
                read = _call(
                    pointer,
                    5,
                    ctypes.c_long,
                    ctypes.POINTER(ctypes.c_long),
                    ctypes.POINTER(ctypes.c_long),
                )(pointer, prop, ctypes.byref(current), ctypes.byref(mode))
                if flags & 2:
                    controls[name] = dict(
                        min=low,
                        max=high,
                        step=step,
                        default=default,
                        current=current.value if read >= 0 else None,
                    )
                auto = {"exposure": "auto_exposure", "focus": "autofocus"}.get(name)
                if auto and flags & 1:
                    modes = ([0] if flags & 2 else []) + [1]
                    controls[auto] = dict(
                        min=min(modes),
                        max=1,
                        step=1,
                        default=1,
                        current=int(bool(mode.value & 1)) if read >= 0 else None,
                        options=modes,
                    )
        finally:
            _release(pointer)
    return controls


def camera_capabilities(index):
    result = dict(
        available=False, device_id=None, controls={}, message="Camera capabilities are unavailable."
    )
    if sys.platform != "win32":
        return result
    from cv2_enumerate_cameras import enumerate_cameras

    devices = list(enumerate_cameras(cv2.CAP_DSHOW))
    match = next(
        (
            device
            for device in devices
            if (
                device.index == index
                if isinstance(index, int)
                else device.path.casefold() == index.casefold()
            )
        ),
        None,
    )
    if match is None or not match.path:
        return dict(result, message="Camera is disconnected or has no stable device identifier.")
    result["device_id"] = hashlib.sha256(match.path.casefold().encode()).hexdigest()
    try:
        with _device(match.path) as device:
            controls = _query(device)
        return dict(
            result,
            available=True,
            controls=controls,
            message="Driver-reported controls and limits."
            if controls
            else "No adjustable picture controls reported.",
        )
    except OSError:
        return dict(
            result,
            message="The camera driver could not report control limits. Stop other camera apps and refresh.",
        )


def validate_controls(profile, capabilities):
    """Validate against the current physical device, never a guessed numeric range."""
    controls = profile["controls"]
    if not controls:
        return
    if profile.get("device_id") and profile["device_id"] != capabilities.get("device_id"):
        raise ValueError(
            "The camera at this index has changed. Refresh its settings before saving."
        )
    for name, value in controls.items():
        spec = capabilities.get("controls", {}).get(name)
        if not spec:
            raise ValueError(f"{name}: this camera does not report support for this control.")
        if (
            not spec["min"] <= value <= spec["max"]
            or abs(
                (value - spec["min"]) / spec["step"] - round((value - spec["min"]) / spec["step"])
            )
            > 1e-6
        ):
            raise ValueError(
                f"{name}: use {spec['min']} to {spec['max']} in steps of {spec['step']}."
            )
