import json
import os
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("upper", "lower")
CLASSES = {
    "upper": ("goggles", "no_goggles", "empty_wrist", "not_empty_wrist"),
    "lower": ("SAFETY_SHOE", "NOT_SAFETY_SHOE"),
}


class CameraProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    width: int = Field(1920, ge=320, le=3840)
    height: int = Field(1080, ge=240, le=2160)
    fps: int = Field(30, ge=5, le=60)
    controls: dict[str, float] = Field(default_factory=dict)
    device_id: str | None = Field(None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def valid_controls(self):
        allowed = {
            "brightness",
            "contrast",
            "saturation",
            "exposure",
            "auto_exposure",
            "focus",
            "autofocus",
            "gain",
        }
        if self.controls.keys() - allowed:
            raise ValueError("Unsupported camera control name")
        if any(not -2147483648 <= v <= 2147483647 for v in self.controls.values()):
            raise ValueError("Camera control outside supported numeric range")
        if any(
            self.controls[name] not in (0, 1)
            for name in {"auto_exposure", "autofocus"} & self.controls.keys()
        ):
            raise ValueError("Automatic controls must be Off (0) or On (1)")
        return self


class ViewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera: int | str | None = None
    confidence: float = Field(0.35, ge=0.05, le=0.95)
    imgsz: int = Field(960, ge=320, le=1536, multiple_of=32)
    roi: tuple[float, float, float, float] = (0, 0, 1, 1)
    sahi_wrist: bool = True
    sahi_slice: int = Field(640, ge=320, le=1280, multiple_of=32)

    @model_validator(mode="after")
    def valid_roi(self):
        if isinstance(self.camera, int) and not 0 <= self.camera <= 15:
            raise ValueError("Camera index must be 0 to 15")
        if isinstance(self.camera, str) and (not self.camera or len(self.camera) > 2048):
            raise ValueError("Camera device ID must be a nonempty string, at most 2048 characters")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Region must satisfy 0 <= left < right <= 1 and top < bottom")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upper: ViewConfig = Field(default_factory=lambda: ViewConfig(camera=0))
    lower: ViewConfig = Field(default_factory=lambda: ViewConfig(camera=1, sahi_wrist=False))
    profiles: dict[str, CameraProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_profiles(self):
        if len(self.profiles) > 32 or any(not k or len(k) > 2048 for k in self.profiles):
            raise ValueError("Use at most 32 camera profiles with valid device IDs")
        return self


class ConfigStore:
    def __init__(self, path=None):
        self.path = path or ROOT / "config" / "settings.json"
        self.lock = RLock()
        self.warning = ""
        self.value = Settings()
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                for profile in saved.get("profiles", {}).values():
                    for name in {"auto_exposure", "autofocus"} & profile.get("controls", {}).keys():
                        if profile["controls"][name] not in (0, 1):
                            del profile["controls"][name]
                self.value = Settings.model_validate(saved)
            except (ValueError, OSError):
                self.warning = "Saved settings could not be read. Defaults loaded; save to repair."

    def snapshot(self):
        with self.lock:
            return self.value.model_copy(deep=True)

    def save(self, value: Settings):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            try:
                with temp.open("w", encoding="utf-8") as handle:
                    json.dump(value.model_dump(), handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
            finally:
                temp.unlink(missing_ok=True)
            self.value = value.model_copy(deep=True)
            self.warning = ""
