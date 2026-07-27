"""
Stable USB camera device identification, via DirectShow's per-device
DevicePath property -- OpenCV's index is not safe to persist as a
camera's identity, since Windows can and does reassign indices across
reboots or replugging (index N might be a different physical camera
next time).

pygrabber's own camera.py-facing helpers (list_device_names() etc.) only
ever surface FriendlyName, via get_moniker_name() internally reading one
property off each device's moniker property bag. DevicePath is read the
exact same way -- same property bag, different key -- so this module
does its own moniker walk (reusing pygrabber's SystemDeviceEnum /
DeviceCategories.VideoInputDevice, not modifying pygrabber or camera.py)
to read both in one pass.

DevicePath is stable across reboots and replugging into the same port
for most UVC devices, but not guaranteed by every driver -- some devices
simply don't expose one. That's the documented fallback case: identity
then falls back to FriendlyName (with the known limitation that two
identical unnamed devices can't be told apart by name alone), backed up
by the user-assigned alias/role stored once a profile exists for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from comtypes import GUID, client
from comtypes.persist import IPropertyBag
from pygrabber.dshow_core import ICreateDevEnum
from pygrabber.dshow_ids import DeviceCategories, clsids


@dataclass
class CameraIdentity:
    index: int  # current enumeration position -- NOT stable, resolve fresh each time
    name: str  # DirectShow FriendlyName
    device_path: str | None  # stable identity when the driver exposes one


def list_camera_identities() -> list[CameraIdentity]:
    """
    Enumerates video input devices in DirectShow's own order -- the same
    order cv2.VideoCapture(index, CAP_DSHOW) indexes into, which is why
    `index` here is meaningful as a *momentary* connect parameter (never
    as a persisted identity). Returns [] if enumeration fails for any
    reason (no pygrabber/comtypes issue should ever crash the caller --
    same defensive stance as camera.py's own list_device_names()).
    """
    try:
        system_device_enum = client.CreateObject(clsids.CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
        enum_moniker = system_device_enum.CreateClassEnumerator(GUID(DeviceCategories.VideoInputDevice), dwFlags=0)

        if enum_moniker is None:
            return []  # zero devices in this category -- CreateClassEnumerator returns None, not an empty enumerator

        identities: list[CameraIdentity] = []
        index = 0
        moniker, count = enum_moniker.Next(1)

        while count > 0:
            name, device_path = _read_moniker_identity(moniker)
            identities.append(CameraIdentity(index=index, name=name, device_path=device_path))
            index += 1
            moniker, count = enum_moniker.Next(1)

        return identities
    except Exception:
        return []


def _read_moniker_identity(moniker) -> tuple[str, str | None]:
    property_bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)

    name = property_bag.Read("FriendlyName", pErrorLog=None)

    try:
        device_path = property_bag.Read("DevicePath", pErrorLog=None)
    except Exception:
        device_path = None  # not every driver exposes this -- documented fallback, not an error

    return name, (device_path or None)


def find_identity_by_device_path(device_path: str) -> CameraIdentity | None:
    for identity in list_camera_identities():
        if identity.device_path is not None and identity.device_path == device_path:
            return identity
    return None


def find_identity_by_name(name: str) -> CameraIdentity | None:
    for identity in list_camera_identities():
        if identity.name == name:
            return identity
    return None
