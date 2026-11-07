"""WinMM endpoint discovery using persistent Plug and Play interface identities.

Numeric WinMM indexes are resolved only at open time, never saved. Device
interface queries follow Microsoft's documented DRV_QUERYDEVICEINTERFACE API.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from uuid import UUID

WAVE_MAPPER = 0xFFFFFFFF
DRV_QUERYDEVICEINTERFACE = 0x080C
DRV_QUERYDEVICEINTERFACESIZE = 0x080D
DRV_QUERYFUNCTIONINSTANCEID = 0x0811
DRV_QUERYFUNCTIONINSTANCEIDSIZE = 0x0812


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str
    index: int = -1


class _WaveInCaps(ctypes.Structure):
    _fields_ = [('manufacturer', ctypes.c_uint16), ('product', ctypes.c_uint16),
                ('version', ctypes.c_uint32), ('name', ctypes.c_wchar * 32),
                ('formats', ctypes.c_uint32), ('channels', ctypes.c_uint16),
                ('reserved', ctypes.c_uint16)]


class _WaveOutCaps(ctypes.Structure):
    _fields_ = _WaveInCaps._fields_ + [('support', ctypes.c_uint32)]


class _Guid(ctypes.Structure):
    _fields_ = [('data1', ctypes.c_uint32), ('data2', ctypes.c_uint16),
                ('data3', ctypes.c_uint16), ('data4', ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(UUID(value).bytes_le)


class _PropertyKey(ctypes.Structure):
    _fields_ = [('format', _Guid), ('id', ctypes.c_uint32)]


class _VariantData(ctypes.Union):
    _fields_ = [('text', ctypes.c_wchar_p),
                ('storage', ctypes.c_byte * (2 * ctypes.sizeof(ctypes.c_void_p)))]


class _PropVariant(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint16), ('reserved', ctypes.c_uint16 * 3),
                ('value', _VariantData)]


def _com_call(interface, slot, types, *args):
    table = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *types)(table[slot])
    return method(interface, *args)


class _EndpointNames:
    """Read complete endpoint names; WinMM's caps names stop at 31 characters.

    The endpoint ID is opaque and is passed directly to IMMDeviceEnumerator.
    All COM pointers stay on the calling thread and are released on exit.
    """
    def __enter__(self):
        self.enumerator = ctypes.c_void_p()
        self.ole = ctypes.WinDLL('ole32.dll')
        self.ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.ole.CoInitializeEx.restype = ctypes.c_long
        self.ole.CoUninitialize.argtypes, self.ole.CoUninitialize.restype = [], None
        self.ole.CoCreateInstance.argtypes = [ctypes.POINTER(_Guid), ctypes.c_void_p,
                                             ctypes.c_uint32, ctypes.POINTER(_Guid),
                                             ctypes.POINTER(ctypes.c_void_p)]
        self.ole.CoCreateInstance.restype = ctypes.c_long
        self.ole.PropVariantClear.argtypes = [ctypes.POINTER(_PropVariant)]
        self.ole.PropVariantClear.restype = ctypes.c_long
        initialized = self.ole.CoInitializeEx(None, 0)
        self.initialized = initialized >= 0
        # RPC_E_CHANGED_MODE means the caller already has a usable apartment.
        if self.initialized or initialized == -2147417850:
            clsid = _Guid.parse('bcde0395-e52f-467c-8e3d-c4579291692e')
            iid = _Guid.parse('a95664d2-9614-4f35-a746-de8db63617e6')
            self.ole.CoCreateInstance(ctypes.byref(clsid), None, 1,
                                      ctypes.byref(iid), ctypes.byref(self.enumerator))
        return self

    def get(self, endpoint_id):
        if not endpoint_id or not self.enumerator:
            return ''
        device, properties = ctypes.c_void_p(), ctypes.c_void_p()
        value = _PropVariant()
        try:
            if _com_call(self.enumerator, 5, [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)],
                         endpoint_id, ctypes.byref(device)) < 0:
                return ''
            if _com_call(device, 4, [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
                         0, ctypes.byref(properties)) < 0:
                return ''
            key = _PropertyKey(_Guid.parse('a45c254e-df1c-4efd-8020-67d146a850e0'), 14)
            if _com_call(properties, 5, [ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant)],
                         ctypes.byref(key), ctypes.byref(value)) >= 0 and value.type == 31:
                return value.value.text or ''  # VT_LPWSTR / PKEY_Device_FriendlyName
            return ''
        finally:
            self.ole.PropVariantClear(ctypes.byref(value))
            for pointer in (properties, device):
                if pointer:
                    _com_call(pointer, 2, [])

    def __exit__(self, *_args):
        if self.enumerator:
            _com_call(self.enumerator, 2, [])
        if self.initialized:
            self.ole.CoUninitialize()


def _query_string(message, index, size_query, string_query):
    size = ctypes.c_size_t()
    if not message(index, size_query, ctypes.addressof(size), 0):
        if 2 <= size.value <= 65536 and not size.value % 2:
            buffer = ctypes.create_unicode_buffer(size.value // 2)
            if not message(index, string_query, ctypes.addressof(buffer), size.value):
                return buffer.value
    return ''


def enumerate_audio_devices(direction: str) -> list[AudioDevice]:
    if direction not in ('input', 'output'):
        raise ValueError("Audio direction must be input or output")
    if os.name != 'nt':
        return []
    lib = ctypes.WinDLL('winmm.dll')
    prefix = 'waveIn' if direction == 'input' else 'waveOut'
    caps_type = _WaveInCaps if direction == 'input' else _WaveOutCaps
    count = getattr(lib, prefix + 'GetNumDevs')
    count.argtypes, count.restype = [], ctypes.c_uint
    get_caps = getattr(lib, prefix + 'GetDevCapsW')
    get_caps.argtypes = [ctypes.c_size_t, ctypes.POINTER(caps_type), ctypes.c_uint]
    get_caps.restype = ctypes.c_uint
    message = getattr(lib, prefix + 'Message')
    message.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t]
    message.restype = ctypes.c_uint
    devices = []
    with _EndpointNames() as names:
        for index in range(count()):
            caps = caps_type()
            if get_caps(index, ctypes.byref(caps), ctypes.sizeof(caps)):
                continue
            identity = _query_string(message, index, DRV_QUERYDEVICEINTERFACESIZE,
                                     DRV_QUERYDEVICEINTERFACE)
            if identity:
                identity = 'interface:' + identity
            else:
                # Retain the legacy fallback identity even when a complete
                # display name is available, to preserve existing selections.
                identity = f'name:{caps.manufacturer}:{caps.product}:{caps.name}'
            endpoint = _query_string(message, index, DRV_QUERYFUNCTIONINSTANCEIDSIZE,
                                     DRV_QUERYFUNCTIONINSTANCEID)
            name = names.get(endpoint) or caps.name or f'Audio device {index + 1}'
            devices.append(AudioDevice(identity, name, index))
    return devices


def resolve_audio_device(direction: str, device_id: str) -> int:
    if not device_id:
        return WAVE_MAPPER
    try:
        devices = enumerate_audio_devices(direction)
    except OSError:
        return WAVE_MAPPER
    matches = [(i, device) for i, device in enumerate(devices)
               if device.id.casefold() == device_id.casefold()]
    if len(matches) != 1:
        return WAVE_MAPPER
    index, device = matches[0]
    return device.index if device.index >= 0 else index


def open_with_default(open_device, preferred: int) -> tuple[int, int]:
    """Try a currently resolved endpoint, then the Windows default if needed."""
    result = open_device(preferred)
    if result and preferred != WAVE_MAPPER:
        return open_device(WAVE_MAPPER), WAVE_MAPPER
    return result, preferred
