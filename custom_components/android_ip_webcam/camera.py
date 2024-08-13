"""Support for Android IP Webcam Cameras."""

from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Task, StreamWriter, StreamReader
from typing import Coroutine, Optional, Set

from haffmpeg.core import HAFFmpeg, FFMPEG_STDERR
from homeassistant.components import ffmpeg
from homeassistant.components.camera import CAMERA_STREAM_SOURCE_TIMEOUT, DATA_CAMERA_PREFS, \
    CameraEntityFeature
from homeassistant.components.ffmpeg import FFmpegManager
from homeassistant.components.mjpeg import MjpegCamera, filter_urllib3_logging
from homeassistant.components.stream import Stream, create_stream
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_HOST,
    CONF_PASSWORD,
    CONF_USERNAME,
    HTTP_BASIC_AUTHENTICATION,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import AndroidIPCamDataUpdateCoordinator

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the IP Webcam camera from config entry."""
    filter_urllib3_logging()
    coordinator: AndroidIPCamDataUpdateCoordinator = hass.data[DOMAIN][
        config_entry.entry_id
    ]

    ffmpeg_manager = ffmpeg.get_ffmpeg_manager(hass)

    async_add_entities([IPWebcamCamera(coordinator, ffmpeg_manager)])


async def _write_and_drain(writer: StreamWriter, data: bytes) -> None:
    writer.write(data)
    await writer.drain()


class UnixSocketProxy:
    def __init__(self, socket_path, write_timeout=5):
        self.socket_path: str = socket_path
        self.listeners: Set[StreamWriter] = set()
        self.write_timeout = write_timeout

    async def start(self):
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = await asyncio.start_unix_server(self.handle_connection, path=self.socket_path)
        logger.debug(f"Unix socket server started at {self.socket_path}")
        async with server:
            await server.serve_forever()

    async def handle_connection(self, reader: StreamReader, writer: StreamWriter) -> None:
        self.listeners.add(writer)
        writer_task = asyncio.create_task(self.handle_reader(reader, writer))
        try:
            await writer.wait_closed()
        finally:
            self.listeners.remove(writer)
            writer_task.cancel()

    async def handle_reader(self, reader: StreamReader, writer: StreamWriter) -> None:
        try:
            while True:
                data = await reader.read(188)  # Read MPEG-TS packet size (lower latency)
                if not data:
                    break
                logger.log(logging.DEBUG, f"Read {len(data)} bytes from writer")

                async def write_to_listener(listener: StreamWriter):
                    if listener != writer and not listener.is_closing():
                        try:
                            await asyncio.wait_for(_write_and_drain(listener, data), timeout=self.write_timeout)
                        except TimeoutError:
                            logger.warning(f"Write operation timed out after {self.write_timeout} seconds")
                        except Exception as e:
                            logger.error(f"Error writing to listener: {e}")
                            # Optionally, remove the problematic listener
                            # self.listeners.remove(listener)
                        else:
                            logger.log(logging.DEBUG, f"Wrote {len(data)} bytes to listener")

                # Create tasks for writing to each listener
                write_tasks = [asyncio.create_task(write_to_listener(listener))
                               for listener in self.listeners]

                # Wait for all write operations to complete
                await asyncio.gather(*write_tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Reader task was cancelled")
        except Exception as e:
            logger.error(f"Error in handle_reader: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


class CameraMjpegWithAudio(HAFFmpeg):
    """Representation of a mjpeg camera with audio for Home Assistant."""

    def __init__(self, binary: str) -> None:
        super().__init__(binary)
        self.socket_path = f"/tmp/camera_socket_{id(self)}"
        self.unix_socket_proxy = UnixSocketProxy(self.socket_path)
        self.proxy_task: Optional[Task] = None

    def open_camera(self, mjpeg_source: str, audio_source: str) -> Coroutine:
        """Open FFmpeg process to combine MJPEG video stream with audio into a single H.264/AAC stream.

        Args:
            mjpeg_source (str): The source of the MJPEG video stream.
            audio_source (str): The source of the audio stream. Must end with '.wav', '.aac', or '.opus'.

        Returns:
            Coroutine: A coroutine representing the FFmpeg process.
        """

        # Validate audio source
        if not audio_source.endswith(('.wav', '.aac', '.opus')):
            raise ValueError("Audio source must end with '.wav', '.aac', or '.opus'")

        input_source = f"-fflags nobuffer -i {mjpeg_source} -i {audio_source}"

        command = [
            "-c:v", "libx264",  # Transcode MJPEG to H.264 (home assistant stream requirement)
            "-preset", "ultrafast",  # Use the fastest encoding preset
            "-tune", "zerolatency",  # Optimize for low-latency streaming
            "-c:a", "aac",  # Always transcode audio to AAC (home assistant stream requirement)
            "-ac", "2",  # Ensure stereo audio output
            "-f", "mpegts",  # Output MPEG-TS format
            "-nostdin",  # Do not read from stdin
            "-reconnect", "1",  # Enable auto-reconnect
            "-reconnect_streamed", "1",  # Enable auto-reconnect for streamed inputs
            "-reconnect_at_eof", "1",  # Enable auto-reconnect at end of file
            "-reconnect_delay_max", "2",  # Set maximum delay between reconnections
            "-vsync", "1",  # Soft frame dropping
            "-use_wallclock_as_timestamps", "1",  # Use wall clock for RTP timestamps
            "-avoid_negative_ts", "make_zero",
        ]

        if self.proxy_task is not None:
            self.proxy_task.cancel()

        self.proxy_task = asyncio.create_task(self.unix_socket_proxy.start())

        return self.open(
            cmd=command,
            input_source=input_source,
            output=f"unix://{self.socket_path}",
            extra_cmd=None,
            stderr_pipe=True,
        )

    async def close(self, timeout=5):
        if self.proxy_task is not None:
            self.proxy_task.cancel()
            self.proxy_task = None
        await super().close()


class IPWebcamCamera(MjpegCamera):
    """Representation of a IP Webcam camera."""

    _attr_has_entity_name = True
    _attr_supported_features: CameraEntityFeature = CameraEntityFeature.STREAM

    def __init__(self, coordinator: AndroidIPCamDataUpdateCoordinator, ffmpeg_manager: FFmpegManager) -> None:
        """Initialize the camera."""
        self._audio_url = coordinator.cam.audio_aac_url
        self._mjpeg_url = coordinator.cam.mjpeg_url
        self._ffmpeg_manager = ffmpeg_manager

        super().__init__(
            mjpeg_url=self._mjpeg_url,
            still_image_url=coordinator.cam.image_url,
            authentication=HTTP_BASIC_AUTHENTICATION,
            username=coordinator.config_entry.data.get(CONF_USERNAME),
            password=coordinator.config_entry.data.get(CONF_PASSWORD, ""),
        )
        self._attr_unique_id = f"{coordinator.config_entry.entry_id}-camera"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.config_entry.entry_id)},
            name=coordinator.config_entry.data[CONF_HOST],
        )

    async def async_create_stream(self) -> Stream | None:
        """Create a Stream for stream_source."""
        # There is at most one stream (a decode worker) per camera
        if not self._create_stream_lock:
            self._create_stream_lock = asyncio.Lock()
        async with self._create_stream_lock:
            if not self.stream:
                camera_mjpeg_with_audio = CameraMjpegWithAudio(self._ffmpeg_manager.binary)

                async with asyncio.timeout(CAMERA_STREAM_SOURCE_TIMEOUT):
                    mjpeg_source = await self.stream_source()
                if not mjpeg_source:
                    return None

                if not self._audio_url:
                    return None

                await camera_mjpeg_with_audio.open_camera(mjpeg_source, self._audio_url)
                stream_reader = await camera_mjpeg_with_audio.get_reader(FFMPEG_STDERR)

                async def log_stream():
                    async for line in stream_reader:
                        line: bytes
                        logger.debug(line.decode("utf-8", errors="ignore").strip("\n"))

                stream_debug_task = asyncio.create_task(log_stream())

                self.stream = create_stream(
                    self.hass,
                    f"unix://{camera_mjpeg_with_audio.socket_path}",
                    options=self.stream_options,
                    dynamic_stream_settings=await self.hass.data[
                        DATA_CAMERA_PREFS
                    ].get_dynamic_stream_settings(self.entity_id),
                    stream_label=self.entity_id,
                )

                def update_callback() -> None:
                    if not self.stream.available:
                        self.hass.loop.create_task(camera_mjpeg_with_audio.close())
                        if not stream_debug_task.done():
                            stream_debug_task.cancel()
                    self.async_write_ha_state()

                self.stream.set_update_callback(update_callback)
            return self.stream
