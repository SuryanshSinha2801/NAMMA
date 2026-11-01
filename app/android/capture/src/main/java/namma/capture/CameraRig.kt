package namma.capture

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaFormat
import android.hardware.camera2.CameraMetadata
import namma.core.Exposure
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Surface
import java.io.File
import kotlin.math.atan

/** Camera2 + MediaRecorder with the capture rules built in: widest back lens (0.5x / 0.6x), 4K (1080p if asked) at a fixed
 *  30 fps, HEVC at the encoder's top bitrate, video stabilisation (EIS) off, no HDR, landscape; after LOCK: a fast shutter
 *  (1/500 s, ISO raised to match: noise is fine, blur is not), white balance and focus fixed for the whole video. */
class CameraRig(private val ctx: Context, private val small: Boolean, private val onInfo: (String) -> Unit) {
    private val mgr = ctx.getSystemService(CameraManager::class.java)!!
    private val thread = HandlerThread("camera").apply { start() }
    private val handler = Handler(thread.looper)
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var previewSurface: Surface? = null
    private val recSurface: Surface = MediaCodec.createPersistentInputSurface()
    private var recorder: MediaRecorder? = null
    @Volatile private var last: TotalCaptureResult? = null

    lateinit var id: String; private set
    lateinit var chars: CameraCharacteristics; private set
    var zoom = 1f; private set
    var size = Size(1920, 1080); private set
    var fovDeg = 0.0; private set
    var locked = false; private set
    var recording = false; private set
    var bitrate = 24_000_000; private set
    var hevc = false; private set
    var hint = 0                                                                     // orientation hint, set by the screen
    var manual: Exposure.Pick? = null; private set
    val preview = Size(1920, 1080)
    var lens = ""; private set

    /** Widest back lens: a logical camera whose zoom goes below 1x (the ultra-wide behind it), else the back camera with the
     *  shortest focal length for its sensor width. */
    fun pick() {
        data class C(val id: String, val ch: CameraCharacteristics, val zoom: Float, val fov: Double)
        val back = mgr.cameraIdList.map { it to mgr.getCameraCharacteristics(it) }
            .filter { it.second.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        require(back.isNotEmpty()) { "no back camera" }
        val c = back.map { (i, ch) ->
            val z = if (Build.VERSION.SDK_INT >= 30) ch.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.lower ?: 1f else 1f
            val w = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width ?: 5f
            val f = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull() ?: 4f
            C(i, ch, z, Math.toDegrees(2 * atan(w / (2.0 * f))) / z)                 // horizontal FOV of the sensor's long side
        }.maxByOrNull { it.fov }!!
        id = c.id; chars = c.ch; zoom = c.zoom; fovDeg = c.fov
        lens = if (zoom < 1f) "ultra-wide (${"%.1f".format(zoom)}x)" else "widest lens (${"%.0f".format(fovDeg)}°)"
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val sizes = map.getOutputSizes(MediaRecorder::class.java).toList()
        size = (if (small) null else sizes.firstOrNull { it.width == 3840 && it.height == 2160 })
            ?: sizes.firstOrNull { it.width == 1920 && it.height == 1080 }
            ?: sizes.filter { it.width * 9 == it.height * 16 && it.width <= 1920 }.maxByOrNull { it.width } ?: sizes.maxByOrNull { it.width * it.height }!!
        // HEVC if an encoder takes this size (half the storage of H.264 at the same quality), at its top bitrate (cap 100 Mbps)
        fun cap(mime: String) = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder && mime in it.supportedTypes }
            .mapNotNull { runCatching { it.getCapabilitiesForType(mime).videoCapabilities }.getOrNull() }
            .firstOrNull { it.isSizeSupported(size.width, size.height) }
        val h = cap(MediaFormat.MIMETYPE_VIDEO_HEVC); val a = cap(MediaFormat.MIMETYPE_VIDEO_AVC)
        hevc = h != null
        bitrate = minOf(100_000_000, (h ?: a)?.bitrateRange?.upper ?: 40_000_000)
    }

    val sensorOrientation get() = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

    @SuppressLint("MissingPermission")
    fun open(texture: SurfaceTexture, ready: () -> Unit) {
        pick()
        mgr.openCamera(id, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) { device = d; startSession(texture, ready) }
            override fun onDisconnected(d: CameraDevice) { d.close(); device = null }
            override fun onError(d: CameraDevice, e: Int) { d.close(); device = null; onInfo("Camera error $e: close other camera apps and try again") }
        }, handler)
    }

    @Suppress("DEPRECATION")
    private fun startSession(texture: SurfaceTexture, ready: () -> Unit) {
        recorder(File(ctx.cacheDir, "probe.mp4")).apply { prepare(); release() }             // sizes the persistent surface
        texture.setDefaultBufferSize(preview.width, preview.height); val p = Surface(texture); previewSurface = p
        device!!.createCaptureSession(listOf(p, recSurface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) { session = s; repeat(); ready() }
            override fun onConfigureFailed(s: CameraCaptureSession) { onInfo("This phone refused the camera setup") }
        }, handler)
    }

    private fun request(): CaptureRequest {
        val b = device!!.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        b.addTarget(previewSurface!!); if (recording) b.addTarget(recSurface)
        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        val fps = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.firstOrNull { it.lower == 30 && it.upper == 30 }
        b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps ?: Range(15, 30))
        val stabilizationModes =
            chars.get(
                CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES
            )
        
        val stabilizationMode =
            if (
                stabilizationModes?.contains(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                ) == true
            ) {
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
            } else {
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF
            }
        
        b.set(
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            stabilizationMode
        )
        b.set(CaptureRequest.CONTROL_SCENE_MODE, CaptureRequest.CONTROL_SCENE_MODE_DISABLED)            // no HDR scene mode
        if (Build.VERSION.SDK_INT >= 30 && zoom != 1f) b.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoom)
        if (locked) {
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
            val m = manual
            if (m != null) {                                                       // manual: the fast shutter picked at LOCK
                b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, m.ns); b.set(CaptureRequest.SENSOR_SENSITIVITY, m.iso)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, 33_333_333L)
            } else b.set(CaptureRequest.CONTROL_AE_LOCK, true)
            val minFocus = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            val fd = lockedFocus
            if (minFocus > 0f && fd != null) { b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF); b.set(CaptureRequest.LENS_FOCUS_DISTANCE, fd) }
        } else b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        return b.build()
    }

    private var lockedFocus: Float? = null

    private fun repeat() {
        session?.setRepeatingRequest(request(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult) { last = result }
        }, handler)
    }

    /** Fix exposure, white balance and focus as they are now (the user points at the brightest wall first). */
    fun lock(on: Boolean): Exposure.Pick? {
        lockedFocus = if (on) last?.get(CaptureResult.LENS_FOCUS_DISTANCE) else null
        manual = null
        val canManual = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
        val ns = last?.get(CaptureResult.SENSOR_EXPOSURE_TIME); val iso = last?.get(CaptureResult.SENSOR_SENSITIVITY)
        val isoR = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE); val nsR = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        if (on && canManual && ns != null && iso != null && isoR != null && nsR != null)
            manual = Exposure.pick(ns, iso, isoR.lower, isoR.upper, nsR.lower, minOf(nsR.upper, 33_333_333L))
        locked = on; handler.post { repeat() }
        return manual
    }

    fun exposureInfo(): String {
        manual?.let { return it.words }
        val r = last ?: return ""
        val t = r.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { "1/${(1e9 / it).toInt()} s" } ?: "?"
        return "$t, ISO ${r.get(CaptureResult.SENSOR_SENSITIVITY) ?: "?"}"
    }

    private fun recorder(file: File): MediaRecorder {
        val m = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(ctx) else @Suppress("DEPRECATION") MediaRecorder()
        m.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        m.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        m.setOutputFile(file.path)
        m.setVideoEncodingBitRate(bitrate); m.setVideoFrameRate(30); m.setVideoSize(size.width, size.height)
        m.setVideoEncoder(if (hevc) MediaRecorder.VideoEncoder.HEVC else MediaRecorder.VideoEncoder.H264)
        m.setOrientationHint(hint)                                                // landscape (the screen sets it from its rotation)
        m.setInputSurface(recSurface)
        return m
    }

    fun start(file: File) {
        val r = recorder(file); r.prepare(); r.start(); recorder = r; recording = true; handler.post { repeat() }
    }

    fun stop() {
        if (!recording) return
        recording = false; handler.post { repeat() }
        try { recorder?.stop() } catch (e: RuntimeException) { onInfo("The recording was too short") }
        recorder?.release(); recorder = null
    }

    fun close() {
        stop(); locked = false; lockedFocus = null; manual = null; session?.close(); session = null; device?.close(); device = null; previewSurface?.release(); previewSurface = null
    }

    fun quit() { close(); recSurface.release(); thread.quitSafely() }
}
