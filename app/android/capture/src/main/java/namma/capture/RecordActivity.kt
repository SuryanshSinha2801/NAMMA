package namma.capture

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.StatFs
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.view.Gravity
import android.view.TextureView
import android.view.Surface
import android.graphics.Matrix
import android.graphics.RectF
import android.view.View
import android.view.WindowManager
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import namma.core.Coach
import namma.core.Guide
import namma.core.Jobs
import namma.core.Quality
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** Guided recording: checklist -> LOCK on the brightest wall -> record with pass prompts (lap, reverse lap, diagonals, doorway)
 *  and live cues from the gyroscope (turn speed), gravity (aim) and the preview picture (blur, darkness, light changes).
 *  After STOP the video check (core Quality: tracking breaks, blur, blank walls, dark, light jumps, floor, length) says whether
 *  the take is worth the GPU hours, before anything is uploaded. */
class RecordActivity : Activity(), SensorEventListener {
    private lateinit var rig: CameraRig
    private lateinit var texture: TextureView
    private lateinit var passTitle: TextView; private lateinit var timer: TextView; private lateinit var cue: TextView
    private lateinit var tip: TextView; private lateinit var info: TextView
    private lateinit var lockBtn: View; private lateinit var recBtn: android.widget.Button; private lateinit var nextBtn: View
    private lateinit var checklist: View
    private var tts: TextToSpeech? = null
    private val coach = Coach()
    private var job = ""; private var file: File? = null
    private var startedAt = 0L; private var pass = 0; private var room = 1
    private val passLog = JSONArray()
    private var lastCueAt = 0L
    private val fw = 480; private val fh = 270                                         // preview samples: 1/4 of 1920x1080 (landscape)
    private val frame = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
    private val px = IntArray(fw * fh)
    private val worker = HandlerThread("check").apply { start() }
    private val work = Handler(worker.looper)
    @Volatile private var busy = false
    private var extractor = Quality.Extractor()
    private val samples = ArrayList<Quality.Sample>()
    @Volatile private var turnSince = 0.0; @Volatile private var elevation = Double.NaN; private var lastGyro = 0L
    @Volatile private var stepsSince = 0.0; private var stepSensor = false; private var totalSteps = 0
    private var camReady = false

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        rig = CameraRig(this, Store.prefs(this).getBoolean("small", false)) { m -> runOnUiThread { say(m, speak = false) } }
        tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) tts?.language = Locale.UK }
        build()
        val need = listOfNotNull(Manifest.permission.CAMERA, if (Build.VERSION.SDK_INT >= 29) Manifest.permission.ACTIVITY_RECOGNITION else null)
            .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (need.isNotEmpty()) requestPermissions(need.toTypedArray(), 1)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) openCamera() else { say("The camera permission is needed", false); finish() }
    }

    private fun build() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        val dm = resources.displayMetrics; val sw = maxOf(dm.widthPixels, dm.heightPixels); val sh = minOf(dm.widthPixels, dm.heightPixels)
        val pw = minOf(sw, sh * 16 / 9); val ph = pw * 9 / 16                         // landscape 16:9 preview, centred
        texture = TextureView(this)
        root.addView(texture, FrameLayout.LayoutParams(pw, ph, Gravity.CENTER))
        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, a: Int, b: Int) = openCamera()
            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, a: Int, b: Int) = fitPreview()
            override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
        }

        val top = Ui.column(this, 8).apply { setBackgroundColor(0x99000000.toInt()) }
        passTitle = Ui.text(this, "Before you start", 18f, bold = true); timer = Ui.text(this, "", 14f, Ui.DIM)
        top.addView(passTitle); top.addView(timer)
        root.addView(top, FrameLayout.LayoutParams(sw * 62 / 100, -2, Gravity.TOP or Gravity.START))

        cue = Ui.text(this, "", 22f, Color.WHITE, true).apply {
            gravity = Gravity.CENTER; setBackgroundColor(0xCCE5484D.toInt()); visibility = View.GONE; setPadding(24, 16, 24, 16)
        }
        root.addView(cue, FrameLayout.LayoutParams(sw * 60 / 100, -2, Gravity.CENTER))

        val bottom = Ui.column(this, 8).apply { setBackgroundColor(0x99000000.toInt()) }
        tip = Ui.text(this, Guide.LOCK, 13f); info = Ui.text(this, "", 11f, Ui.DIM)
        bottom.addView(tip); bottom.addView(info)
        root.addView(bottom, FrameLayout.LayoutParams(sw * 70 / 100, -2, Gravity.BOTTOM or Gravity.START))

        val col = Ui.column(this, 8).apply { gravity = Gravity.CENTER }                // buttons on the right, under the thumb
        lockBtn = Ui.button(this, "LOCK", Ui.CARD) { toggleLock() }
        recBtn = Ui.button(this, "●  REC", Ui.WARN) { if (rig.recording) stopRec() else startRec() }
        nextBtn = Ui.button(this, "Next ▶", Ui.ACCENT) { nextPass() }
        for (v in listOf(lockBtn, recBtn, nextBtn)) col.addView(v, LinearLayout.LayoutParams(Ui.dp(this, 120), -2).apply { setMargins(0, 8, 0, 8) })
        root.addView(col, FrameLayout.LayoutParams(-2, -1, Gravity.END))
        recBtn.isEnabled = false; recBtn.alpha = 0.4f; nextBtn.visibility = View.GONE

        checklist = makeChecklist(); root.addView(checklist, FrameLayout.LayoutParams(-1, -1))
        setContentView(root)
    }

    @Suppress("DEPRECATION")
    private fun rotationDeg() = when (windowManager.defaultDisplay.rotation) { Surface.ROTATION_90 -> 90; Surface.ROTATION_180 -> 180; Surface.ROTATION_270 -> 270; else -> 0 }

    /** Camera buffers come in sensor orientation; turn and scale them to fill the landscape view (as Camera2Basic does). */
    private fun fitPreview() {
        val vw = texture.width.toFloat(); val vh = texture.height.toFloat(); if (vw == 0f) return
        val bw = rig.preview.width.toFloat(); val bh = rig.preview.height.toFloat(); val rot = rotationDeg()
        val m = Matrix(); val view = RectF(0f, 0f, vw, vh); val buf = RectF(0f, 0f, bh, bw); val cx = view.centerX(); val cy = view.centerY()
        if (rot == 90 || rot == 270) {
            buf.offset(cx - buf.centerX(), cy - buf.centerY()); m.setRectToRect(view, buf, Matrix.ScaleToFit.FILL)
            val k = maxOf(vh / bh, vw / bw); m.postScale(k, k, cx, cy); m.postRotate((rot - 180).toFloat(), cx, cy)
        } else if (rot == 180) m.postRotate(180f, cx, cy)
        texture.setTransform(m)
    }

    private fun makeChecklist(): View {
        val c = Ui.column(this, 20).apply { setBackgroundColor(Ui.BG) }
        c.addView(Ui.text(this, "Before you film", 24f, bold = true))
        c.addView(Ui.text(this, "Tick each one. A room takes about 2 minutes, a flat about 3; up to 10 minutes per floor.", 14f, Ui.DIM))
        val boxes = Guide.checklist.map { s -> CheckBox(this).apply { text = s; textSize = 16f; setTextColor(Ui.TEXT); setPadding(8, 16, 8, 16) } }
        boxes.forEach { c.addView(it) }
        c.addView(Ui.wide(Ui.text(this, "While filming the app tells you what to do: three passes per room, and it warns you when you turn too fast, " +
            "aim at the floor, the picture blurs, or the light changes.", 14f, Ui.DIM), this, 12))
        val go = Ui.button(this, "Ready") {
            if (boxes.all { it.isChecked }) { checklist.visibility = View.GONE; passTitle.text = "Lock the camera"; say(Guide.LOCK, speak = true, warn = false) }
            else say("Tick all of them first", false)
        }
        c.addView(Ui.wide(go, this, 20))
        return ScrollView(this).apply { addView(c) }
    }

    private fun openCamera() {
        if (camReady || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val st = texture.surfaceTexture ?: return
        camReady = true
        try { rig.open(st) { runOnUiThread {
            rig.hint = (rig.sensorOrientation - rotationDeg() + 360) % 360; fitPreview()
            info.text = "${rig.lens} · ${rig.size.width}x${rig.size.height} ${if (rig.hevc) "HEVC" else "H.264"} ${rig.bitrate / 1_000_000} Mbps · 30 fps · stabilisation off"
        } } }
        catch (e: Exception) { camReady = false; say("Camera: ${e.message}", false) }
    }

    private fun toggleLock() {
        if (rig.recording) { say("The lock stays on while recording", false); return }
        val pick = rig.lock(!rig.locked)
        (lockBtn as android.widget.Button).text = if (rig.locked) "UNLOCK" else "LOCK"
        recBtn.isEnabled = rig.locked; recBtn.alpha = if (rig.locked) 1f else 0.4f
        if (rig.locked) {
            tip.text = "Locked: ${rig.exposureInfo()}${if (pick == null) " (this phone has no manual shutter: exposure locked as is)" else ""}. Go to the entrance, then tap REC."
            pick?.warn?.let { say(it, true) } ?: say("Locked. Go to the entrance, then tap record.", true, warn = false)
        }
        else tip.text = Guide.LOCK
    }

    private fun startRec() {
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        if (StatFs(dir.path).availableBytes < 2L shl 30) { say("Less than 2 GB free on the phone: free some space first", false); return }
        job = Jobs.newId(); file = File(dir, "$job.mp4")
        try { rig.start(file!!) } catch (e: Exception) { say("Could not start recording: ${e.message}", false); return }
        startedAt = SystemClock.elapsedRealtime(); pass = 0; room = 1
        synchronized(samples) { samples.clear() }; extractor = Quality.Extractor(); turnSince = 0.0; lastGyro = 0L; stepsSince = 0.0; totalSteps = 0
        recBtn.text = "■  STOP"; lockBtn.alpha = 0.4f; nextBtn.visibility = View.VISIBLE
        showPass(); listen(true); tickUi()
    }

    private fun showPass() {
        val p = Guide.passes[pass]
        passTitle.text = "Room $room · ${p.title}"
        tip.text = p.tip + "\n" + Guide.rules.joinToString("\n") { "· $it" }
        passLog.put(JSONObject().put("room", room).put("pass", p.title).put("t", (SystemClock.elapsedRealtime() - startedAt) / 1000.0))
        say(p.say, speak = true, warn = false, show = false)
    }

    private fun nextPass() {
        if (pass < Guide.passes.size - 1) pass++ else { room++; pass = 0 }
        showPass()
    }

    private fun tickUi() {
        if (!rig.recording) return
        val s = (SystemClock.elapsedRealtime() - startedAt) / 1000
        timer.text = "REC  %d:%02d   ·   turn %.0f°/s".format(s / 60, s % 60, coach.turnDegPerS)
        if (texture.isAvailable && !busy) {                                          // the preview picture, 4 per second
            texture.getBitmap(frame); frame.getPixels(px, 0, fw, 0, 0, fw, fh); busy = true
            val t = (now() - startedAt) / 1000.0; val turn = turnSince; turnSince = 0.0; val el = elevation
            val st = if (stepSensor) stepsSince.also { stepsSince = 0.0 } else Double.NaN
            val luma = IntArray(fw * fh) { val c = px[it]; ((c shr 16 and 255) * 299 + (c shr 8 and 255) * 587 + (c and 255) * 114) / 1000 }
            work.post {                                                                  // sample + cues off the UI thread
                val smp = extractor.add(t, luma, fw, fh, turn, el, st); synchronized(samples) { samples.add(smp) }
                val c = coach.frame(now(), smp.luma, smp.sharp); busy = false
                if (c != null) runOnUiThread { show(c) }
            }
        }
        if (SystemClock.elapsedRealtime() - lastCueAt > 2500) cue.visibility = View.GONE
        timer.postDelayed({ tickUi() }, 250)
    }

    private fun now() = SystemClock.elapsedRealtime()

    private fun listen(on: Boolean) {
        val sm = getSystemService(SensorManager::class.java)!!
        if (!on) { sm.unregisterListener(this); return }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        val stepOk = Build.VERSION.SDK_INT < 29 || checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
        stepSensor = stepOk && sm.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) } == true
        coach.hasSteps = stepSensor
        (sm.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER))?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (!rig.recording) return
        val v = e.values
        val c = when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                val t = now(); if (lastGyro > 0) turnSince += Math.toDegrees(Math.sqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).toDouble())) * (t - lastGyro) / 1000.0
                lastGyro = t; coach.gyro(t, v[0], v[1], v[2])
            }
            Sensor.TYPE_STEP_DETECTOR -> { stepsSince += 1.0; totalSteps++; coach.step(now()); null }
            else -> { elevation = Coach.elevation(v[0], v[1], v[2]); coach.gravity(now(), v[0], v[1], v[2]) }
        }
        c?.let { show(it) }
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    private fun show(c: Coach.Cue) = say(c.text, speak = true, warn = true)

    private fun say(text: String, speak: Boolean, warn: Boolean = true, show: Boolean = true) {
        if (show) { cue.text = text; cue.setBackgroundColor(if (warn) 0xCCE5484D.toInt() else 0xCC2BB3A3.toInt()); cue.visibility = View.VISIBLE; lastCueAt = now() }
        if (speak) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue")
        if (warn && speak) vibrate()
    }

    private fun vibrate() {
        @Suppress("DEPRECATION") val v = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
        if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    private fun stopRec() {
        rig.stop(); listen(false)
        val f = file ?: return; val seconds = (SystemClock.elapsedRealtime() - startedAt) / 1000
        recBtn.text = "●  REC"; lockBtn.alpha = 1f; nextBtn.visibility = View.GONE; cue.visibility = View.GONE
        if (!f.exists() || f.length() < 100_000) { say("The recording is empty, try again", false); f.delete(); return }
        work.post {                                                                      // after the last sample is in
            val list = synchronized(samples) { ArrayList(samples) }; val q = Quality.score(list, rig.locked)
            runOnUiThread { checked(f, seconds, q, list) }
        }
    }

    /** The video check's answer, then name + upload (or record again). A bad take is not uploaded unless asked for. */
    private fun checked(f: File, seconds: Long, q: Quality.Report, list: List<Quality.Sample>) {
        val word = when (q.verdict) { Quality.GOOD -> "Good"; Quality.RISKY -> "Risky"; else -> "Bad" }
        val box = Ui.column(this, 20)
        box.addView(Ui.text(this, "Video check: $word (${q.score}/100)", 20f, if (q.good) Ui.ACCENT else Ui.WARN, true))
        box.addView(Ui.text(this, when (q.verdict) {
            Quality.GOOD -> "This take should give a good 3D model."
            Quality.RISKY -> "It may work, but parts of the model can come out broken or smeared."
            else -> "This take will most likely give a bad model. Record again instead of spending GPU hours on it."
        }, 15f))
        box.addView(Ui.text(this, q.words(), 14f, Ui.DIM))
        val name = EditText(this).apply { setText("Room ${java.text.SimpleDateFormat("d MMM HH:mm", Locale.getDefault()).format(java.util.Date())}"); setSelectAllOnFocus(true); setTextColor(Ui.TEXT) }
        box.addView(Ui.text(this, "Name (${seconds / 60} min ${seconds % 60} s)", 14f, Ui.DIM)); box.addView(name)
        val nm = { name.text.toString().ifBlank { "Room" } }
        val d = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(box) }).setCancelable(false)
        if (q.verdict == Quality.BAD) d.setPositiveButton("Record again") { _, _ -> f.delete(); say("Deleted. Go back to the entrance and tap REC when ready.", true, warn = false) }
            .setNeutralButton("Upload anyway") { _, _ -> save(f, nm(), seconds, q, list, forced = true) }
        else d.setPositiveButton("Upload") { _, _ -> save(f, nm(), seconds, q, list, forced = false) }
            .setNeutralButton("Delete") { _, _ -> f.delete() }
        d.show()
    }

    private fun save(f: File, name: String, seconds: Long, q: Quality.Report, list: List<Quality.Sample>, forced: Boolean) {
        val j = LocalJob(job, name, f.path, System.currentTimeMillis()).apply { total = f.length() }
        val meta = JSONObject().put("name", name).put("preset", "best").put("seconds", seconds).put("rooms", room)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL} (Android ${Build.VERSION.RELEASE})")
            .put("camera", JSONObject().put("lens", rig.lens).put("zoom", rig.zoom.toDouble()).put("width", rig.size.width).put("height", rig.size.height)
                .put("fps", 30).put("bitrate", rig.bitrate).put("codec", if (rig.hevc) "hevc" else "h264").put("locked", rig.locked)
                .put("exposure", rig.exposureInfo()).put("manual_shutter", rig.manual != null).put("stabilisation", "off").put("orientation", "landscape"))
            .put("steps", if (stepSensor) totalSteps else null)
            .put("passes", passLog).put("cues", JSONObject(coach.counts.mapKeys { it.key.name }.mapValues { it.value }))
            .put("quality", q.json().put("uploaded_anyway", forced))
            .put("quality_samples", JSONObject().put("columns", JSONArray(listOf("t", "sharp", "crisp", "luma", "texture", "shift", "ncc", "turn_deg", "elevation")))
                .put("rows", JSONArray(list.map { it.json() })))
        j.meta.writeText(meta.toString(1))
        Store.add(this, j); UploadService.start(this); CheckJob.schedule(this)
        say("Saved. Uploading in the background; you can close the app.", true, warn = false)
        texture.postDelayed({ finish() }, 1500)
    }

    override fun onResume() { super.onResume(); if (texture.isAvailable) openCamera() }

    override fun onPause() {
        if (rig.recording) stopRec()                                    // leaving mid-recording: keep what was filmed
        rig.close(); camReady = false                                   // the camera is given back; LOCK again after returning
        (lockBtn as android.widget.Button).text = "LOCK"; recBtn.isEnabled = false; recBtn.alpha = 0.4f
        if (checklist.visibility != View.VISIBLE) tip.text = Guide.LOCK
        super.onPause()
    }

    override fun onDestroy() { listen(false); rig.quit(); tts?.shutdown(); worker.quitSafely(); super.onDestroy() }
}
