package namma.core

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.sqrt

/** Live guidance while recording, from the phone's sensors and the preview picture. Pure logic (unit-tested): the app feeds it
 *  gyroscope, gravity and preview-frame numbers and shows / speaks the cues it returns. */
class Coach(private val cooldownMs: Long = 6_000) {
    enum class Cue(val text: String) {
        TURN("Turn slower"),
        FLOOR("Aim higher: keep the floor out"),
        CEILING("Aim lower: eye level"),
        BLUR("Slow down: the picture is blurry"),
        DARK("Too dark: switch more lights on"),
        LIGHTS("Keep the same lights"),
        PAN("Walk, don't just turn: zoom with your feet"),
    }

    val counts = mutableMapOf<Cue, Int>()                      // how often each cue fired (saved in meta.json)
    private val lastSaid = mutableMapOf<Cue, Long>()
    private var turnRate = 0.0; private var lastGyroT = 0L     // deg/s, smoothed
    private var fastSince = -1L
    private val sharpHistory = ArrayDeque<Double>(); private var blurRun = 0
    private val lumaHistory = ArrayDeque<Pair<Long, Double>>()

    val turnDegPerS get() = turnRate
    var hasSteps = false                                       // a step sensor is there (else no pan cue)
    private var lastStep = 0L; private var turnSince = -1L

    fun step(tMs: Long) { lastStep = tMs }

    private fun fire(c: Cue, t: Long): Cue? {
        if (t - (lastSaid[c] ?: -cooldownMs) < cooldownMs) return null
        lastSaid[c] = t; counts[c] = (counts[c] ?: 0) + 1; return c
    }

    /** Angular rate in rad/s (any axes). Over 35 deg/s for half a second = turning too fast (a quarter turn should take 3 s). */
    fun gyro(tMs: Long, wx: Float, wy: Float, wz: Float): Cue? {
        val deg = Math.toDegrees(sqrt((wx * wx + wy * wy + wz * wz).toDouble()))
        val dt = if (lastGyroT == 0L) 0.0 else (tMs - lastGyroT) / 1000.0; lastGyroT = tMs
        val a = (dt / 0.3).coerceIn(0.0, 1.0); turnRate += a * (deg - turnRate)
        if (turnRate > TURN_MAX) { if (fastSince < 0) fastSince = tMs; if (tMs - fastSince >= 500) return fire(Cue.TURN, tMs) } else fastSince = -1
        if (hasSteps && turnRate > PAN_DEG) {                                         // turning on the spot: no parallax
            if (turnSince < 0) turnSince = tMs
            if (tMs - turnSince >= 3000 && tMs - lastStep > 3000) return fire(Cue.PAN, tMs)
        } else turnSince = -1
        return null
    }

    /** Gravity / accelerometer (device axes, m/s^2). The back camera looks along -z: its elevation decides floor / ceiling. */
    fun gravity(tMs: Long, gx: Float, gy: Float, gz: Float): Cue? {
        val e = elevation(gx, gy, gz)
        return when { e < -FLOOR_DEG -> fire(Cue.FLOOR, tMs); e > CEILING_DEG -> fire(Cue.CEILING, tMs); else -> null }
    }

    /** One small preview frame: mean brightness 0..255 and sharpness (variance of the Laplacian, see sharpness()). */
    fun frame(tMs: Long, luma: Double, sharp: Double): Cue? {
        if (luma < DARK_LUMA) return fire(Cue.DARK, tMs)
        lumaHistory.addLast(tMs to luma); while (lumaHistory.isNotEmpty() && tMs - lumaHistory.first().first > 700) lumaHistory.removeFirst()
        val old = lumaHistory.first().second
        if (turnRate < 10 && old > 0 && abs(luma - old) / old > 0.4) return fire(Cue.LIGHTS, tMs)   // a jump while holding still
        val base = if (sharpHistory.size >= 8) sharpHistory.sorted()[sharpHistory.size / 2] else null
        if (base != null && sharp < BLUR_RATIO * base) { blurRun++; if (blurRun >= 2) return fire(Cue.BLUR, tMs) }
        else { blurRun = 0; sharpHistory.addLast(sharp); if (sharpHistory.size > 40) sharpHistory.removeFirst() }
        return null
    }

    companion object {
        const val TURN_MAX = 35.0
        const val PAN_DEG = 15.0
        const val FLOOR_DEG = 25.0
        const val CEILING_DEG = 30.0
        const val DARK_LUMA = 45.0
        const val BLUR_RATIO = 0.4

        /** Degrees above the horizon the back camera points at (gravity sensor reads +g "up" in device axes). */
        fun elevation(gx: Float, gy: Float, gz: Float): Double {
            val n = sqrt((gx * gx + gy * gy + gz * gz).toDouble()); if (n < 1e-3) return 0.0
            return Math.toDegrees(asin((-gz / n).coerceIn(-1.0, 1.0)))
        }

        /** Mean of a w x h luma image (0..255). */
        fun mean(l: IntArray) = if (l.isEmpty()) 0.0 else l.sumOf { it.toLong() }.toDouble() / l.size

        /** Variance of the 4-neighbour Laplacian: drops sharply when the frame is motion-blurred. */
        fun sharpness(l: IntArray, w: Int, h: Int): Double {
            var s = 0.0; var s2 = 0.0; var n = 0
            for (y in 1 until h - 1) for (x in 1 until w - 1) {
                val i = y * w + x; val v = (4 * l[i] - l[i - 1] - l[i + 1] - l[i - w] - l[i + w]).toDouble()
                s += v; s2 += v * v; n++
            }
            if (n == 0) return 0.0
            val m = s / n; return s2 / n - m * m
        }
    }
}
