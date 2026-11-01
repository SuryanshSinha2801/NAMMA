package namma.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Will this video give a good 3D model? Checked on the phone before anything is uploaded, so a bad take costs no GPU hours.
 *  The same numbers the pipeline's failures came from (docs/CAPTURE_GUIDE.md, NOTES.md):
 *   - tracking breaks: consecutive frames that barely overlap (fast turns: 22 Oct lost 4 of 9 gaps this way) -> the camera solve
 *     splits the video into pieces
 *   - blurry stretches: >= 1.5 s with no sharp frame, so the frame picker has nothing good to choose there
 *   - featureless stretches: blank walls (22 Oct lost track exactly there)
 *   - dark frames, light jumps while holding still (blotchy walls), floor-filled frames (gyro), length
 *  Frames: the preview (or, in tests, the decoded video) at ~4 per second, about 1/4 of the video size. */
object Quality {
    class Sample(
        val t: Double,          // seconds
        val sharp: Double,      // variance of the Laplacian on the centre crop
        val crisp: Double,      // that variance / the same at half size: fine detail left (blur or soft focus lowers it)
        val luma: Double,       // mean brightness 0..255
        val texture: Double,    // share of pixels with a clear edge (blank wall: tiny)
        val shift: Double,      // movement since the previous sample, share of the frame width
        val ncc: Double,        // how well the previous sample matches at that movement (1 = same picture)
        val turnDeg: Double,    // gyro: degrees turned since the previous sample (NaN: no gyro, e.g. a test video)
        val elevation: Double,  // gyro: degrees the camera points above the horizon (NaN: unknown)
        val steps: Double = Double.NaN,  // steps taken since the previous sample (NaN: no step sensor, e.g. a test video)
    ) {
        fun json(): JSONArray = JSONArray(listOf(r(t, 2), r(sharp, 1), r(crisp, 3), r(luma, 1), r(texture, 4), r(shift, 3), r(ncc, 3),
            if (turnDeg.isNaN()) null else r(turnDeg, 1), if (elevation.isNaN()) null else r(elevation, 1), if (steps.isNaN()) null else steps))
        private fun r(v: Double, d: Int) = Math.round(v * Math.pow(10.0, d.toDouble())) / Math.pow(10.0, d.toDouble())
    }

    /** Turns frames into samples; keeps the previous small frame for the movement estimate. */
    class Extractor {
        private var prev: IntArray? = null; private var pw = 0; private var ph = 0

        fun add(t: Double, luma: IntArray, w: Int, h: Int, turnDeg: Double = Double.NaN, elevation: Double = Double.NaN, steps: Double = Double.NaN): Sample {
            val c = min(w, h) * 6 / 10; val x0 = (w - c) / 2; val y0 = (h - c) / 2
            val sharp = laplacianVar(luma, w, x0, y0, c, c)
            val half = down(luma, w, h, 2); val half2 = laplacianVar(half, w / 2, x0 / 2, y0 / 2, c / 2, c / 2)
            val crisp = if (half2 > 1e-6) sharp / half2 else 0.0
            var sum = 0L; for (v in luma) sum += v
            val f = max(1, min(w, h) / 45); val sw = w / f; val sh = h / f; val small = down(luma, w, h, f)
            var edges = 0
            for (y in 1 until sh - 1) for (x in 1 until sw - 1) {
                val i = y * sw + x
                if (abs(small[i + 1] - small[i - 1]) + abs(small[i + sw] - small[i - sw]) > 24) edges++
            }
            var shift = 0.0; var ncc = 1.0
            val p = prev
            if (p != null && pw == sw && ph == sh) { val m = match(p, small, sw, sh); shift = m.first; ncc = m.second }
            prev = small; pw = sw; ph = sh
            return Sample(t, sharp, crisp, sum.toDouble() / luma.size, edges.toDouble() / ((sw - 2) * (sh - 2)), shift, ncc, turnDeg, elevation, steps)
        }
    }

    class Reason(val text: String, val at: Double?, val penalty: Int)

    class Report(val score: Int, val verdict: String, val reasons: List<Reason>, val stats: JSONObject) {
        val good get() = verdict == GOOD
        fun words(): String = (if (reasons.isEmpty()) listOf("Nothing wrong found") else reasons.map { r ->
            (r.at?.let { "%d:%02d  ".format(it.toInt() / 60, it.toInt() % 60) } ?: "") + r.text }).joinToString("\n")
        fun json(): JSONObject = JSONObject().put("score", score).put("verdict", verdict).put("stats", stats)
            .put("reasons", JSONArray(reasons.map { JSONObject().put("text", it.text).putOpt("at", it.at).put("penalty", it.penalty) }))
    }

    const val GOOD = "good"; const val RISKY = "risky"; const val BAD = "bad"

    // limits (CI calibrates them on the hotel-room video, which gave run 4, and on worse copies of it: QualityTest)
    const val BREAK_SHIFT = 0.30        // moved over 30 % of the frame between two samples (~0.25 s)
    const val BREAK_NCC = 0.55          // or the best match is this poor
    const val FAST_SHIFT = 0.15         // ~35 deg/s with a wide lens: too fast, though it still matches
    const val BLUR_RATIO = 0.30         // sharpness under 30 % of the video's own sharp frames
    const val CRISP_MIN = 0.25          // the whole video soft (out of focus, smeared lens, heavy blur): median under this (CI: good 0.44, soft copy 0.17)
    const val FLAT_TEXTURE = 0.015      // under 1.5 % edge pixels: a blank wall
    const val DARK_LUMA = 45.0
    const val JUMP = 0.35               // brightness change while holding still

    fun score(samples: List<Sample>, locked: Boolean = true): Report {
        val reasons = ArrayList<Reason>(); var hard = false
        val n = samples.size; val dur = if (n > 1) samples.last().t - samples.first().t else 0.0
        val dt = if (n > 1) dur / (n - 1) else 0.25
        fun add(text: String, at: Double?, pen: Int) = reasons.add(Reason(text, at, pen))
        if (dur < 30) { add("Too short (${dur.toInt()} s): a room needs about 2 minutes", null, 60); hard = true }
        else if (dur < 60) add("Short (${dur.toInt()} s): a room usually needs about 2 minutes", null, 15)
        if (dur > 12 * 60) add("Very long (${(dur / 60).toInt()} min): film one floor per video", null, 10)
        if (!locked) add("Exposure was not locked: walls will change brightness", null, 20)
        if (n < 8) return finish(reasons, true, samples, dur)

        // tracking breaks: an event per run of consecutive bad pairs (dark frames excluded: counted below)
        val brk = BooleanArray(n) { i -> i > 0 && samples[i].luma >= DARK_LUMA && (samples[i].shift > BREAK_SHIFT || samples[i].ncc < BREAK_NCC ||
            (!samples[i].turnDeg.isNaN() && samples[i].turnDeg / dt > 100)) }
        val breaks = runs(brk, 1)
        breaks.take(3).forEach { add("Lost track here (moved or turned too fast, or the view changed at once)", samples[it.first].t, 15) }
        if (breaks.size > 3) add("…and ${breaks.size - 3} more places like that", null, 15 * (breaks.size - 3))
        if (breaks.size >= 4) hard = true

        // turned fast (still matching, but at the edge)
        val fastSec = samples.count { it.shift in FAST_SHIFT..BREAK_SHIFT || (!it.turnDeg.isNaN() && it.turnDeg / dt > 35) } * dt
        if (fastSec >= 3) add("Turned too fast for ${fastSec.toInt()} s in total (3 s per quarter turn)", firstRun(samples) { it.shift > FAST_SHIFT }, min(20, (2 * fastSec).toInt()))

        // blurry stretches: >= 1.5 s with no sharp sample
        val ref = samples.map { it.sharp }.sorted()[(n * 0.8).toInt().coerceAtMost(n - 1)]
        val blurry = BooleanArray(n) { samples[it].sharp < BLUR_RATIO * ref }
        val blurRuns = runs(blurry, max(2, (1.5 / dt).toInt()))
        val blurSec = blurRuns.sumOf { it.second - it.first + 1 } * dt
        if (blurSec > 0) add("Blurry for ${blurSec.toInt()} s (moving too fast or shaking)", samples[blurRuns[0].first].t, min(40, (100 * blurSec / dur).toInt() + 5))
        if (blurSec > 0.3 * dur) hard = true

        // the whole video soft: the relative test above cannot see it
        val crispMed = samples.map { it.crisp }.sorted()[n / 2]
        if (crispMed < CRISP_MIN) { add("The whole video is soft (out of focus, a smudged lens or constant motion blur): clean the lens and move slower", null, 50); hard = true }

        // blank walls
        val flatRuns = runs(BooleanArray(n) { samples[it].texture < FLAT_TEXTURE && samples[it].luma >= DARK_LUMA }, max(2, (3.0 / dt).toInt()))
        val flatSec = flatRuns.sumOf { it.second - it.first + 1 } * dt
        if (flatSec > 0) add("Blank wall filled the picture for ${flatSec.toInt()} s: put something in view, or film it from farther away", samples[flatRuns[0].first].t, min(25, (3 * flatSec).toInt()))

        // dark
        val darkFrac = samples.count { it.luma < DARK_LUMA }.toDouble() / n
        if (darkFrac > 0.1) add("Too dark in ${(100 * darkFrac).toInt()} % of the video: switch more lights on", firstRun(samples) { it.luma < DARK_LUMA }, min(50, (100 * darkFrac).toInt()))
        if (darkFrac > 0.4) hard = true

        // light jumps while holding still
        val jumps = (1 until n).filter { val a = samples[it - 1].luma; val b = samples[it].luma
            a > DARK_LUMA && samples[it].shift < 0.05 && abs(b - a) / a > JUMP }
        if (jumps.isNotEmpty()) add("The light changed ${jumps.size} time(s): keep the same lights on the whole time", samples[jumps[0]].t, min(20, 5 * jumps.size))

        // walk, don't pan (step sensor + gyro only): turning with no step within 1 s gives no parallax
        var panSec = 0.0; var turnSec = 0.0; var totalSteps = Double.NaN
        if (samples.count { !it.steps.isNaN() && !it.turnDeg.isNaN() } > n / 2) {
            val pre = DoubleArray(n + 1); for (i in 0 until n) pre[i + 1] = pre[i] + (samples[i].steps.takeIf { !it.isNaN() } ?: 0.0)
            val k = max(1, (1.0 / dt).toInt())
            for (i in 0 until n) {
                val s = samples[i]; if (s.turnDeg.isNaN() || s.turnDeg / dt < Coach.PAN_DEG) continue
                turnSec += dt; if (pre[min(n, i + k + 1)] - pre[max(0, i - k)] == 0.0) panSec += dt
            }
            totalSteps = pre[n]
            if (panSec >= 3) add("Turned on the spot for ${panSec.toInt()} s without walking: walk, don't pan", null, min(30, (2 * panSec).toInt()))
            if (panSec > 10 && panSec > 0.3 * turnSec) hard = true
            if (dur >= 30 && totalSteps / dur < 0.4) add("Barely walked (${totalSteps.toInt()} steps in ${dur.toInt()} s): walk around the room", null, 25)
        }

        // floor filling the picture (gyro only)
        val el = samples.filter { !it.elevation.isNaN() }
        if (el.size > n / 2) {
            val floor = el.count { it.elevation < -25 }.toDouble() / el.size
            if (floor > 0.2) add("Aimed at the floor ${(100 * floor).toInt()} % of the time: aim at eye level", null, min(25, (60 * floor).toInt()))
        }
        return finish(reasons, hard, samples, dur, JSONObject().put("breaks", breaks.size).put("fast_s", fastSec).put("blurry_s", blurSec)
            .put("pan_s", panSec).put("turn_s", turnSec).put("steps", if (totalSteps.isNaN()) null else totalSteps).put("blank_s", flatSec).put("crisp_median", crispMed).put("dark_frac", darkFrac).put("light_jumps", jumps.size).put("sharp_ref", ref))
    }

    private fun finish(reasons: List<Reason>, hard: Boolean, samples: List<Sample>, dur: Double, stats: JSONObject = JSONObject()): Report {
        val score = (100 - reasons.sumOf { it.penalty }).coerceIn(0, 100)
        val verdict = if (hard || score < 50) BAD else if (score < 75) RISKY else GOOD
        return Report(score, verdict, reasons.sortedByDescending { it.penalty }, stats.put("seconds", dur).put("samples", samples.size))
    }

    private fun firstRun(s: List<Sample>, p: (Sample) -> Boolean) = s.firstOrNull(p)?.t

    /** [start, end] index ranges where the flag holds for at least minLen samples in a row. */
    fun runs(flag: BooleanArray, minLen: Int): List<Pair<Int, Int>> {
        val out = ArrayList<Pair<Int, Int>>(); var i = 0
        while (i < flag.size) {
            if (!flag[i]) { i++; continue }
            var j = i; while (j + 1 < flag.size && flag[j + 1]) j++
            if (j - i + 1 >= minLen) out.add(i to j); i = j + 1
        }
        return out
    }

    fun laplacianVar(l: IntArray, w: Int, x0: Int, y0: Int, cw: Int, ch: Int): Double {
        var s = 0.0; var s2 = 0.0; var k = 0
        for (y in y0 + 1 until y0 + ch - 1) for (x in x0 + 1 until x0 + cw - 1) {
            val i = y * w + x; val v = (4 * l[i] - l[i - 1] - l[i + 1] - l[i - w] - l[i + w]).toDouble(); s += v; s2 += v * v; k++
        }
        if (k == 0) return 0.0
        val m = s / k; return s2 / k - m * m
    }

    fun down(l: IntArray, w: Int, h: Int, f: Int): IntArray {
        val sw = w / f; val sh = h / f; val o = IntArray(sw * sh)
        for (y in 0 until sh) for (x in 0 until sw) {
            var s = 0; for (dy in 0 until f) for (dx in 0 until f) s += l[(y * f + dy) * w + x * f + dx]
            o[y * sw + x] = s / (f * f)
        }
        return o
    }

    /** Best shift of b against a (search up to 40 % across, 25 % up/down): (moved share of the width, normalised correlation). */
    fun match(a: IntArray, b: IntArray, w: Int, h: Int): Pair<Double, Double> {
        val mx = (w * 0.4).toInt(); val my = (h * 0.25).toInt(); var best = -2.0; var bx = 0; var by = 0
        for (dy in -my..my) for (dx in -mx..mx) {
            val c = ncc(a, b, w, h, dx, dy, 2); if (c > best) { best = c; bx = dx; by = dy }
        }
        val fine = ncc(a, b, w, h, bx, by, 1)
        return sqrt((bx * bx + by * by).toDouble()) / w to fine
    }

    /** Correlation of a(x, y) with b(x - dx, y - dy) over the overlap (at least 35 % of the frame), every `step` pixels. */
    fun ncc(a: IntArray, b: IntArray, w: Int, h: Int, dx: Int, dy: Int, step: Int): Double {
        val x0 = max(0, dx); val x1 = min(w, w + dx); val y0 = max(0, dy); val y1 = min(h, h + dy)
        if ((x1 - x0) * (y1 - y0) < 0.35 * w * h) return -1.0
        var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0; var k = 0
        var y = y0
        while (y < y1) {
            var x = x0
            while (x < x1) {
                val p = a[y * w + x].toDouble(); val q = b[(y - dy) * w + (x - dx)].toDouble()
                sa += p; sb += q; saa += p * p; sbb += q * q; sab += p * q; k++; x += step
            }
            y += step
        }
        val va = saa - sa * sa / k; val vb = sbb - sb * sb / k
        if (va < 1e-6 || vb < 1e-6) return if (va < 1e-6 && vb < 1e-6) 1.0 else 0.0                 // flat images: no evidence
        return (sab - sa * sb / k) / sqrt(va * vb)
    }
}
