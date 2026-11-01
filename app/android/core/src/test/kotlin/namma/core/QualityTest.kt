package namma.core

import org.json.JSONObject
import java.io.DataInputStream
import java.io.File
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The video check: synthetic pans here, real video in CI (QUALITY_DIR: the hotel-room video that gave run 4, plus copies made
 *  worse with ffmpeg; each <name>.gray = raw 8-bit frames at 4 fps, <name>.json = {w, h, fps, expect}). */
class QualityTest {
    // a wide "room" picture: smooth random texture with some edges, the camera looks at a window of it
    private val W = 2400; private val H = 300
    private val world: IntArray by lazy {
        val r = Random(5); val raw = IntArray(W * H) { r.nextInt(256) }
        val o = IntArray(W * H)
        for (y in 0 until H) for (x in 0 until W) {
            var s = 0; var k = 0
            for (dy in -2..2) for (dx in -2..2) { val yy = y + dy; val xx = x + dx; if (yy in 0 until H && xx in 0 until W) { s += raw[yy * W + xx]; k++ } }
            o[y * W + x] = s / k
        }
        for (i in 0 until 40) { val x0 = r.nextInt(W - 60); for (y in 0 until H) for (x in x0 until x0 + 6) o[y * W + x] = 20 }   // door frames
        o
    }

    private fun view(x0: Int, w: Int = 480, h: Int = 270, gain: Double = 1.0, blur: Int = 0): IntArray {
        val out = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val sx = (x0 + x * 0.5).toInt().mod(W); val sy = (y * H / h).coerceAtMost(H - 1)
            out[y * w + x] = (world[sy * W + sx] * gain).toInt().coerceIn(0, 255)
        }
        if (blur == 0) return out
        val b = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0; var k = 0
            for (dx in -blur..blur) { val xx = x + dx; if (xx in 0 until w) { s += out[y * w + xx]; k++ } }
            b[y * w + x] = s / k
        }
        return b
    }

    private fun run(n: Int, gain: (Int) -> Double = { 1.0 }, blur: (Int) -> Int = { 0 }, pos: (Int) -> Int): Quality.Report {
        val e = Quality.Extractor()
        val s = (0 until n).map { i -> e.add(i * 0.25, view(pos(i), gain = gain(i), blur = blur(i)), 480, 270) }
        return Quality.score(s).also { println("score ${it.score} ${it.verdict}: ${it.words().replace("\n", " | ")}") }
    }

    @Test fun steadyWalkIsGood() {
        val r = run(400) { it * 6 }                                            // 6 px of 240 per sample: slow, steady
        assertEquals(Quality.GOOD, r.verdict); assertEquals(0, r.stats.getInt("breaks"))
    }

    @Test fun jumpsBreakTracking() {
        val r = run(400) { it * 6 + (it / 50) * 300 }                          // a big jump every 12 s
        assertEquals(Quality.BAD, r.verdict); assertTrue(r.stats.getInt("breaks") >= 4, r.words())
    }

    @Test fun blurIsFound() {
        val r = run(400, blur = { if (it in 100..220) 8 else 0 }) { it * 6 }  // 30 s of motion blur
        assertTrue(r.verdict != Quality.GOOD, r.words()); assertTrue(r.stats.getDouble("blurry_s") > 20, r.words())
    }

    @Test fun darkIsFound() {
        val r = run(400, gain = { 0.12 }) { it * 6 }
        assertEquals(Quality.BAD, r.verdict); assertTrue(r.words().contains("dark"), r.words())
    }

    @Test fun shortIsBad() = assertEquals(Quality.BAD, run(80) { it * 6 }.verdict)

    @Test fun matchFindsTheShift() {
        val a = Quality.down(view(0), 480, 270, 6); val b = Quality.down(view(12), 480, 270, 6)      // 12 world px = 24 of 480 view px = 5 %
        val (shift, c) = Quality.match(a, b, 80, 45)
        assertTrue(shift in 0.03..0.07 && c > 0.9, "shift $shift ncc $c")
    }

    @Test fun realVideos() {
        val dir = System.getenv("QUALITY_DIR").orEmpty()
        if (dir.isEmpty()) { println("no QUALITY_DIR: real-video calibration skipped"); return }
        val out = JSONObject(); var bad = 0
        for (meta in File(dir).listFiles { f -> f.name.endsWith(".json") }!!.sortedBy { it.name }) {
            val m = JSONObject(meta.readText()); val w = m.getInt("w"); val h = m.getInt("h"); val fps = m.getDouble("fps")
            val e = Quality.Extractor(); val samples = ArrayList<Quality.Sample>()
            DataInputStream(File(dir, meta.name.removeSuffix(".json") + ".gray").inputStream().buffered()).use { inp ->
                val buf = ByteArray(w * h); var i = 0
                while (true) {
                    try { inp.readFully(buf) } catch (_: java.io.EOFException) { break }
                    samples.add(e.add(i / fps, IntArray(w * h) { buf[it].toInt() and 255 }, w, h)); i++
                }
            }
            val r = Quality.score(samples); val name = meta.name.removeSuffix(".json")
            println("== $name (${samples.size} frames): score ${r.score} ${r.verdict}, expected ${m.getString("expect")}\n${r.words()}\n${r.stats}")
            out.put(name, r.json())
            fun q(v: List<Double>) = listOf(0.05, 0.25, 0.5, 0.75, 0.95).map { v.sorted()[(it * (v.size - 1)).toInt()] }.joinToString(" ") { "%.3g".format(it) }
            val cm = samples.map { it.crisp }.sorted()[samples.size / 2]
            println("   crisp q05..q95: ${q(samples.map { it.crisp })} | sharp: ${q(samples.map { it.sharp })} | shift: ${q(samples.map { it.shift })} | ncc: ${q(samples.map { it.ncc })} | texture: ${q(samples.map { it.texture })}")
            for (k in listOf(0.5, 0.6, 0.7)) {
                val runs = Quality.runs(BooleanArray(samples.size) { samples[it].crisp < k * cm }, 6)
                println("   crisp < $k x median for >= 1.5 s: ${runs.sumOf { it.second - it.first + 1 } * 0.25} s in ${runs.size} stretches")
            }
            val ok = when (m.getString("expect")) { "good" -> r.verdict == Quality.GOOD; "bad" -> r.verdict == Quality.BAD; else -> r.verdict != Quality.GOOD }
            if (!ok) bad++
        }
        File(dir, "report.json").writeText(out.toString(1))
        assertEquals(0, bad, "some videos scored against expectation (see above)")
    }
}
