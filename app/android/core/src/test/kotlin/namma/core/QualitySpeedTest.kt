package namma.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/** The check must be instant at STOP: samples are made while filming (4 per second, off the UI thread), STOP only scores them. */
class QualitySpeedTest {
    @Test fun fastEnough() {
        val r = Random(1); val w = 270; val h = 480
        val frames = (0 until 8).map { IntArray(w * h) { r.nextInt(256) } }
        val e = Quality.Extractor(); val samples = ArrayList<Quality.Sample>()
        repeat(20) { samples.add(e.add(it * 0.25, frames[it % 8], w, h)) }                          // warm-up
        val t0 = System.nanoTime(); repeat(200) { samples.add(e.add(it * 0.25, frames[it % 8], w, h)) }
        val perSample = (System.nanoTime() - t0) / 200 / 1e6
        val many = (0 until 2880).map { samples[it % samples.size].let { s -> Quality.Sample(it * 0.25, s.sharp, s.crisp, s.luma, s.texture, s.shift, s.ncc, s.turnDeg, s.elevation) } }
        val t1 = System.nanoTime(); Quality.score(many); val scoreMs = (System.nanoTime() - t1) / 1e6
        println("per sample while filming: %.1f ms (budget 250 ms); score of a 12-minute video at STOP: %.1f ms".format(perSample, scoreMs))
        assertTrue(perSample < 60 && scoreMs < 500)
    }
}
