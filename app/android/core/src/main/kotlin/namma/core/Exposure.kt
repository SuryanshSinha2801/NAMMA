package namma.core

import kotlin.math.max
import kotlin.math.min

/** Sharp frames first: a fast shutter (1/500 s) with whatever ISO keeps the same brightness. Noise is fine for splats, blur is
 *  not. Too dark for 1/500 at the highest ISO: allow up to 1/250, then say "more lights". */
object Exposure {
    const val TARGET_NS = 2_000_000L        // 1/500 s
    const val LIMIT_NS = 4_000_000L         // 1/250 s: slower than this smears when walking

    class Pick(val ns: Long, val iso: Int, val warn: String?) {
        val words get() = "1/${(1e9 / ns).toInt()} s, ISO $iso"
    }

    /** autoNs / autoIso: what auto exposure chose on the brightest wall; ranges: the sensor's. */
    fun pick(autoNs: Long, autoIso: Int, isoMin: Int, isoMax: Int, nsMin: Long, nsMax: Long): Pick {
        val ev = autoNs.toDouble() * autoIso                                         // brightness = time x gain
        var t = min(autoNs, TARGET_NS).coerceIn(nsMin, nsMax)
        if (ev / t > isoMax) t = min(max(t.toDouble(), ev / isoMax).toLong(), max(LIMIT_NS, t)).coerceIn(nsMin, nsMax)
        val need = ev / t
        val iso = need.toInt().coerceIn(isoMin, isoMax)
        val warn = if (need > isoMax * 1.1) "Too dark for a sharp picture: switch more lights on" else null
        return Pick(t, iso, warn)
    }
}
