package namma.core

import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.random.Random

object Pipeline {
    const val INBOX = "namma-inbox"                                    // private HF dataset <hf user>/namma-inbox
    const val GITHUB_REPO = "RazyAnas/NAMMA"
    const val BRANCH = "claude/wonderful-cori-i3fn41"
    const val TRIGGER_DIR = "colab/real_scenes/v3/inbox"
    const val WORKFLOW = "inbox.yml"
    val ACTIVE = setOf("uploaded", "training", "publishing")            // the pipeline still has work to do
    val FINAL = setOf("done", "failed")
}

/** Stage of a job on the pipeline side (<job>/status.json, written by v3/inbox.py). */
class Status(val stage: String, val link: String?, val reason: String?, val updated: String?, val note: String?) {
    companion object {
        fun parse(text: String?): Status? {
            if (text.isNullOrBlank()) return null
            val j = JSONObject(text)
            fun s(k: String) = j.optString(k).takeIf { it.isNotEmpty() }
            return Status(j.optString("stage", "?"), s("link"), s("reason"), s("updated"), s("note"))
        }
    }
    val words get() = when (stage) {
        "uploaded" -> "Uploaded, waiting for the GPU"
        "training" -> "Building the 3D model (several hours)"
        "publishing" -> "Making the tour"
        "done" -> "Ready"
        "failed" -> "Failed: ${reason ?: "see the GitHub run"}"
        else -> stage
    }
}

object Jobs {
    fun newId(now: Long = System.currentTimeMillis(), rnd: Random = Random.Default): String =
        utc("yyyyMMdd-HHmmss").format(Date(now)) + "-" + "%04x".format(rnd.nextInt(0x10000))

    fun isoNow(now: Long = System.currentTimeMillis()): String = utc("yyyy-MM-dd'T'HH:mm:ss'Z'").format(Date(now))

    private fun utc(p: String) = SimpleDateFormat(p, Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }

    /** Push a trigger again? Only when a job still waits on the pipeline, no inbox workflow is queued or running, and the last
     *  push is at least minGap old (a job can wait hours for the GPU: one check every ~20 min is plenty). */
    fun needsKick(stages: Collection<String?>, activeRuns: Int, lastKickMs: Long, nowMs: Long, minGapMs: Long = 20 * 60_000L): Boolean =
        stages.any { it in Pipeline.ACTIVE } && activeRuns == 0 && nowMs - lastKickMs >= minGapMs
}
