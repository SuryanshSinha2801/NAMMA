package namma.capture

import android.content.Context
import namma.core.GitHub
import namma.core.Hub
import namma.core.Jobs
import namma.core.Pipeline
import namma.core.Status
import namma.core.Upload
import org.json.JSONObject
import java.util.concurrent.locks.ReentrantLock

/** What the upload service and the 15-minute background check both do: send unsent recordings (resumable), then read each
 *  job's status.json and, when a job waits and no pipeline run is going, push a trigger again. */
object Work {
    private val lock = ReentrantLock()

    private fun inbox(ctx: Context, hub: Hub): String {
        val p = Store.prefs(ctx); p.getString("inbox", null)?.let { return it }
        return hub.ensureDataset(Pipeline.INBOX).also { p.edit().putString("inbox", it).apply() }
    }

    fun pending(ctx: Context) = Store.all(ctx).filter { it.upload != "sent" && it.file.exists() }

    /** Send every unsent recording. stop() pauses between parts. onChange: after each progress step. True = nothing left. */
    fun uploadAll(ctx: Context, stop: () -> Boolean, onChange: (LocalJob) -> Unit = {}): Boolean {
        if (!lock.tryLock()) return false                                      // the service or the check job is already on it
        try {
            val s = Secrets(ctx); val hub = Hub(s.hf ?: return false); val gh = GitHub(s.github ?: return false)
            val repo = inbox(ctx, hub)
            for (j in pending(ctx)) {
                if (stop()) return false
                val up = Upload(j.file, j.state, j.id); up.prepare()
                Store.update(ctx, j.id) { it.upload = "sending"; it.total = up.total; it.sent = up.sent; it.error = null }
                val meta = if (j.meta.exists()) JSONObject(j.meta.readText()) else JSONObject().put("name", j.name)
                var lastSave = 0L
                val done = try {
                    up.run(hub, repo, meta, stop) { sent, total ->
                        val now = System.currentTimeMillis()
                        if (now - lastSave > 1500 || sent == total) {
                            lastSave = now; Store.update(ctx, j.id) { it.sent = sent; it.total = total }
                            Store.all(ctx).find { it.id == j.id }?.let(onChange)
                        }
                    }
                } catch (e: Exception) {
                    Store.update(ctx, j.id) { it.error = e.message?.take(200) ?: e.toString() }; throw e
                }
                if (!done) return false
                Store.update(ctx, j.id) { it.upload = "sent"; it.stage = "uploaded"; it.error = null; it.sent = it.total }
                gh.trigger(j.id, "uploaded"); Store.prefs(ctx).edit().putLong("lastKick", System.currentTimeMillis()).apply()
                Store.all(ctx).find { it.id == j.id }?.let(onChange)
            }
            return true
        } finally { lock.unlock() }
    }

    /** Read status.json of every sent job that is not finished; re-trigger the pipeline if a job waits and nothing runs. */
    fun refresh(ctx: Context) {
        val s = Secrets(ctx); val hub = Hub(s.hf ?: return); val gh = GitHub(s.github ?: return); val repo = inbox(ctx, hub)
        val open = Store.all(ctx).filter { it.upload == "sent" && it.stage !in Pipeline.FINAL }
        if (open.isEmpty()) return
        for (j in open) Status.parse(hub.readText(repo, "${j.id}/status.json"))?.let { st ->
            Store.update(ctx, j.id) { it.stage = st.stage; it.link = st.link; it.reason = st.reason; it.updated = st.updated }
        }
        val waiting = Store.all(ctx).filter { it.upload == "sent" && it.stage in Pipeline.ACTIVE }
        val p = Store.prefs(ctx); val now = System.currentTimeMillis()
        if (waiting.isNotEmpty() && Jobs.needsKick(waiting.map { it.stage }, gh.activeRuns(), p.getLong("lastKick", 0), now)) {
            gh.trigger(waiting.first().id, "check"); p.edit().putLong("lastKick", now).apply()
        }
    }
}
