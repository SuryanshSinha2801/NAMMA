package namma.capture

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** A recording on this phone and what became of it. upload: waiting | sending | sent. stage/link/reason: from status.json. */
class LocalJob(val id: String, val name: String, val video: String, val created: Long) {
    var upload = "waiting"; var sent = 0L; var total = 0L; var error: String? = null
    var stage: String? = null; var link: String? = null; var reason: String? = null; var updated: String? = null

    val file get() = File(video)
    val meta get() = File(file.parentFile, "$id.meta.json")
    val state get() = File(file.parentFile, "$id.upload.json")

    fun json(): JSONObject = JSONObject().put("id", id).put("name", name).put("video", video).put("created", created).put("upload", upload)
        .put("sent", sent).put("total", total).putOpt("error", error).putOpt("stage", stage).putOpt("link", link).putOpt("reason", reason)
        .putOpt("updated", updated)

    companion object {
        fun from(j: JSONObject) = LocalJob(j.getString("id"), j.getString("name"), j.getString("video"), j.getLong("created")).apply {
            upload = j.optString("upload", "waiting"); sent = j.optLong("sent"); total = j.optLong("total")
            error = j.optString("error").ifEmpty { null }; stage = j.optString("stage").ifEmpty { null }
            link = j.optString("link").ifEmpty { null }; reason = j.optString("reason").ifEmpty { null }; updated = j.optString("updated").ifEmpty { null }
        }
    }
}

/** jobs.json in the app's private files; every change is written at once (the app can be closed at any moment). */
object Store {
    private fun f(ctx: Context) = File(ctx.filesDir, "jobs.json")

    @Synchronized fun all(ctx: Context): List<LocalJob> {
        val file = f(ctx); if (!file.exists()) return emptyList()
        return try { JSONArray(file.readText()).let { a -> (0 until a.length()).map { LocalJob.from(a.getJSONObject(it)) } } } catch (e: Exception) { emptyList() }
    }

    @Synchronized fun save(ctx: Context, jobs: List<LocalJob>) {
        val a = JSONArray(); jobs.forEach { a.put(it.json()) }
        val tmp = File(f(ctx).path + ".tmp"); tmp.writeText(a.toString()); tmp.renameTo(f(ctx))
    }

    @Synchronized fun add(ctx: Context, j: LocalJob) = save(ctx, all(ctx) + j)

    @Synchronized fun update(ctx: Context, id: String, change: (LocalJob) -> Unit) {
        val jobs = all(ctx); jobs.find { it.id == id }?.let(change); save(ctx, jobs)
    }

    @Synchronized fun remove(ctx: Context, id: String) {
        val jobs = all(ctx); jobs.find { it.id == id }?.let { j -> j.file.delete(); j.state.delete(); j.meta.delete() }
        save(ctx, jobs.filter { it.id != id })
    }

    fun prefs(ctx: Context) = ctx.getSharedPreferences("namma", Context.MODE_PRIVATE)
}
