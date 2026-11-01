package namma.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.io.RandomAccessFile

/** A recording's upload to the inbox, resumable at any point: the video is cut into parts (each its own LFS object, hashed
 *  once), every finished part is written to the state file at once, and a resumed upload asks the Hub which parts it already
 *  has. At the end one commit adds <job>/parts/000.., <job>/video.json (how to join them), meta.json and status.json
 *  ('uploaded'); the pipeline joins the parts again (v3/inbox.py assemble). */
class Upload(val video: File, val state: File, val job: String, private val partSize: Long = 32L shl 20) {
    class Part(val index: Int, val offset: Long, val length: Long, val sha256: String, var done: Boolean) {
        val path get() = "%03d".format(index)
    }

    var parts: List<Part> = emptyList(); private set
    var sha256 = ""; private set
    var committed = false; private set
    val total get() = video.length()
    val sent get() = parts.filter { it.done }.sumOf { it.length }

    /** Load the saved state, or cut and hash the video (seconds for a few hundred MB). */
    fun prepare() {
        if (state.exists()) {
            val s = JSONObject(state.readText())
            if (s.getLong("size") == video.length() && s.getString("job") == job) {
                sha256 = s.getString("sha256"); committed = s.optBoolean("committed")
                val a = s.getJSONArray("parts")
                parts = (0 until a.length()).map { a.getJSONObject(it).let { p -> Part(p.getInt("i"), p.getLong("o"), p.getLong("n"), p.getString("h"), p.getBoolean("d")) } }
                return
            }
        }
        val whole = MessageDigest.getInstance("SHA-256"); val out = ArrayList<Part>(); val buf = ByteArray(1 shl 20)
        RandomAccessFile(video, "r").use { raf ->
            var o = 0L; var i = 0
            while (o < video.length()) {
                val n = minOf(partSize, video.length() - o); val md = MessageDigest.getInstance("SHA-256"); var left = n
                while (left > 0) { val k = raf.read(buf, 0, minOf(buf.size.toLong(), left).toInt()); if (k < 0) error("video shorter than expected"); md.update(buf, 0, k); whole.update(buf, 0, k); left -= k }
                out.add(Part(i++, o, n, hex(md.digest()), false)); o += n
            }
        }
        parts = out; sha256 = hex(whole.digest()); save()
    }

    fun save() {
        val a = JSONArray(); parts.forEach { a.put(JSONObject().put("i", it.index).put("o", it.offset).put("n", it.length).put("h", it.sha256).put("d", it.done)) }
        val tmp = File(state.path + ".tmp")
        tmp.writeText(JSONObject().put("job", job).put("size", video.length()).put("sha256", sha256).put("committed", committed).put("parts", a).toString())
        tmp.renameTo(state)
    }

    /** Send what is missing, then commit. stop(): checked between parts (return true to pause; run() again resumes).
     *  progress(sent, total) in bytes. Returns true once committed. */
    fun run(hub: Hub, repo: String, meta: JSONObject, stop: () -> Boolean = { false }, progress: (Long, Long) -> Unit = { _, _ -> }): Boolean {
        if (committed) return true
        if (parts.isEmpty()) prepare()
        for (p in parts) {
            if (p.done) continue
            if (stop()) return false
            val base = sent
            hub.putObject(repo, video, p.offset, p.length, p.sha256) { progress(base + it, total) }
            p.done = true; save(); progress(sent, total)
        }
        if (stop()) return false
        if (hub.readText(repo, "$job/status.json") != null) { committed = true; save(); return true }   // committed before the app died
        val now = Jobs.isoNow()
        val manifest = JSONObject().put("name", video.name).put("size", total).put("sha256", sha256)
            .put("parts", JSONArray(parts.map { "$job/parts/${it.path}" }))
        val status = JSONObject().put("stage", "uploaded").put("updated", now).put("history", JSONArray().put(JSONArray(listOf(now, "uploaded"))))
        hub.commit(repo, "$job: uploaded", parts.map { Hub.LfsFile("$job/parts/${it.path}", it.sha256, it.length) },
            linkedMapOf("$job/video.json" to manifest.toString(1).toByteArray(), "$job/meta.json" to meta.toString(1).toByteArray(),
                "$job/status.json" to status.toString(1).toByteArray()))
        committed = true; save(); return true
    }

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
}
