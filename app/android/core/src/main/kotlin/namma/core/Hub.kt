package namma.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.Base64

/** The parts of the Hugging Face Hub API the app needs (the same calls huggingface_hub makes): who am I, create the private inbox
 *  dataset, read a file, upload a large object through the Git LFS batch API (basic or multipart), commit files. */
class Hub(private val token: String, private val endpoint: String = "https://huggingface.co") {
    private val auth get() = mapOf("Authorization" to "Bearer $token", "User-Agent" to "namma-capture")
    private val lfsHeaders get() = auth + mapOf("Accept" to "application/vnd.git-lfs+json", "Content-Type" to "application/vnd.git-lfs+json")

    fun whoami(): String {
        val r = Http.request("GET", "$endpoint/api/whoami-v2", auth)
        if (!r.ok) throw HttpError(r.code, "Hugging Face token not accepted (${r.code})")
        return JSONObject(r.body).getString("name")
    }

    /** Private dataset `name` of this token's user (exists already: fine). Returns owner/name. */
    fun ensureDataset(name: String): String {
        val user = whoami()
        val r = Http.request("POST", "$endpoint/api/repos/create", auth + ("Content-Type" to "application/json"),
            JSONObject().put("type", "dataset").put("name", name).put("private", true).toString().toByteArray())
        if (!r.ok && r.code != 409) throw HttpError(r.code, "could not create dataset $user/$name: ${r.code} ${r.body.take(200)}")
        return "$user/$name"
    }

    /** A small text file of a dataset, or null when it does not exist (yet). */
    fun readText(repo: String, path: String): String? {
        val r = Http.request("GET", "$endpoint/datasets/$repo/resolve/main/$path", auth)
        if (r.code == 404) return null
        if (!r.ok) throw HttpError(r.code, "read $path: ${r.code}")
        return r.body
    }

    /** Upload `length` bytes of `file` at `offset` as one LFS object with this sha256. False: the Hub already has it (a resumed upload). */
    fun putObject(repo: String, file: File, offset: Long, length: Long, oid: String, progress: (Long) -> Unit = {}): Boolean {
        val req = JSONObject().put("operation", "upload").put("transfers", JSONArray(listOf("basic", "multipart")))
            .put("hash_algo", "sha256").put("ref", JSONObject().put("name", "main"))
            .put("objects", JSONArray().put(JSONObject().put("oid", oid).put("size", length)))
        val r = Http.request("POST", "$endpoint/datasets/$repo.git/info/lfs/objects/batch", lfsHeaders, req.toString().toByteArray())
        if (!r.ok) throw HttpError(r.code, "LFS batch: ${r.code} ${r.body.take(300)}")
        val obj = JSONObject(r.body).getJSONArray("objects").getJSONObject(0)
        if (obj.has("error")) throw IOException("LFS batch error: ${obj.getJSONObject("error")}")
        val actions = obj.optJSONObject("actions") ?: return false                         // no actions: already stored
        val up = actions.optJSONObject("upload") ?: return false
        val hdr = up.optJSONObject("header") ?: JSONObject()
        if (hdr.has("chunk_size")) multipart(up, hdr, file, offset, length, oid, progress)
        else {
            val p = Http.putFile(up.getString("href"), headerMap(hdr), file, offset, length, progress)
            if (!p.ok) throw HttpError(p.code, "upload: ${p.code} ${p.body.take(300)}")
        }
        actions.optJSONObject("verify")?.let { v ->
            val vr = Http.request("POST", v.getString("href"), lfsHeaders + headerMap(v.optJSONObject("header") ?: JSONObject()),
                JSONObject().put("oid", oid).put("size", length).toString().toByteArray())
            if (!vr.ok) throw HttpError(vr.code, "verify: ${vr.code} ${vr.body.take(300)}")
        }
        return true
    }

    private fun multipart(up: JSONObject, hdr: JSONObject, file: File, offset: Long, length: Long, oid: String, progress: (Long) -> Unit) {
        val chunk = hdr.getString("chunk_size").toLong()
        val urls = hdr.keys().asSequence().filter { k -> k.all { it.isDigit() } }.sortedBy { it.toInt() }.toList()
        val parts = JSONArray(); var sent = 0L
        for (k in urls) {
            val o = (k.toInt() - 1) * chunk; val n = minOf(chunk, length - o)
            val p = Http.putFile(hdr.getString(k), emptyMap(), file, offset + o, n) { progress(sent + it) }
            if (!p.ok) throw HttpError(p.code, "upload part $k: ${p.code}")
            parts.put(JSONObject().put("partNumber", k.toInt()).put("etag", p.header("ETag") ?: "")); sent += n
        }
        val c = Http.request("POST", up.getString("href"), lfsHeaders, JSONObject().put("oid", oid).put("parts", parts).toString().toByteArray())
        if (!c.ok) throw HttpError(c.code, "complete multipart: ${c.code} ${c.body.take(300)}")
    }

    class LfsFile(val path: String, val oid: String, val size: Long)

    /** One commit on main: LFS files already uploaded + small files given as bytes. */
    fun commit(repo: String, summary: String, lfs: List<LfsFile>, files: Map<String, ByteArray>) {
        val lines = StringBuilder()
        lines.append(JSONObject().put("key", "header").put("value", JSONObject().put("summary", summary).put("description", ""))).append('\n')
        for ((path, b) in files) lines.append(JSONObject().put("key", "file").put("value",
            JSONObject().put("path", path).put("content", Base64.getEncoder().encodeToString(b)).put("encoding", "base64"))).append('\n')
        for (f in lfs) lines.append(JSONObject().put("key", "lfsFile").put("value",
            JSONObject().put("path", f.path).put("algo", "sha256").put("oid", f.oid).put("size", f.size))).append('\n')
        val r = Http.request("POST", "$endpoint/api/datasets/$repo/commit/main", auth + ("Content-Type" to "application/x-ndjson"),
            lines.toString().toByteArray())
        if (!r.ok) throw HttpError(r.code, "commit: ${r.code} ${r.body.take(300)}")
    }

    private fun headerMap(h: JSONObject) = h.keys().asSequence().associateWith { h.getString(it) }

    companion object {
        fun sha256(file: File, offset: Long, length: Long): String {
            val md = MessageDigest.getInstance("SHA-256"); val buf = ByteArray(1 shl 20)
            RandomAccessFile(file, "r").use { raf ->
                raf.seek(offset); var left = length
                while (left > 0) { val n = raf.read(buf, 0, minOf(buf.size.toLong(), left).toInt()); if (n < 0) break; md.update(buf, 0, n); left -= n }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
