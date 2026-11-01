package namma.core

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL

/** One HTTP answer. */
class Resp(val code: Int, val body: String, val headers: Map<String, List<String>>) {
    val ok get() = code in 200..299
    fun header(name: String): String? = headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value?.firstOrNull()
}

class HttpError(val code: Int, msg: String) : IOException(msg)

/** Plain HttpURLConnection (part of both Android and the JVM: no libraries in the app). */
object Http {
    var connectTimeoutMs = 20_000
    var readTimeoutMs = 120_000

    fun request(method: String, url: String, headers: Map<String, String> = emptyMap(), body: ByteArray? = null): Resp {
        val c = open(method, url, headers)
        if (body != null) {
            c.doOutput = true; c.setFixedLengthStreamingMode(body.size)
            c.outputStream.use { it.write(body) }
        }
        return finish(c)
    }

    /** PUT `length` bytes of `file` from `offset` (one upload part), reporting bytes sent. */
    fun putFile(url: String, headers: Map<String, String>, file: File, offset: Long, length: Long, progress: (Long) -> Unit = {}): Resp {
        val c = open("PUT", url, headers)
        c.doOutput = true; c.setFixedLengthStreamingMode(length)
        RandomAccessFile(file, "r").use { raf ->
            raf.seek(offset)
            val buf = ByteArray(256 * 1024); var left = length; var sent = 0L
            c.outputStream.use { out ->
                while (left > 0) {
                    val n = raf.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                    if (n < 0) throw IOException("file ended early")
                    out.write(buf, 0, n); left -= n; sent += n; progress(sent)
                }
            }
        }
        return finish(c)
    }

    private fun open(method: String, url: String, headers: Map<String, String>): HttpURLConnection {
        val c = URL(url).openConnection() as HttpURLConnection
        c.requestMethod = method; c.connectTimeout = connectTimeoutMs; c.readTimeout = readTimeoutMs
        c.instanceFollowRedirects = true; c.useCaches = false
        headers.forEach { (k, v) -> c.setRequestProperty(k, v) }
        return c
    }

    private fun finish(c: HttpURLConnection): Resp {
        try {
            val code = c.responseCode
            val stream: InputStream? = if (code >= 400) c.errorStream else c.inputStream
            val text = stream?.use { it.readBytes().toString(Charsets.UTF_8) } ?: ""
            return Resp(code, text, c.headerFields.filterKeys { it != null })
        } finally { c.disconnect() }
    }
}
