package namma.core

import org.json.JSONObject
import java.io.File
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Real uploads (needs HF_TOKEN; skipped without it). Goes to the private test dataset <user>/namma-inbox-test, never the real
 *  inbox, so nothing starts a Kaggle run. The CI workflow then joins the parts with v3/inbox.py (the pipeline's own code) and
 *  compares them with the file. E2E_VIDEO=<path>: also upload that video to the real inbox as a 'quick' job (end-to-end test). */
class UploadTest {
    private val token = System.getenv("HF_TOKEN").orEmpty()
    private val out = File(System.getenv("UPLOAD_TEST_OUT") ?: "build/upload-test").apply { mkdirs() }

    @Test fun interruptedUploadResumes() {
        if (token.isEmpty()) { println("no HF_TOKEN: upload test skipped"); return }
        val hub = Hub(token); val repo = hub.ensureDataset("namma-inbox-test")
        val video = File(out, "test.mp4"); video.writeBytes(Random(7).nextBytes(20 * (1 shl 20) + 12345))   // 3 parts of 8 MB
        val job = "test-" + Jobs.newId(); val state = File(out, "test.upload.json"); state.delete()
        val meta = JSONObject().put("name", "upload test").put("preset", "quick")

        val a = Upload(video, state, job, partSize = 8L shl 20); a.prepare(); assertEquals(3, a.parts.size)
        var partsSent = 0
        assertFalse(a.run(hub, repo, meta, stop = { partsSent >= 1 }, progress = { s, _ -> partsSent = a.parts.count { it.done } }))
        assertEquals(1, a.parts.count { it.done }, "stopped after one part (like closing the app)")

        val b = Upload(video, state, job, partSize = 8L shl 20); b.prepare()                         // the app opened again
        assertEquals(1, b.parts.count { it.done }, "the finished part is remembered")
        var firstReport = -1L
        assertTrue(b.run(hub, repo, meta, progress = { s, _ -> if (firstReport < 0) firstReport = s }))
        assertTrue(firstReport >= b.parts[0].length, "the resumed upload starts after the finished part")

        // the Hub already has every object: a fresh state (e.g. app data cleared) sends nothing and still commits
        for (p in b.parts) assertFalse(hub.putObject(repo, video, p.offset, p.length, p.sha256), "part ${p.index} should already be stored")

        val status = Status.parse(hub.readText(repo, "$job/status.json"))!!; assertEquals("uploaded", status.stage)
        val manifest = JSONObject(hub.readText(repo, "$job/video.json")!!); assertEquals(b.sha256, manifest.getString("sha256"))
        File(out, "job.txt").writeText("$repo $job/video.json ${b.sha256}\n")
        println("uploaded $job: ${video.length()} bytes in ${b.parts.size} parts, resumed after part 1")
    }

    @Test fun endToEndJob() {
        val path = System.getenv("E2E_VIDEO").orEmpty()
        if (token.isEmpty() || path.isEmpty()) { println("no E2E_VIDEO: end-to-end upload skipped"); return }
        val hub = Hub(token); val repo = hub.ensureDataset(Pipeline.INBOX); val video = File(path)
        val job = Jobs.newId(); val up = Upload(video, File(out, "e2e.upload.json"), job); up.prepare()
        val meta = JSONObject().put("name", "end-to-end test (CI)").put("preset", "quick").put("device", "GitHub runner")
        assertTrue(up.run(hub, repo, meta))
        File(out, "e2e_job.txt").writeText(job)
        println("end-to-end job uploaded: $job (${video.length()} bytes)")
    }
}
