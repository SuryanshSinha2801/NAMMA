package namma.core

import org.json.JSONObject
import java.net.URLEncoder
import java.util.Base64

/** Starts the pipeline: a file pushed to the trigger folder on the pipeline branch starts .github/workflows/inbox.yml (GitHub runs
 *  workflows on other branches only from pushes). Token: fine-grained, this repository only, Contents read/write + Actions read. */
class GitHub(private val token: String, val repo: String = Pipeline.GITHUB_REPO, val branch: String = Pipeline.BRANCH,
             private val api: String = "https://api.github.com") {
    private val headers get() = mapOf("Authorization" to "Bearer $token", "Accept" to "application/vnd.github+json",
        "X-GitHub-Api-Version" to "2022-11-28", "User-Agent" to "namma-capture")
    private val ref get() = URLEncoder.encode(branch, "UTF-8")

    /** Can this token push here? (Setup screen check.) */
    fun canPush(): Boolean {
        val r = Http.request("GET", "$api/repos/$repo", headers)
        if (!r.ok) throw HttpError(r.code, "GitHub token not accepted for $repo (${r.code})")
        return JSONObject(r.body).optJSONObject("permissions")?.optBoolean("push") ?: false
    }

    /** Create or update one file on the branch (one commit). */
    fun putFile(path: String, content: ByteArray, message: String) {
        val cur = Http.request("GET", "$api/repos/$repo/contents/$path?ref=$ref", headers)
        val body = JSONObject().put("message", message).put("branch", branch).put("content", Base64.getEncoder().encodeToString(content))
        if (cur.ok) body.put("sha", JSONObject(cur.body).getString("sha"))
        val r = Http.request("PUT", "$api/repos/$repo/contents/$path", headers + ("Content-Type" to "application/json"), body.toString().toByteArray())
        if (!r.ok) throw HttpError(r.code, "GitHub push of $path: ${r.code} ${r.body.take(200)}")
    }

    /** Delete one file on the branch (tests clean up their trigger). */
    fun deleteFile(path: String, message: String) {
        val cur = Http.request("GET", "$api/repos/$repo/contents/$path?ref=$ref", headers); if (!cur.ok) return
        val body = JSONObject().put("message", message).put("branch", branch).put("sha", JSONObject(cur.body).getString("sha"))
        Http.request("DELETE", "$api/repos/$repo/contents/$path", headers + ("Content-Type" to "application/json"), body.toString().toByteArray())
    }

    /** Runs of this workflow on the branch that are queued or running now. */
    fun activeRuns(workflow: String = Pipeline.WORKFLOW): Int {
        val r = Http.request("GET", "$api/repos/$repo/actions/workflows/$workflow/runs?branch=$ref&per_page=20", headers)
        if (!r.ok) throw HttpError(r.code, "GitHub runs: ${r.code}")
        val a = JSONObject(r.body).getJSONArray("workflow_runs")
        return (0 until a.length()).count { a.getJSONObject(it).optString("status") in setOf("queued", "in_progress", "waiting", "requested", "pending") }
    }

    /** Start (or re-check) the pipeline for this job. */
    fun trigger(job: String, why: String) =
        putFile("${Pipeline.TRIGGER_DIR}/$job.json", JSONObject().put("job", job).put("why", why).put("at", Jobs.isoNow()).toString(1).toByteArray(), "inbox: $job ($why)")
}
