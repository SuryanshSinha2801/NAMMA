package namma.core

import kotlin.test.Test
import kotlin.test.assertTrue

/** The app's GitHub calls with the CI token (GITHUB_TEST_TOKEN; skipped without it): push a file to the pipeline branch, count
 *  runs, delete it. A different folder from the trigger folder, so nothing starts. */
class GitHubTest {
    @Test fun pushCountDelete() {
        val tok = System.getenv("GITHUB_TEST_TOKEN").orEmpty()
        if (tok.isEmpty()) { println("no GITHUB_TEST_TOKEN: GitHub test skipped"); return }
        val gh = GitHub(tok, repo = System.getenv("GITHUB_REPOSITORY") ?: Pipeline.GITHUB_REPO)
        val path = "app/android/ci/trigger-test.json"
        gh.putFile(path, """{"test": "${Jobs.isoNow()}"}""".toByteArray(), "app CI: GitHub API test [skip ci]")
        gh.putFile(path, """{"test": "${Jobs.isoNow()}", "again": true}""".toByteArray(), "app CI: GitHub API test (update) [skip ci]")
        val n = gh.activeRuns("android-app.yml"); assertTrue(n >= 1, "this CI run itself should count as running ($n)")
        gh.deleteFile(path, "app CI: GitHub API test done [skip ci]")
        println("GitHub: push, update, run count ($n), delete OK")
    }
}
