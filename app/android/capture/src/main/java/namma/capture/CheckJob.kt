package namma.capture

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context

/** Every ~15 min with a network, also after a reboot (persisted): continue unsent uploads, read the jobs' status, re-trigger a
 *  waiting job. A job window is ~10 min; stopping mid-upload is fine (it resumes at the last finished part). */
class CheckJob : JobService() {
    @Volatile private var stop = false

    override fun onStartJob(params: JobParameters): Boolean {
        stop = false
        Thread {
            try { Work.uploadAll(this, { stop }) } catch (_: Exception) {}
            try { if (!stop) Work.refresh(this) } catch (_: Exception) {}
            jobFinished(params, false)
        }.start()
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean { stop = true; return true }

    companion object {
        fun schedule(ctx: Context) {
            val js = ctx.getSystemService(JobScheduler::class.java)!!
            if (js.getPendingJob(7) != null) return
            js.schedule(JobInfo.Builder(7, ComponentName(ctx, CheckJob::class.java)).setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                .setPersisted(true).setPeriodic(15 * 60_000L).build())
        }
    }
}
