package namma.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/** Sends recordings in the foreground (notification with progress), also after the app is closed. If the phone kills it, the
 *  15-minute CheckJob carries on; either way the upload resumes at the last finished part. */
class UploadService : Service() {
    @Volatile private var stop = false
    @Volatile private var running = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val n = note(this, "Uploading your recording", "Starting…", 0, 0)
        if (Build.VERSION.SDK_INT >= 29) startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC) else startForeground(ID, n)
        if (!running) { running = true; stop = false; Thread { loop() }.start() }
        return START_STICKY
    }

    private fun loop() {
        val wl = (getSystemService(Context.POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "namma:upload")
        wl.acquire(6 * 3600_000L)
        val nm = getSystemService(NotificationManager::class.java)!!
        try {
            var tries = 0
            while (!stop && Work.pending(this).isNotEmpty() && tries < 200) {
                try {
                    if (Work.uploadAll(this, { stop }) { j ->
                            nm.notify(ID, note(this, "Uploading ${j.name}", "${j.sent shr 20} of ${j.total shr 20} MB", j.sent, j.total))
                        }) break
                    if (!stop) Thread.sleep(5_000)                                 // the check job holds the lock: wait
                } catch (e: Exception) {
                    tries++; nm.notify(ID, note(this, "Upload paused", "No connection? Trying again in 30 s (${e.message?.take(60)})", 0, 0))
                    Thread.sleep(30_000)
                }
            }
            try { Work.refresh(this) } catch (_: Exception) {}
        } finally {
            wl.release(); running = false
            stopForeground(STOP_FOREGROUND_REMOVE)
            if (Work.pending(this).isEmpty()) nm.notify(DONE, note(this, "Upload finished", "The 3D tour is being made; the app shows the link when it is ready", 0, 0, ongoing = false))
            stopSelf()
        }
    }

    override fun onTimeout(startId: Int, fgsType: Int) { stop = true; stopSelf() }       // Android 15: dataSync time limit
    override fun onDestroy() { stop = true; super.onDestroy() }

    companion object {
        const val ID = 1; const val DONE = 2; const val CHANNEL = "upload"

        fun start(ctx: Context) {
            if (Work.pending(ctx).isEmpty()) return
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(Intent(ctx, UploadService::class.java)) else ctx.startService(Intent(ctx, UploadService::class.java))
        }

        fun note(ctx: Context, title: String, text: String, sent: Long, total: Long, ongoing: Boolean = true): Notification {
            val nm = ctx.getSystemService(NotificationManager::class.java)!!
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Uploads", NotificationManager.IMPORTANCE_LOW))
            val open = PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
            val b = Notification.Builder(ctx, CHANNEL).setSmallIcon(R.drawable.ic_stat).setContentTitle(title).setContentText(text)
                .setContentIntent(open).setOngoing(ongoing).setOnlyAlertOnce(true)
            if (total > 0) b.setProgress(1000, (1000 * sent / total).toInt(), false)
            return b.build()
        }
    }
}
