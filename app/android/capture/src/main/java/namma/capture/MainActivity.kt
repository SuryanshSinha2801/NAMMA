package namma.capture

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.ScrollView
import namma.core.Status
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Home: record, settings, and every recording with its upload / pipeline state and the tour link when it is ready. */
class MainActivity : Activity() {
    private lateinit var list: LinearLayout
    @Volatile private var visible = false

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val root = Ui.column(this).apply { setBackgroundColor(Ui.BG) }
        root.addView(Ui.text(this, "NAMMA Capture", 26f, bold = true))
        root.addView(Ui.text(this, "Film a room the way the 3D model needs it; the app uploads it and brings back the tour link.", 14f, Ui.DIM))
        root.addView(Ui.wide(Ui.button(this, "●  Record a room", Ui.WARN) { startActivity(Intent(this, RecordActivity::class.java)) }, this, 16))
        val r = Ui.row(this)
        r.addView(Ui.button(this, "Settings", Ui.CARD) { startActivity(Intent(this, SetupActivity::class.java)) })
        r.addView(Ui.gap(this, 0).apply { layoutParams = LinearLayout.LayoutParams(Ui.dp(this@MainActivity, 8), 1) })
        r.addView(Ui.button(this, "Refresh", Ui.CARD) { refresh() })
        root.addView(Ui.wide(r, this))
        root.addView(Ui.wide(Ui.text(this, "Your recordings", 18f, bold = true), this, 20))
        list = Ui.column(this, 0); root.addView(list)
        setContentView(ScrollView(this).apply { setBackgroundColor(Ui.BG); addView(root) })
        if (Build.VERSION.SDK_INT >= 33) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 2)
        CheckJob.schedule(this)
    }

    override fun onResume() {
        super.onResume(); visible = true; render()
        if (!Secrets(this).ready) { if (!asked) { asked = true; startActivity(Intent(this, SetupActivity::class.java)) } }
        else { UploadService.start(this); refresh() }
        tick()
    }

    override fun onPause() { visible = false; super.onPause() }

    private fun quality(j: LocalJob): Pair<String, Int>? = try {
        org.json.JSONObject(j.meta.readText()).optJSONObject("quality")?.let { it.getString("verdict") to it.getInt("score") }
    } catch (e: Exception) { null }

    companion object { private var asked = false }                                     // open Settings by itself once

    private fun tick() { list.postDelayed({ if (visible) { render(); tick() } }, 2000) }        // upload progress

    private fun refresh() = Thread {
        val err = try { Work.refresh(this); null } catch (e: Exception) { e.message }
        runOnUiThread { render(); if (err != null) list.addView(Ui.text(this, "Could not check: $err", 13f, Ui.WARN), 0) }
    }.start()

    private fun render() {
        list.removeAllViews()
        val jobs = Store.all(this).sortedByDescending { it.created }
        if (jobs.isEmpty()) list.addView(Ui.text(this, "Nothing yet. Tap Record a room.", 14f, Ui.DIM))
        val fmt = SimpleDateFormat("d MMM, HH:mm", Locale.getDefault())
        for (j in jobs) {
            val c = Ui.card(this)
            c.addView(Ui.text(this, j.name, 17f, bold = true))
            c.addView(Ui.text(this, fmt.format(Date(j.created)) + "  ·  " + (j.total shr 20) + " MB", 13f, Ui.DIM))
            val state = when {
                j.upload != "sent" && j.error != null -> "Upload paused (${j.error}); it continues by itself"
                j.upload != "sent" -> "Uploading: ${if (j.total > 0) 100 * j.sent / j.total else 0} %"
                else -> Status(j.stage ?: "uploaded", j.link, j.reason, j.updated, null).words
            }
            c.addView(Ui.text(this, state, 15f, if (j.stage == "failed") Ui.WARN else Ui.TEXT))
            quality(j)?.let { (v, sc) -> c.addView(Ui.text(this, "Video check: $v ($sc/100)", 13f, if (v == "good") Ui.ACCENT else Ui.WARN)) }
            j.link?.let { l -> c.addView(Ui.wide(Ui.button(this, "Open the 3D tour") { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(l))) }, this)) }
            if (j.upload != "sent" || j.stage in setOf("done", "failed")) c.addView(Ui.wide(Ui.button(this, "Delete from phone", Ui.CARD) {
                AlertDialog.Builder(this).setMessage("Delete ${j.name} from this phone?" + if (j.upload != "sent") " It is not uploaded yet." else "")
                    .setPositiveButton("Delete") { _, _ -> Store.remove(this, j.id); render() }.setNegativeButton("Keep", null).show()
            }, this))
            list.addView(Ui.wide(c, this, 10))
        }
    }
}
