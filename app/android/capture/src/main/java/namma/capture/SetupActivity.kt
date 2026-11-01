package namma.capture

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.widget.EditText
import android.widget.ScrollView
import namma.core.GitHub
import namma.core.Hub
import namma.core.Pipeline

/** The two tokens, entered once, checked, then kept encrypted on this phone only. */
class SetupActivity : Activity() {
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val s = Secrets(this)
        val root = Ui.column(this).apply { setBackgroundColor(Ui.BG) }
        root.addView(Ui.text(this, "Settings", 24f, bold = true))
        root.addView(Ui.text(this, "Both tokens stay on this phone, encrypted with a key from the phone's keystore. They are never shown again.", 14f, Ui.DIM))

        root.addView(Ui.wide(Ui.text(this, "1. Hugging Face write token", 17f, bold = true), this, 16))
        root.addView(Ui.text(this, "Uploads go to your private dataset ${Pipeline.INBOX}.", 14f, Ui.DIM))
        val hf = field(if (s.hf != null) "saved (type to replace)" else "hf_…")
        root.addView(hf)
        root.addView(Ui.wide(Ui.button(this, "Make a token on huggingface.co", Ui.CARD) { open("https://huggingface.co/settings/tokens/new?tokenType=write") }, this))

        root.addView(Ui.wide(Ui.text(this, "2. GitHub token (starts the run)", 17f, bold = true), this, 20))
        root.addView(Ui.text(this, "Fine-grained token. Repository access: only ${Pipeline.GITHUB_REPO}. Permissions: Contents = Read and write, Actions = Read.", 14f, Ui.DIM))
        val gh = field(if (s.github != null) "saved (type to replace)" else "github_pat_…")
        root.addView(gh)
        root.addView(Ui.wide(Ui.button(this, "Make a token on github.com", Ui.CARD) { open("https://github.com/settings/personal-access-tokens/new") }, this))

        root.addView(Ui.wide(android.widget.CheckBox(this).apply {
            text = "Smaller files: record 1080p instead of 4K (for slow uploads)"; setTextColor(Ui.TEXT)
            isChecked = Store.prefs(this@SetupActivity).getBoolean("small", false)
            setOnCheckedChangeListener { _, on -> Store.prefs(this@SetupActivity).edit().putBoolean("small", on).apply() }
        }, this, 20))
        val msg = Ui.text(this, "", 15f)
        root.addView(Ui.wide(Ui.button(this, "Check and save") {
            val h = hf.text.toString().trim().ifEmpty { s.hf ?: "" }; val g = gh.text.toString().trim().ifEmpty { s.github ?: "" }
            msg.setTextColor(Ui.TEXT); msg.text = "Checking…"
            Thread {
                val res = try {
                    val user = Hub(h).whoami(); val repo = Hub(h).ensureDataset(Pipeline.INBOX)
                    if (!GitHub(g).canPush()) throw Exception("the GitHub token cannot write to ${Pipeline.GITHUB_REPO}")
                    s.put(Secrets.HF, h); s.put(Secrets.GH, g); Store.prefs(this).edit().putString("inbox", repo).apply()
                    "Saved. Hugging Face: $user, GitHub: OK." to true
                } catch (e: Exception) { "Not saved: ${e.message}" to false }
                runOnUiThread { msg.text = res.first; msg.setTextColor(if (res.second) Ui.ACCENT else Ui.WARN); if (res.second) msg.postDelayed({ finish() }, 1200) }
            }.start()
        }, this, 24))
        root.addView(msg)
        setContentView(ScrollView(this).apply { setBackgroundColor(Ui.BG); addView(root) })
    }

    private fun field(hint: String) = EditText(this).apply {
        this.hint = hint; setTextColor(Ui.TEXT); setHintTextColor(Ui.DIM); isSingleLine = true
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
    }

    private fun open(url: String) = startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
}
