package namma.capture

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** The two tokens, AES-GCM encrypted with a key that never leaves the phone's Android Keystore (not in backups: allowBackup=false). */
class Secrets(ctx: Context) {
    private val prefs = ctx.getSharedPreferences("secrets", Context.MODE_PRIVATE)

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val g = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        g.init(KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256).build())
        return g.generateKey()
    }

    fun put(name: String, value: String) {
        val c = Cipher.getInstance("AES/GCM/NoPadding"); c.init(Cipher.ENCRYPT_MODE, key())
        prefs.edit().putString(name, Base64.getEncoder().encodeToString(c.iv + c.doFinal(value.toByteArray()))).apply()
    }

    fun get(name: String): String? = try {
        prefs.getString(name, null)?.let { s ->
            val b = Base64.getDecoder().decode(s)
            val c = Cipher.getInstance("AES/GCM/NoPadding"); c.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, b, 0, 12))
            String(c.doFinal(b, 12, b.size - 12))
        }
    } catch (e: Exception) { null }

    val hf get() = get(HF)
    val github get() = get(GH)
    val ready get() = !hf.isNullOrEmpty() && !github.isNullOrEmpty()

    companion object { const val ALIAS = "namma-tokens"; const val HF = "hf"; const val GH = "github" }
}
