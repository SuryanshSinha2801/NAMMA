plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "namma.capture"
    compileSdk = 35
    defaultConfig {
        applicationId = "namma.capture"
        minSdk = 26
        targetSdk = 35
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "1.0.$versionCode"
    }
    // CI signs with one key kept in the private HF dataset (KEYSTORE / KEYSTORE_PASS), so a new build installs over the old one
    val ks = System.getenv("KEYSTORE")
    signingConfigs {
        if (ks != null) create("namma") {
            storeFile = file(ks); storePassword = System.getenv("KEYSTORE_PASS"); keyAlias = "namma"; keyPassword = System.getenv("KEYSTORE_PASS")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (ks != null) signingConfigs.getByName("namma") else signingConfigs.getByName("debug")
        }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies { implementation(project(":core")) }
