// NAMMA Capture: guided room recording -> resumable upload to the private HF inbox -> GitHub/Kaggle pipeline -> tour link.
// core = plain Kotlin (upload, trigger, guidance logic; unit-tested on the JVM), capture = the Android app.
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
    plugins {                                    // versions only; each module applies what it needs
        id("org.jetbrains.kotlin.jvm") version "2.0.21"
        id("org.jetbrains.kotlin.android") version "2.0.21"
        id("com.android.application") version "8.7.3"
    }
}
dependencyResolutionManagement {
    repositories { mavenCentral(); google() }
}
rootProject.name = "namma-capture"
include(":core")
if (System.getenv("CORE_ONLY") == null) include(":capture")      // CORE_ONLY=1: build/test core without the Android SDK
