plugins { id("org.jetbrains.kotlin.jvm") }
java { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
dependencies {
    compileOnly("org.json:json:20240303")          // built into Android; only the JVM tests need the jar
    testImplementation("org.json:json:20240303")
    testImplementation(kotlin("test"))
}
tasks.test {
    useJUnitPlatform()
    for (k in listOf("HF_TOKEN", "E2E_VIDEO", "GITHUB_TEST_TOKEN", "GITHUB_REPOSITORY", "UPLOAD_TEST_OUT", "QUALITY_DIR")) environment(k, System.getenv(k) ?: "")
    testLogging { events("passed", "skipped", "failed"); showStandardStreams = true; exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}
