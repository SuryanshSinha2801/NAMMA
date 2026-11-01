package namma.core

/** What the app says, from docs/CAPTURE_GUIDE.md (SplatTour's method + what broke on our own videos). */
object Guide {
    /** Ticked before recording can start. */
    val checklist = listOf(
        "Clean the camera lens (a smudge makes every frame soft)",
        "Hold the phone sideways (landscape), slightly angled, and keep it steady",
        "Lights on, all of them, and keep them the same until the end; no sun moving in through windows",
        "Doors as you want them shown; nobody walks through while you film",
        "Plain walls: tape up a poster or put a chair or plant near them",
        "Start at the entrance. One video per floor, never cut it",
        "The app picks the widest lens, 4K, a fast shutter and locks it all for you",
    )

    const val LOCK = "Point at the brightest wall and tap LOCK: exposure, colour and focus stay fixed for the whole video"

    class Pass(val title: String, val say: String, val tip: String)

    /** The passes in every room (one take). */
    val passes = listOf(
        Pass("Lap 1: along the walls", "Lap one. Walk along the walls, about one metre away, looking straight ahead.",
            "About 1 m from the walls, phone at chest height, looking straight ahead"),
        Pass("Lap 2: the other way", "Lap two. The same lap the other way round.",
            "Same lap, the other direction. Show big furniture from both sides"),
        Pass("Loops around furniture", "Now walk a loop around the bed, the table and other big things, so each is seen from all sides.",
            "Walk around each big piece (spiral), don't stand and turn"),
        Pass("Corner to corner", "Now cross the room corner to corner, both diagonals.",
            "Cross along both diagonals, slowly"),
        Pass("Next room or stop", "Walk through the doorway slowly, a little past where the tour should end. Tap next room, or stop if this was the last one.",
            "Doorways: slowly through the middle, once in and once out; go a bit beyond the edge"),
    )

    /** Always on screen while recording. */
    val rules = listOf(
        "Walk, don't pan: zoom with your feet",
        "Turn slowly: 3 seconds for a quarter turn",
        "Aim at eye level, keep the floor out of the picture",
        "Move smoothly; sharp frames beat more frames",
        "Mirrors: pass them sideways",
    )
}
