# test_section_transition.py
# Tests the section-based mix using chorus/verse timestamps

from sections.section_transition import mix_by_sections

song_a = r"C:\Users\rheam\OneDrive\Documents\ai-dj\09.wav"
song_b = r"C:\Users\rheam\OneDrive\Documents\ai-dj\15.wav"

# ── Test 1: Auto (system picks best strategy) ──
print("\nTest 1: AUTO strategy")
result = mix_by_sections(
    song_a, song_b,
    output_path = r"C:\Users\rheam\OneDrive\Documents\ai-dj\mix_auto.wav",
    preference  = "auto",
    bars        = 8
)

# ── Test 2: Chorus → Chorus (most hype) ──
print("\nTest 2: CHORUS → CHORUS")
result = mix_by_sections(
    song_a, song_b,
    output_path = r"C:\Users\rheam\OneDrive\Documents\ai-dj\mix_chorus_chorus.wav",
    preference  = "chorus_chorus",
    bars        = 8      # 8 bars = fast aggressive swap
)

# ── Test 3: Verse → Verse (smooth) ──
print("\nTest 3: VERSE → VERSE")
result = mix_by_sections(
    song_a, song_b,
    output_path = r"C:\Users\rheam\OneDrive\Documents\ai-dj\mix_verse_verse.wav",
    preference  = "verse_verse",
    bars        = 16     # 16 bars = slow gradual blend
)

# ── Test 4: Chorus → Verse (energy reset) ──
print("\nTest 4: CHORUS → VERSE")
result = mix_by_sections(
    song_a, song_b,
    output_path = r"C:\Users\rheam\OneDrive\Documents\ai-dj\mix_chorus_verse.wav",
    preference  = "chorus_verse",
    bars        = 8
)

# ── Test 5: Verse → Chorus (energy build) ──
print("\nTest 5: VERSE → CHORUS")
result = mix_by_sections(
    song_a, song_b,
    output_path = r"C:\Users\rheam\OneDrive\Documents\ai-dj\mix_verse_chorus.wav",
    preference  = "verse_chorus",
    bars        = 16     # 16 bars = builds slowly into chorus drop
)

print("\n✅ All tests done!")
print("Open each mix_*.wav file to hear the different transitions.")