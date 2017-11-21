"""
Defines two dictionaries for converting
between text and integer sequences.
"""

char_map_str = """
ー 0
<SPACE> 1
あ 2
い 3
う 4
え 5
お 6
か 7
き 8
く 9
け 10
こ 11
さ 12
し 13
す 14
せ 15
そ 16
た 17
ち 18
つ 19
て 20
と 21
な 22
に 23
ぬ 24
ね 25
の 26
は 27
ひ 28
ふ 29
へ 30
ほ 31
ま 32
み 33
む 34
め 35
も 36
や 37
ゆ 38
よ 39
ら 40
り 41
る 42
れ 43
ろ 44
わ 45
を 46
ん 47
ぁ 48
ぃ 49
ぅ 50
ぇ 51
ぉ 52
ゃ 53
ゅ 54
ょ 55
ゎ 56
っ 57
が 58
ぎ 59
ぐ 60
げ 61
ご 62
ざ 63
じ 64
ず 65
ぜ 66
ぞ 67
だ 68
ぢ 69
づ 70
で 71
ど 72
ば 73
び 74
ぶ 75
べ 76
ぼ 77
ぱ 78
ぴ 79
ぷ 80
ぺ 81
ぽ 82
ゔ 83
"""
# the "blank" character is mapped to 28

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)+1] = ch
index_map[2] = ' '
