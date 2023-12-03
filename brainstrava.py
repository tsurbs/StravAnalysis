bf = "+[-->-[>>+>-----<<]<--<---]>-.>>>+.>>..+++[.>]<<<<.+++.------.<<-.>>>>+."
def bf_chrs_to_speed(c):
    if c == ">": return "5:30 // 20.625"
    if c == "<": return "6:00 // 20.625-22.5"
    if c == "+": return "6:30 // 22.5-24.375"
    if c == "-": return "7:00 // 24.375-26.25"
    if c == ".": return "7:30 // 26.25-28.125"
    if c == ",": return "8:00 // 28.125-30"
    if c == "[": return "8:30 // 30-31.875"
    if c == "]": return "9:00 // 31.875-33.75"
    return ""
for i in range(len(bf)):
    print("Lap "+str(i)+": "+bf_chrs_to_speed(bf[i]))