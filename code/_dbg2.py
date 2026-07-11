import re

BT = "\x60\x60\x60"
with open("chapters/ch24-优化算法.md", "r", encoding="utf-8") as f:
    text = f.read()

# Extract code blocks
code_blocks = []
def save_code(m):
    code_blocks.append(m.group(1).strip())
    return "%%CODE_%d%%" % len(code_blocks)

code_pattern = re.compile(BT + "python\n(.*?)" + BT, re.DOTALL)
md_text = code_pattern.sub(save_code, text)
print("Code blocks found:", len(code_blocks))

# Check placeholders in md_text
for i in range(len(code_blocks)):
    if "%%CODE_%d%%" % i in md_text:
        print("  placeholder %d: found" % i)
    else:
        print("  placeholder %d: MISSING!" % i)

# Now parse title
lines = md_text.split("\n")
i = 0
N = len(lines)
while i < N and not lines[i].startswith("## "): i += 1
title_end = i
while i < N and not (lines[i].startswith("---") and i > title_end + 2): i += 1
if i < N and lines[i].startswith("---"): i += 1

# Check remaining for placeholders
remaining = "\n".join(lines[i:])
for j in range(len(code_blocks)):
    if "%%CODE_%d%%" % j in remaining:
        print("  in remaining %d: found" % j)
    else:
        print("  in remaining %d: MISSING!" % j)

parts = re.split(r'%%CODE_(\d+)%%', remaining)
print("Parts:", len(parts))
for k, p in enumerate(parts):
    print("  part %d: %d chars, starts with: %s" % (k, len(p), p[:60].strip()))
