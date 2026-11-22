import sys

p = sys.argv[1]
data = open(p, "rb").read()
if data and not data.endswith(b"\n"):
    cut = data.rfind(b"\n") + 1
    open(p, "wb").write(data[:cut])
    print("trimmed", len(data) - cut, "bytes")
