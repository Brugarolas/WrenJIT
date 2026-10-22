// Compare with WREN_NO_ENTRY_JIT=1 (loop tier only) and WREN_NO_JIT=1.
class EntryBench {
    static kernel(x) {
        var a = x * 1.0001 + 0.25
        var b = a * a - a / 3.0
        var c = b * 0.75 + a * 0.5
        var d = c * c / (b + 1.0)
        return d + b - a + c / 7.0
    }

    static run(count) {
        var total = 0
        var i = 0
        while (i < count) {
            total = total + kernel(i % 97 + 0.125)
            i = i + 1
        }
        return total
    }
}

EntryBench.run(1000)
var start = System.clock
var checksum = EntryBench.run(1000000)
System.print("checksum: %(checksum)")
System.print("elapsed: %(System.clock - start)")
