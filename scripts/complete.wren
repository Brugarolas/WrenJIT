// This is a comment.
/* This
    is
    a
    multi-line
   comment. */
/* This is /* a nested */ comment. */

System.print("***** Lists! *****")

var trees = ["cedar", "birch", "oak", "willow"]
System.print(trees[0])      // cedar
System.print(trees[1])      // birch
System.print(trees[-1])     // willow
System.print(trees[-2])     // oak
System.print(trees.count)   // 4

System.print("***** Slices! *****")

System.print(trees[1..2])   // [birch, oak] — inclusive
System.print(trees[1...3])  // [birch, oak] — exclusive
System.print(trees[1..-1])  // [birch, oak, willow]
System.print(trees[3..1])   // [willow, oak, birch] — reversed
System.print(trees[2...2])  // []

System.print("***** Adding elements! *****")

trees[1] = "spruce"
System.print(trees[1]) //> spruce

trees.add("maple")
System.print(trees.count) //> 5

System.print(trees.insert(2, "hickory"))

var letters_first = ["a", "b", "c"]
letters_first.insert(3, "d")   // OK: inserts at end.
System.print(letters_first)    //> [a, b, c, d]
letters_first.insert(-2, "e")  // Counts back from size after insert.
System.print(letters_first)    //> [a, b, c, e, d]

System.print("***** Combining lists! *****")

var letters_second = ["a", "b", "c"]
var other = ["d", "e", "f"]
var combined = letters_second + other
System.print(combined)  //> [a, b, c, d, e, f]

System.print("***** Removing elements! *****")

var letters_third = ["a", "b", "c", "d"]
letters_third.removeAt(1)
System.print(letters_third) //> [a, c, d]
letters_third.remove("a")
System.print(letters_third) //> [c, d]

System.print(letters_third.removeAt(1)) //> c

System.print(letters_third.remove("not found")) //> null

trees.clear()
System.print(trees) //> []

System.print("***** Maps! *****")

var trees_map = {
    "maple":  "Sugar Maple (Acer Saccharum)",
    "larch":  "Alpine Larch (Larix Lyallii)",
    "oak":    "Red Oak (Quercus Rubra)",
    "fir":    "Fraser Fir (Abies Fraseri)"
}

System.print(trees_map)

var some_capitals = {}
some_capitals["Georgia"] = "Atlanta"
some_capitals["Idaho"] = "Boise"
some_capitals["Maine"] = "Augusta"

System.print(some_capitals)

System.print(some_capitals["Idaho"]) //> Boise

var capitals = {"Georgia": null}

System.print(capitals["Georgia"]) //> null (though key exists)
System.print(capitals["Idaho"])   //> null
System.print(capitals.containsKey("Georgia")) //> true
System.print(capitals.containsKey("Idaho"))   //> false

capitals.remove("Maine")
System.print(capitals.containsKey("Maine")) //> false

System.print(capitals.remove("Georgia")) //> null

capitals.clear()
System.print(capitals.count) //> 0

var birds = {
    "Arizona": "Cactus wren",
    "Hawaii": "Nēnē",
    "Ohio": "Northern Cardinal"
}

System.print("***** Iteration with entries over a map! *****")
for (bird in birds) {
    System.print("The state bird of %(bird.key) is %(bird.value)")
}

System.print("***** Iteration with keys over a map! *****")
for (state in birds.keys) {
    System.print("The state bird of %(state) is " + birds[state])
}

System.print("***** Random! *****")

var random = Random.new()
System.print(random.int(3, 10))
System.print(random.int(4))

System.print("***** Getters! *****")

System.print("string".count)    //> 6
System.print((1..10).count)     //> 10
System.print((1..10).min)       //> 1
System.print(1.23.sin)          //> 0.9424888019317
System.print([1, 2, 3].isEmpty) //> false