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

System.print("***** Matrices! *****")

var matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
System.print(matrix) //> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
System.print(matrix[0, 1]) //> 2

System.print("***** Classes! *****")

class Unicorn {
  // Unicorns are always fancy.
  isFancy { true }

  name { "Scoop" }

  prance() {
    System.print("The unicorn prances in a fancy manner!")
  }

  prance(where) {
    System.print("The unicorn prances in %(where).")
  }

  prance(where, when) {
    System.print("The unicorn prances in %(where) at %(when).")
  }

  rider=(value) {
    System.print("I am being ridden by %(value).")
  }

  printIsFancy() {
    System.print("Is fancy? %(this.isFancy)") //> true
  }

  printName() {
    System.print(name) //> Francis (implicit this)
  }

  -(other) {
    System.print("Subtracting %(other) from a unicorn is weird.")
  }

  [index] {
    System.print("Unicorns are not lists!")
  }

  [x, y] {
    System.print("Unicorns are not matrices either!")
  }
}

var myUnicorn = Unicorn.new()
myUnicorn.prance()
myUnicorn.prance("the meadow")
myUnicorn.prance("the meadow", "sunset")

System.print("***** Scoping! *****")

var shadowed = "surrounding"
var lowercase = "surrounding"
var Capitalized = "surrounding"

class Scope {
  shadowed { "object" }
  lowercase { "object" }
  Capitalized { "object" }

  test() {
    var shadowed = "local"

    System.print(shadowed) //> local (local variables have preference over object fields)
    System.print(lowercase) //> object (object fields have preference over top-level variables if they are lowercased)
    System.print(Capitalized) //> surrounding (the top-level variable is used because it is capitalized)
  }
}

System.print("***** Constructors *****")

class Horse {
  construct new(name, color) {
    System.print("My name is " + name + " and I am " + color + ".")
  }
construct brown(name) {
    System.print("My name is " + name + " and I am brown.")
  }
}

var fred = Horse.new("Fred", "palomino")
var dave = Horse.brown("Dave")

System.print("***** Fields, getters and setters *****")

class Rectangle {
    area { _width * _height }

    width=(value) { _width = value }
    height=(value) { _height = value }
}

class Shape {
  construct new() {
    _shape = "none"
  }
}

class Square is Shape {
  construct new() {
    //This will print null!
    //_shape from the parent class is private,
    //we are reading `_shape` from `this`,
    //which has not been set, so returns null.
    System.print("I am a %(_shape)")

    //a local variable, all variables are private
    _width = 10
    var other = Square.new()

    //other._width is not accessible from here,
    //even though we are also a rectangle. The field
    //is private, and other._width is invalid syntax!
  }
}

System.print("***** Statics! *****")

class Foo {
  construct new() {}

  static setFromStatic(a) { __a = a }
  setFromInstance(a) { __a = a }

  static printFromStatic() {
    System.print(__a)
  }

  printFromInstance() {
    System.print(__a)
  }
}

Foo.printFromStatic() //> null

Foo.setFromStatic("first")
Foo.printFromStatic() //> first

var foo1 = Foo.new()
var foo2 = Foo.new()
foo1.setFromInstance("second")
foo2.printFromInstance() //> second

System.print("***** Inheritance! *****")

class FancyUnicorn {
  construct new(name) {
    System.print("My name is " + name + ".")
  }

  fly() {
    System.print("I am flying!")
  }
}

class Pegasus is FancyUnicorn {
  construct new(name) {
    super(name)
  }

  fly() {
    super.fly() //> base method
  }
}

var pegasus = Pegasus.new("Fred") //> My name is Fred
pegasus.fly()

System.print("***** Async/Await! *****")

class AsyncMath {
  static async value(x) {
    return await Promise.resolve(x)
  }

  static async add(a, b) {
    var left = await AsyncMath.value(a)
    var right = await AsyncMath.value(b)
    return left + right
  }
}

var result_fn = await AsyncMath.add(20, 22)
System.print(result_fn) //> 42

var plusOne = async { |value|
  var resolved = await Promise.resolve(value)
  return resolved + 1
}

var result_closure = await plusOne.call(21)
System.print(result_closure) //> 22

System.print("***** Fibers! *****")

var child = Fiber.new {
  123.methodThatDoesNotExist()
}
var parent = Fiber.new {
  child.call()
}
var error = parent.try()

System.print(error)
System.print(parent.error == error) // true
System.print(child.error == error)  // true
System.print(parent.isDone)         // true
System.print(child.isDone)          // true

var fiber = Fiber.new {|first|
  var resumed = Fiber.yield(first + 1)
  return resumed + 2
}

System.print(fiber.call(10)) // 11
System.print(fiber.call(20)) // 22
System.print(fiber.isDone)   // true

System.print("***** Errors! *****")

Fn.new {
  defer {
    System.print("always")
  }

  errdefer {
    System.print("only on error")
  }

  errdefer (error) {
    System.print(error)
  }

  throw "error from funcWithError"
}

Fn.new {
  try {
    throw "boom"
  } catch (error) {
    System.print(error)
  } finally {
    System.print("always")
  }
}