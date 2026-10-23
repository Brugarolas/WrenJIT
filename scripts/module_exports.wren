// module_exports.wren
//
// Every top-level variable or class is exported by WrenJIT's module system.
// There is intentionally no `export` keyword: module-level bindings are
// available to `import "... " for Name` in another module.

var Answer = 42
var Greeting = "Hello from the exported module!"

class Calculator {
  static add(a, b) {
    return a + b
  }

  static multiply(a, b) {
    return a * b
  }
}
