// module_import.wren
//
// Imports several exported bindings from module_exports.wren.

import "./module_exports.wren" for Answer, Greeting, Calculator

System.print(Greeting)
System.print("Answer: %(Answer)")
System.print("20 + 22 = %(Calculator.add(20, 22))")
System.print("6 * 7 = %(Calculator.multiply(6, 7))")
