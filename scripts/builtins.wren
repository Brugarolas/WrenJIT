System.print("Hola, Wren!")
System.print("6 * 7 = %(6 * 7)")
System.write("Sin salto de linea")
System.print()

var greet = Fn.new { |name| System.print("Hola, %(name)!") }
greet.call("Andres")
System.gc()
