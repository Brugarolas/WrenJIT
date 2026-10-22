// Source-level class regressions, using the production compilation pipeline.
fn run_class_source(source: &str, expected: &str) {
    let (program, entry, manifest) = compile_wren_module(source);
    for jit_enabled in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(jit_enabled.then(|| TracingJitOptions {
            hot_loop_threshold: 2,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        let value = *vm.modules.last().unwrap().variables.last().unwrap();
        let context = HostContext { heap: &mut vm.heap };
        assert_eq!(context.format_value(value), expected, "jit={jit_enabled}");
    }
}

#[test]
fn source_classes_default_constructor_and_method_signatures() {
    run_class_source(r#"
        class Unicorn {
            isFancy { true }
            name { "Scoop" }
            prance() { "fancy" }
            prance(where) { where }
            prance(where, when) { where + " at " + when }
            rider=(value) { _rider = value }
            rider { _rider }
            implicitName() { name }
            explicitFancy() { this.isFancy }
            -(other) { other + 1 }
            [index] { index + 2 }
            [x, y] { x + y }
            [x, y]=(value) { _rider = value + x + y }
        }
        var u = Unicorn.new()
        var empty = u.rider
        u.rider = "rider"
        var assigned = u.rider
        u[1, 2] = 10
        var result = [empty, assigned, u.rider, u.implicitName(), u.explicitFancy(),
            u.prance(), u.prance("meadow"), u.prance("meadow", "sunset"), u - 1, u[2], u[2, 3]]
    "#, "[null, rider, 13, Scoop, true, fancy, meadow, meadow at sunset, 2, 4, 5]");
}

#[test]
fn source_classes_method_scope_and_implicit_this_in_closures() {
    run_class_source(r#"
        var shadowed = "surrounding"
        var lowercase = "surrounding"
        var Capitalized = "surrounding"
        class Scope {
            shadowed { "object" }
            lowercase { _value == null ? "object" : _value }
            lowercase=(value) { _value = value }
            Capitalized { "object" }
            test() {
                var shadowed = "local"
                var closure = Fn.new { lowercase }
                var values = [shadowed, lowercase, Capitalized, closure.call()]
                var setter = Fn.new { lowercase = "updated" }
                setter.call()
                values.add(lowercase)
                return values
            }
        }
        var values = Scope.new().test()
        var result = [values, lowercase]
    "#, "[[local, object, surrounding, object, updated], surrounding]");
}

#[test]
fn source_classes_named_and_overloaded_constructors() {
    run_class_source(r#"
        class Horse {
            construct new(name, color) {
                _name = name
                _color = color
            }
            construct new(name) {
                _name = name
                _color = "white"
                return
            }
            construct brown(name) {
                _name = name
                _color = "brown"
            }
            description { _name + ":" + _color }
        }
        var result = [Horse.new("Fred", "palomino").description,
            Horse.new("Snow").description, Horse.brown("Dave").description]
    "#, "[Fred:palomino, Snow:white, Dave:brown]");
}

#[test]
fn source_classes_super_initializes_same_object_and_fields_are_private() {
    run_class_source(r#"
        class Parent {
            construct new(name) { _name = name }
            parentName { _name }
            describe() { "parent:" + _name }
            virtual() { this.describe() }
            capturedName { Fn.new { _name } }
        }
        class Middle is Parent {}
        class Child is Middle {
            construct new(name) {
                super(name)
                _name = "child"
            }
            ownName { _name }
            describe() { super.describe() + ":child" }
        }
        var child = Child.new("Fred")
        System.gc()
        var result = [child.parentName, child.ownName, child.describe(), child.virtual(),
            child.capturedName.call(), child is Child, child is Parent, Middle.new().parentName]
    "#, "[Fred, child, parent:Fred:child, parent:Fred:child, Fred, true, true, null]");
}

#[test]
fn source_classes_dynamic_superclass_and_super_in_closures() {
    run_class_source(r#"
        class Base {
            construct new(value) { _value = value }
            value { _value }
            plus(n) { _value + n }
        }
        class Provider { static base { Base } }
        class Child is Provider.base {
            construct new(value) { super(value) }
            plus(n) { super(n) + 1 }
            callback { Fn.new { super.plus(2) } }
        }
        var Alias = Child
        class Leaf is Alias {
            construct new(value) { super(value) }
        }
        var leaf = Leaf.new(10)
        var result = [leaf.value, leaf.plus(2), leaf.callback.call()]
    "#, "[10, 13, 12]");
}

#[test]
fn source_classes_static_fields_are_shared_and_lexically_owned() {
    run_class_source(r#"
        class Foo {
            construct new() {}
            static value { __a }
            static set(value) { __a = value }
            set(value) { __a = value }
            value { __a }
            callback { Fn.new { __a } }
            static callback { Fn.new { __a } }
        }
        class Child is Foo {
            ownValue { __a }
            setOwn(value) { __a = value }
        }
        var initial = Foo.value
        Foo.set("first")
        var first = Foo.new()
        var second = Foo.new()
        first.set("second")
        var child = Child.new()
        child.setOwn("child")
        var instanceCallback = first.callback
        var staticCallback = Foo.callback
        Foo = null
        System.gc()
        var result = [initial, second.value, child.value, child.ownValue,
            instanceCallback.call(), staticCallback.call()]
    "#, "[null, second, second, child, second, second]");
}

#[test]
fn source_classes_fields_default_to_null_and_instances_are_independent() {
    run_class_source(r#"
        class Rectangle {
            area { _width * _height }
            width { _width }
            width=(value) { _width = value }
            height=(value) { _height = value }
        }
        var first = Rectangle.new()
        var second = Rectangle.new()
        var initial = first.width
        first.width = 3
        first.height = 4
        var result = [initial, first.area, second.width]
    "#, "[null, 12, null]");
}

#[test]
fn source_classes_constructor_chains_survive_gc_and_repeated_dispatch() {
    run_class_source(r#"
        class Base {
            construct new(value) { _value = [value] }
            value { _value[0] }
        }
        class Child is Base {
            construct new(value) {
                super(value)
                _extra = [1]
            }
            value { super.value + _extra[0] }
        }
        class Test {
            static run() {
                var total = 0
                var i = 0
                while (i < 100) {
                    var child = Child.new(i)
                    if (i % 10 == 0) System.gc()
                    total = total + child.value
                    i = i + 1
                }
                return total
            }
        }
        var result = Test.run()
    "#, "5050");
}

#[test]
fn source_classes_do_not_inherit_static_methods_or_constructor_factories() {
    for source in [
        "class Base { static value { 1 } }\nclass Child is Base {}\nChild.value\n",
        "class Base { construct named(value) {} }\nclass Child is Base {}\nChild.named(1)\n",
        "class Horse { construct new(name) {} }\nHorse.new()\n",
        "class Root { construct new() { super() } }\nRoot.new()\n",
    ] {
        let (program, entry, manifest) = compile_wren_module(source);
        for jit_enabled in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(jit_enabled.then(TracingJitOptions::default));
            let error = vm.run_module(entry).err().expect(source);
            assert!(error.message.contains("not defined for receiver"), "{error:?}");
        }
    }
}

#[test]
fn source_classes_reject_private_field_access_and_constructor_return_values() {
    let parsed = parse("classes.wren", "class Test {}\nvar a = Test.new()\na._field\n");
    assert!(parsed.has_errors());
    for source in [
        "class Test { static value { _field } }\n",
        "class Test { static set(value) { _field = value } }\n",
        "class Test { construct new() {\nreturn 1\n} }\n",
    ] {
        let parsed = parse("classes.wren", source);
        assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
        let mut hir = Program::new();
        wren_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(&mut hir, "classes.wren", source, "classes", &parsed.module);
        assert!(lowered.has_errors(), "expected a lowering error for {source}");
    }
}
