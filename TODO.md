[ ] **Inlining real de llamadas calientes, especialmente monomorphic Invoke y CallDirect**. Ya tienes parte de la infraestructura especulativa; convertir más llamadas en una única región nativa puede ser probablemente la siguiente gran barbaridad.
[ ] **Trace MemorySSA / forwarding global de loads/stores**. Quitar cargas repetidas de field/cell/list/map y hacer store→load forwarding puede reducir muchísimo el tráfico incluso después de Cranelift.
[ ] **Guard elimination / guard hoisting más agresivo**. Si un heap kind, class, bounds condition o method epoch ya está demostrado para una iteración, pagar el guard repetidamente es desperdicio.
[ ] **Range analysis + BCE + induction variables**. Los loops de arrays/lists son precisamente donde un tracing JIT debería poder acercarse mucho a código nativo sencillo.
[ ] **Allocation elimination, donde tu PEA/SROA ya está creciendo**. Cada objeto que nunca llega a GC es una victoria doble: menos alloc y menos trabajo posterior de collector/barriers.
[ ] **Direct allocation fast path** cuando la ABI de tu GC esté suficientemente estable. No antes.
[ ] **Reducir runtime boundaries restantes**. Cada opcode que todavía sale a Rust merece preguntarse si realmente necesita hacerlo.
[ ] **Especialización de call sites polimórficos pequeños**, por ejemplo 2–4 clases calientes antes de rendirse al generic path.
[ ] **Code layout / branch probabilities / cold blocks**, especialmente ahora que cada vez más tiempo real va a estar dentro del código generado y microdetalles empiezan a importar.