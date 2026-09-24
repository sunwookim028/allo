# ARC2HS — Agent-Driven Construction of Computer Hardware Simulation

Reference for **task 4** in `claude_simulator.md` (the three-phase DES rewrite option).
A colleague's **event-based hardware simulator** (also OpenMP-adjacent, but the core loop
shown is single-threaded). Its goal is an event-based HW simulator bound to a controlled
subset of natural language so an LLM can generate hardware flow as Models + Events via its
API. For us it is interesting as a **proven design that gives determinism, combinational
loops, and deadlock detection structurally** — the things our JIT simulator lacks.

---

## Why this matters to us (takeaways from the 2026-07-26 analysis)

- **Three-phase time step = the model we kept circling.** setup → **block (converge)** →
  commit. The block phase is the fixed-point solver for same-time interactions
  (backpressure, combinational logic). This is exactly what our async per-PE-clock model
  cannot do.
- **The monotonicity trick is the gem.** During `block()`, pressure is *monotone*
  (same-time NRDY only *adds* backpressure; RDY *releases* are deferred to `commit()`), so
  the converge loop provably terminates — the `kMaxBlockIter=1000` cap is a bug-catcher,
  not the real bound. A true unresolvable combinational loop hits the cap → **errors with
  a diagnostic instead of hanging**. That is our missing deadlock detection.
- **Lazy cancellation** (mark cancelled, skip everywhere, keep the Event alive in the
  owning model's `trace_` until `cleanup()`) is how it retracts a same-cycle decision
  without dangling pointers.
- **The leaf model's `block()` is trivial** (pure observation latching — see LocalModel);
  all converge complexity lives in the interconnect (router). → strong evidence for a
  **hybrid**: keep our JIT'd PE bodies as threads, adopt three-phase converge only for the
  **channel/interconnect layer**. The `LocalModel` inject/stall/retry protocol is the shape
  our `put`/`get` hook would take.
- **It is a control/timing simulator, not a datapath one.** Every model here routes /
  handshakes; `payload` data is carried opaquely, never computed on (no arithmetic-in-model
  example anywhere). So it gives us the interconnect discipline but *no* template for a
  compute PE — which is exactly why the hybrid (JIT'd compute + converge interconnect) is
  the only path that accounts for the datapath.
- **Open unknown before adopting:** how a PE gets "stepped" — thread-per-PE + a global
  per-cycle barrier (evolution of what we have) vs. coroutine/state-machine PEs (rewrite).
  This reference answers it as FSM-models (the expensive answer); the hybrid sidesteps it.

---

## Spec (verbatim)

The goal of this project is an Event Based Hardware Simulator that is bound with a subset of
natural language (we designed) such that when input to an LLM (along with other context), it
can "accurately" generate descriptions of hardware flow as Models and Events using the API we
expose.

### Discrete Event Simulator

The Discrete Event Simulator drives a global priority queue of Event pointers ordered by
simulation time. At each time step the simulator executes three phases — setup, block, and
commit — across every registered model before advancing to the next time.

### Three-Phase Time Step

**Setup** — The simulator clears every model's `active_queue_`, pops all events at time `t`
from the global priority queue, and routes each to the target model's `active_queue_` via the
`routing_table_`.

**Block (converge)** — Each model's `block()` is called. Inside `block()`, models inspect
their `active_queue_`, resolve same-time interactions (backpressure, combinational logic,
etc.), and may:

- Emit same-time blocking events via `sim.schedule_blocking()` keeping the same payload as the
  event that was blocked — these events are routed directly into the global priority queue and
  with a reassigning of the blocking bit.
- Cancel events via `sim.cancel_event()` - the event and all children events that relate to it
  are marked cancelled and detached from parent/child relationships.
- Back-pressure / blocking events will cause cancellation, and leave a note in the module
  itself such that it needs to reschedule that non-blocking event in the commit phase.

The block phase repeats until convergence: a full pass over all models produces no new
blocking events and no cancellations. A safety cap (set to 1000 iterations for default)
prevents infinite loops from model logic bugs at which point we error out. If the simulator
does not converge, it is likely due to an error in the model designs.

**Commit** — Once the block phase converges, each model's `commit()` is called. Models process
the settled active state and emit future events (time > now) via `sim.schedule()`, which
inserts them into the global priority queue as well as handling rescheduling. Events here have
the blocking bit set to 0.

### The core loop

```cpp
void run(sim_time until) {
    while (!queue_.empty()) {
        while (!queue_.empty() && queue_.top()->cancelled)
            queue_.pop();
        if (queue_.empty()) break;

        sim_time t = queue_.top()->time;
        if (t > until) break;
        now_ = t;

        // ── Setup ──────────────────────────────────────────────
        for (auto& model : models_)
            model->setup();                     // clears active_queue_ + cancel_notifications_

        while (!queue_.empty()) {
            if (queue_.top()->cancelled) { queue_.pop(); continue; }
            if (queue_.top()->time != t) break;
            Event* e = queue_.top();
            queue_.pop();
            e->blocking = 0;                    // normal (non-blocking) event
            auto it = routing_table_.find(e->name);
            if (it != routing_table_.end())
                it->second->push_active(e);
        }

        // ── Block (converge) ───────────────────────────────────
        bool changed = true;
        int iter = 0;
        while (changed && iter < kMaxBlockIter) {
            activity_ = false;
            for (auto& model : models_)
                model->block(*this);            // may call schedule_blocking / cancel_event

            bool new_events = false;            // drain new time-t events produced during block
            while (!queue_.empty()) {
                if (queue_.top()->cancelled) { queue_.pop(); continue; }
                if (queue_.top()->time != t) break;
                Event* e = queue_.top();
                queue_.pop();
                e->blocking = 1;                // blocking event
                auto it = routing_table_.find(e->name);
                if (it != routing_table_.end())
                    it->second->push_active(e);
                new_events = true;
            }

            changed = new_events || activity_;  // activity_ set by cancel_event
            ++iter;
        }

        // ── Commit ─────────────────────────────────────────────
        for (auto& model : models_)
            model->commit(*this);               // inspect cancel_notifications_, reschedule, emit future events
    }
}
```

### Events

```cpp
struct Event {
    sim_time time = 0;
    std::string name;
    bit* payload = nullptr;
    bit  blocking = 0;
    int64_t payload_size = 0;
    std::vector<bit> payload_owned;
    bool cancelled = false;
};
```

- **time** — The simulation time at which this event is scheduled.
- **name** — Identifies the event type. The simulator's `routing_table_` maps names to handler models.
- **payload / payload_size** — Optional binary payload. The producer allocates it and guarantees it stays valid until the event is processed; the consumer frees it after use. The Event object itself is owned only by its creating model's `trace_`.
- **blocking** - A single bit that is bit 0 if nonblocking and bit 1 if blocking. This value is assigned by the simulator `sim` when the event gets scheduled.
- **payload_owned** — Optional owned payload storage (avoids manual allocation when the creator wants the Event to own its data).
- **cancelled** — Set to true by `cancel_event()`. All queues (global and per-model) lazily skip cancelled events. The Event object stays alive in its owning model's `trace_` so that no raw pointer ever dangles.

### Cancellation

Cancellation is lazy and safe. Calling `sim.cancel_event(e)`:

- Sets `e->cancelled = true`.
- Notifies the target model (looked up via `routing_table_`) by appending `e` to its `cancel_notifications_`. This lets the model reschedule the work during `commit()`.
- Detaches `e` from its parent/child relationships in the owning model.
- Sets the `activity_` flag so the block convergence loop continues.

The Event object is not freed — it remains in the owning model's `trace_` so that raw pointers
held by the global queue or any model's `active_queue_` never dangle. Cancelled events are
skipped wherever they appear (global queue skips on pop; models should check `e->cancelled` /
`sim.is_cancelled(e)`). After the simulation ends, call `sim.cleanup()` to free cancelled event
memory from all model traces.

### Models

Models inherit from `BaseModel` and implement three virtual methods:

| Method | Phase | Purpose |
|---|---|---|
| `setup()` | Setup | Clears `active_queue_`. Override to add per-step initialization. |
| `block(Simulator& sim)` | Block | Resolve same-time interactions. `sim.schedule_blocking()` to emit same-time events, `sim.cancel_event()` to cancel. Called repeatedly until convergence. |
| `commit(Simulator& sim)` | Commit | Generate future events from the settled state. `sim.schedule()` to emit events at future times. Inspect `cancelled_notifications()` to reschedule cancelled events. |

Additionally, models must define `accepted_event_names()` — the set of event names this model
handles; the simulator builds the `routing_table_` from these at registration. Each model owns
its events via `trace_` (`vector<unique_ptr<Event>>`) and maintains `parent_of_` / `children_of_`
maps for causal tracking.

### Simulator API

| Method | Description |
|---|---|
| `add_model(ModelPtr model)` | Register a model; its `accepted_event_names()` are added to the routing table. |
| `schedule(creator, event)` | Schedule a future event into the global priority queue. Stored in the creator's trace. |
| `schedule(creator, parent, event)` | Same, with explicit parent for causal tracking. |
| `schedule_blocking(creator, event)` | Schedule a same-time event (enforces `time=now()`). The converge loop pops it, sets `blocking=1`, routes it to the target's active queue. |
| `schedule_blocking(creator, parent, event)` | Same, with explicit parent. |
| `cancel_event(Event* e)` | Mark cancelled; notify target model (reschedule in `commit()`); set `activity_`. |
| `is_cancelled(Event* e)` | Check if an event has been cancelled. |
| `now()` | Current simulation time. |
| `run(sim_time until)` | Run the simulation until the given time. |
| `cleanup()` | Free cancelled event memory from all model traces (only safe when no queue holds cancelled pointers). |

### Analogue

Many interconnected FSMs (Models). All FSMs communicate by pushing new events to the DES, which
dispatches them to the correct models. Each Model has internal state and, based on it, either
resolves same-time interactions (block) or produces future events (commit). The three-phase
design ensures combinational/backpressure logic fully settles before any model commits new work.

### Build / Run (abridged)

Out-of-source CMake build (`cmake -S .. -B . && cmake --build . -- -j`). Options:
`-DBUILD_EXAMPLES=ON`, `-DEXAMPLE=noc_example`, `-DBUILD_TESTS=ON`, `-DBUILD_PYTHON_MODULE=ON`
(pybind11 → `import DES`). Examples build into `build/` (`./build/noc_example`). Frontend: conda
env from `environment.yml` (`conda activate transit`), `cd frontend && npm install`, then
`python visualizer.py ___.log` (NoC visualizer for the `noc_router.cpp` playground example).

### NoC Sim Invariants (runtime asserts)

**Router (NocRouterModel):** single-occupancy input ports (≤1 VAL/input/cycle); single-occupancy
output ports (≤1 VAL/output/cycle, enforced by RRAStreamMux2/4 + per-port counter); deterministic
crossbar routing — NESW inputs route only to the opposite direction or to L; L input routes to any
output:

| Input | Valid Outputs |
|---|---|
| N (0) | S (1) or L (4) |
| S (1) | N (0) or L (4) |
| E (2) | W (3) or L (4) |
| W (3) | E (2) or L (4) |
| L (4) | N,S,E,W, or L |

Two packets cannot exit the same output in one cycle (mux arbitrates; loser is NRDY'd).
Back-pressured inputs are not forwarded — they are rescheduled for next cycle.

**Local (LocalModel):** ≤1 injection/cycle (only when the previous in-flight packet is RDY'd and
not stalled); ≤1 delivery/cycle; never both `has_in_flight_` and `stalled_` (NRDY re-queues the
in-flight packet + sets stalled; RDY clears stalled). Consequence: a local can inject (L input →
NESW) the same cycle the router delivers to it (NESW → L output) — different ports, no conflict;
but two packets can't both be delivered to one local (LoMux picks one), and a local can't inject two.

---

## Model code (verbatim)

### noc_router.cpp

```cpp
// Noc Router Model — mirrors noc_impl.hpp SwitchNode architecture
// NESWiRoute / LiRoute → internal streams → RRAStreamMux2/4 per output
#include "src/noc_router.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════
// RRAStreamMux2
// ═══════════════════════════════════════════════════════════════════════════
bool RRAStreamMux2::step(std::vector<NoCRequest>& q0, std::vector<NoCRequest>& q1,
                         NoCRequest& winner, int& winner_input) {
    if (!valid0 && !q0.empty()) { pkt0 = q0.front(); q0.erase(q0.begin()); valid0 = true; }
    if (!valid1 && !q1.empty()) { pkt1 = q1.front(); q1.erase(q1.begin()); valid1 = true; }

    if (valid0 && !valid1) {
        winner = pkt0; valid0 = false; winner_input = 0;
        rotate = 1; return true;
    } else if (!valid0 && valid1) {
        winner = pkt1; valid1 = false; winner_input = 1;
        rotate = 0; return true;
    } else if (valid0 && valid1) {
        if (rotate == 0) {
            winner = pkt0; valid0 = false; winner_input = 0;
        } else {
            winner = pkt1; valid1 = false; winner_input = 1;
        }
        rotate = !rotate; return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// RRAStreamMux4
// ═══════════════════════════════════════════════════════════════════════════
bool RRAStreamMux4::step(std::vector<NoCRequest>& q0, std::vector<NoCRequest>& q1,
                         std::vector<NoCRequest>& q2, std::vector<NoCRequest>& q3,
                         NoCRequest& winner, int& winner_input) {
    std::vector<NoCRequest>* qs[4] = {&q0, &q1, &q2, &q3};
    for (int i = 0; i < 4; ++i)
        if (!(valid & (1 << i)) && !qs[i]->empty()) {
            pkts[i] = qs[i]->front(); qs[i]->erase(qs[i]->begin());
            valid |= static_cast<uint8_t>(1 << i);
        }
    if (valid == 0) return false;

    uint8_t req = static_cast<uint8_t>(((valid >> rotate) | (valid << (4 - rotate))) & 0xF);
    int grt = 0;
    for (int i = 0; i < 4; ++i) if (req & (1 << i)) { grt = i; break; }
    int sel = (grt + rotate) & 3;

    winner = pkts[sel]; winner_input = sel;
    valid &= static_cast<uint8_t>(~(1 << sel));
    rotate = static_cast<uint8_t>((rotate + 1) & 3);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructors
// ═══════════════════════════════════════════════════════════════════════════
NocRouterModel::NocRouterModel()
    : NocRouterModel({"N","S","E","W","L"}, {"N","S","E","W","L"}, RouterIds{0,0}) {}

NocRouterModel::NocRouterModel(int x, int y)
    : NocRouterModel({"N","S","E","W","L"}, {"N","S","E","W","L"}, RouterIds{x,y}) {}

NocRouterModel::NocRouterModel(std::vector<std::string> upstream_ids,
                               std::vector<std::string> downstream_ids,
                               RouterIds ids)
    : bfu_id_(ids.bfu_id)
    , group_id_(ids.group_id)
{
    IRV_.resize(kNumPorts, 0);
    REV_.resize(kNumPorts, 0);
    BPV_.resize(kNumPorts, 0);
    WEV_.resize(kNumPorts, 0);
    input_values_.resize(kNumPorts);

    scheduled_output_events_.fill(nullptr);
    mux_last_winner_input_.fill(-1);
    nrdy_outstanding_.fill(false);

    const std::string self_coord =
        "(" + std::to_string(bfu_id_) + "," + std::to_string(group_id_) + ")";
    const std::string self_r = "R" + self_coord;
    const std::string self_l = "L" + self_coord;

    router_label_ = "ROUTER" + self_coord;

    auto coord_str = [](int b, int g) {
        return "(" + std::to_string(b) + "," + std::to_string(g) + ")";
    };
    auto wire = [](const std::string& from, const std::string& to,
                   const std::string& type) {
        return from + ">" + to + ":" + type;
    };
    auto nbr = [&](const std::string& port) -> std::pair<int,int> {
        if (port == "N") return {bfu_id_, group_id_ - 1};
        if (port == "S") return {bfu_id_, group_id_ + 1};
        if (port == "E") return {bfu_id_ + 1, group_id_};
        if (port == "W") return {bfu_id_ - 1, group_id_};
        return {bfu_id_, group_id_};
    };

    for (size_t i = 0; i < upstream_ids.size() && i < kNumPorts; ++i) {
        const std::string& port = upstream_ids[i];
        if (port == "L") {
            upstream_val_names_.push_back(wire(self_l, self_r, "VAL"));
            upstream_nrdy_names_.push_back(wire(self_r, self_l, "NRDY"));
            upstream_rdy_names_.push_back(wire(self_r, self_l, "RDY"));
        } else {
            auto [nb, ng] = nbr(port);
            std::string nb_r = "R" + coord_str(nb, ng);
            upstream_val_names_.push_back(wire(nb_r, self_r, "VAL"));
            upstream_nrdy_names_.push_back(wire(self_r, nb_r, "NRDY"));
            upstream_rdy_names_.push_back(wire(self_r, nb_r, "RDY"));
        }
    }
    for (size_t i = 0; i < downstream_ids.size() && i < kNumPorts; ++i) {
        const std::string& port = downstream_ids[i];
        if (port == "L") {
            downstream_nrdy_names_.push_back(wire(self_l, self_r, "NRDY"));
            downstream_rdy_names_.push_back(wire(self_l, self_r, "RDY"));
            downstream_val_names_.push_back(wire(self_r, self_l, "VAL"));
        } else {
            auto [nb, ng] = nbr(port);
            std::string nb_r = "R" + coord_str(nb, ng);
            downstream_nrdy_names_.push_back(wire(nb_r, self_r, "NRDY"));
            downstream_rdy_names_.push_back(wire(nb_r, self_r, "RDY"));
            downstream_val_names_.push_back(wire(self_r, nb_r, "VAL"));
        }
    }

    static const char* port_tags[kNumPorts] = {"N", "S", "E", "W", "L"};
    for (size_t i = 0; i < kNumPorts; ++i)
        mux_tick_names_[i] = self_r + ":MUX." + port_tags[i];
}

// ═══════════════════════════════════════════════════════════════════════════
// accepted_event_names
// ═══════════════════════════════════════════════════════════════════════════
std::unordered_set<std::string> NocRouterModel::accepted_event_names() const {
    std::unordered_set<std::string> s;
    for (const auto& n : upstream_val_names_)    s.insert(n);
    for (const auto& n : downstream_rdy_names_)  s.insert(n);
    for (const auto& n : downstream_nrdy_names_) s.insert(n);
    for (const auto& n : mux_tick_names_)         s.insert(n);
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// Routing helpers
// ═══════════════════════════════════════════════════════════════════════════
bool NocRouterModel::intercept(const NoCRequest& pkt) const {
    return get_group_id(pkt.dst) == static_cast<uint32_t>(group_id_) &&
           get_intra_group_bfu_id(pkt.dst) == static_cast<uint32_t>(bfu_id_);
}

size_t NocRouterModel::NESWiRoute(size_t in_port, const NoCRequest& pkt) {
    if (intercept(pkt)) {
        switch (in_port) {
            case 0: NiL_.push_back(pkt); break;
            case 1: SiL_.push_back(pkt); break;
            case 2: EiL_.push_back(pkt); break;
            case 3: WiL_.push_back(pkt); break;
        }
        return 4;
    }
    switch (in_port) {
        case 0: NiS_.push_back(pkt); return 1;
        case 1: SiN_.push_back(pkt); return 0;
        case 2: EiW_.push_back(pkt); return 3;
        case 3: WiE_.push_back(pkt); return 2;
    }
    return 4;
}

size_t NocRouterModel::LiRoute(const NoCRequest& pkt) {
    const uint32_t dg = get_group_id(pkt.dst);
    const uint32_t db = get_intra_group_bfu_id(pkt.dst);
    const uint32_t mg = static_cast<uint32_t>(group_id_);
    const uint32_t mb = static_cast<uint32_t>(bfu_id_);
    if (dg < mg) { LiN_.push_back(pkt); return 0; }
    if (dg > mg) { LiS_.push_back(pkt); return 1; }
    if (db > mb) { LiE_.push_back(pkt); return 2; }
    if (db < mb) { LiW_.push_back(pkt); return 3; }
    return 4;
}

size_t NocRouterModel::compute_output_port(size_t in_port, const NoCRequest& pkt) const {
    if (in_port < 4) {
        if (intercept(pkt)) return 4;
        switch (in_port) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
        }
    } else {
        const uint32_t dg = get_group_id(pkt.dst);
        const uint32_t db = get_intra_group_bfu_id(pkt.dst);
        const uint32_t mg = static_cast<uint32_t>(group_id_);
        const uint32_t mb = static_cast<uint32_t>(bfu_id_);
        if (dg < mg) return 0;
        if (dg > mg) return 1;
        if (db > mb) return 2;
        if (db < mb) return 3;
        return 4;
    }
    return 4;
}

size_t NocRouterModel::input_port_for_event(const des::Event& e) const {
    for (size_t i = 0; i < upstream_val_names_.size() && i < kNumPorts; ++i)
        if (e.name == upstream_val_names_[i]) return i;

    return kNoPort;
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-destructive arbitration (reads mux rotation without modifying it)
// ═══════════════════════════════════════════════════════════════════════════
size_t NocRouterModel::arbitrate_output(size_t out_port,
                                         const std::vector<size_t>& contenders) const {
    if (contenders.empty()) return kNoPort;
    if (contenders.size() == 1) return contenders[0];

    if (out_port < 4) {
        const RRAStreamMux2* mux = nullptr;
        switch (out_port) {
            case 0: mux = &NoMux_; break;
            case 1: mux = &SoMux_; break;
            case 2: mux = &EoMux_; break;
            case 3: mux = &WoMux_; break;
        }
        size_t port0 = kMux2SrcPort[out_port][0];
        size_t port1 = kMux2SrcPort[out_port][1];
        bool has0 = false, has1 = false;
        for (size_t c : contenders) {
            if (c == port0) has0 = true;
            if (c == port1) has1 = true;
        }
        if (has0 && has1)
            return (mux->rotate == 0) ? port0 : port1;
        return has0 ? port0 : port1;
    } else {
        uint8_t req = 0;
        for (size_t c : contenders)
            for (int i = 0; i < 4; ++i)
                if (kMux4SrcPort[i] == c) req |= static_cast<uint8_t>(1 << i);

        uint8_t rot = LoMux_.rotate;
        uint8_t rot_req = static_cast<uint8_t>(((req >> rot) | (req << (4 - rot))) & 0xF);
        int grt = 0;
        for (int i = 0; i < 4; ++i)
            if (rot_req & (1 << i)) { grt = i; break; }
        int sel = (grt + rot) & 3;
        return kMux4SrcPort[sel];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal queue mapping (for re-injection of cancelled mux outputs)
// ═══════════════════════════════════════════════════════════════════════════
std::vector<NoCRequest>* NocRouterModel::internal_queue_for(size_t out_port, int mux_input) {
    if (out_port < 4) {
        size_t src = kMux2SrcPort[out_port][mux_input];
        switch (out_port) {
            case 0: return (src == 1) ? &SiN_ : &LiN_;
            case 1: return (src == 0) ? &NiS_ : &LiS_;
            case 2: return (src == 3) ? &WiE_ : &LiE_;
            case 3: return (src == 2) ? &EiW_ : &LiW_;
        }
    } else {
        switch (mux_input) {
            case 0: return &NiL_;
            case 1: return &SiL_;
            case 2: return &EiL_;
            case 3: return &WiL_;
        }
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// run_mux — step the output mux for out_port
// Saves Event* to scheduled_output_events_ and winner source to
// mux_last_winner_input_ for re-injection on cancel.
// ═══════════════════════════════════════════════════════════════════════════
void NocRouterModel::run_mux(size_t out_port, des::Simulator& sim) {
    const des::sim_time t = sim.now();

    if (BPV_[out_port]) return;
    if (last_mux_output_time_[out_port] == t && t != 0) {
        sim.schedule(this, transit_utils::make_event(mux_tick_names_[out_port], t + 1));
        return;
    }

    NoCRequest winner{};
    int winner_input = -1;
    bool has_winner = false;
    size_t src_port = 0;

    if (out_port < 4) {
        RRAStreamMux2* mux = nullptr;
        std::vector<NoCRequest>* q0 = nullptr;
        std::vector<NoCRequest>* q1 = nullptr;
        switch (out_port) {
            case 0: mux = &NoMux_; q0 = &SiN_; q1 = &LiN_; break;
            case 1: mux = &SoMux_; q0 = &NiS_; q1 = &LiS_; break;
            case 2: mux = &EoMux_; q0 = &WiE_; q1 = &LiE_; break;
            case 3: mux = &WoMux_; q0 = &EiW_; q1 = &LiW_; break;
        }
        has_winner = mux->step(*q0, *q1, winner, winner_input);
        if (has_winner) src_port = kMux2SrcPort[out_port][winner_input];
        if (has_winner && (mux->has_pending() || !q0->empty() || !q1->empty()))
            sim.schedule(this, transit_utils::make_event(mux_tick_names_[out_port], t + 1));
    } else {
        has_winner = LoMux_.step(NiL_, SiL_, EiL_, WiL_, winner, winner_input);
        if (has_winner) src_port = kMux4SrcPort[winner_input];
        if (has_winner && (LoMux_.has_pending() ||
            !NiL_.empty() || !SiL_.empty() || !EiL_.empty() || !WiL_.empty()))
            sim.schedule(this, transit_utils::make_event(mux_tick_names_[4], t + 1));
    }

    if (!has_winner) return;
    last_mux_output_time_[out_port] = t;
    WEV_[out_port] = 1;
    mux_last_winner_input_[out_port] = winner_input;

    output_values_[out_port] = winner;
    const des::sim_time sched_t = t + 1;
    des::Event* val_evt = sim.schedule(this, transit_utils::make_event(
        downstream_val_names_[out_port], sched_t,
        reinterpret_cast<uint8_t*>(&output_values_[out_port]),
        static_cast<int64_t>(sizeof(NoCRequest))));
    scheduled_output_events_[out_port] = val_evt;
    ++output_val_count_[out_port];

    const uint32_t d_bfu   = get_intra_group_bfu_id(winner.dst);
    const uint32_t d_group = get_group_id(winner.dst);
    transit_utils::log_event(sched_t, router_label_, downstream_val_names_[out_port],
        std::make_pair(static_cast<int>(d_bfu), static_cast<int>(d_group)),
        static_cast<int>(winner.data), transit_utils::LogAction::PRODUCED);

    if (out_port == 4) {
        transit_utils::record_delivery(sched_t,
            std::make_pair(static_cast<int>(d_bfu), static_cast<int>(d_group)),
            static_cast<int>(winner.data));
    }

    sim.schedule(this, transit_utils::make_event(upstream_rdy_names_[src_port], sched_t));
    nrdy_outstanding_[src_port] = false;
    transit_utils::log_event(t, router_label_, upstream_rdy_names_[src_port],
        std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// setup — reset per-step state
// ═══════════════════════════════════════════════════════════════════════════
void NocRouterModel::setup() {
    BaseModel::setup();
    input_event_.fill(nullptr);
    input_route_target_.fill(kNoPort);
    bp_sent_upstream_.fill(false);
    got_downstream_nrdy_.fill(false);
    got_downstream_rdy_.fill(false);
    mux_tick_pending_.fill(false);
    block_scan_idx_ = 0;
    input_val_count_.fill(0);
    output_val_count_.fill(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// block — inspect current_events_, detect contention, send BP, cancel
// Called repeatedly until convergence.
// ═══════════════════════════════════════════════════════════════════════════
void NocRouterModel::block(des::Simulator& sim) {
    // ── 1. Process new events since last block() call ────────────────────
    for (size_t idx = block_scan_idx_; idx < current_events_.size(); ++idx) {
        des::Event* e = current_events_[idx];
        if (e->cancelled) continue;

        // MUX_TICK
        bool handled = false;
        for (size_t i = 0; i < kNumPorts; ++i) {
            if (e->name == mux_tick_names_[i]) {
                mux_tick_pending_[i] = true;
                handled = true;
                break;
            }
        }
        if (handled) continue;

        // RDY from downstream → record release observation for commit()
        for (size_t i = 0; i < downstream_rdy_names_.size(); ++i) {
            if (e->name == downstream_rdy_names_[i]) {
                got_downstream_rdy_[i] = true;
                transit_utils::log_event(sim.now(), router_label_, e->name,
                    std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
                handled = true;
                break;
            }
        }
        if (handled) continue;

        // NRDY from downstream → record block observation, cancel scheduled output
        for (size_t i = 0; i < downstream_nrdy_names_.size(); ++i) {
            if (e->name == downstream_nrdy_names_[i]) {
                got_downstream_nrdy_[i] = true;
                transit_utils::log_event(sim.now(), router_label_, e->name,
                    std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);

                if (scheduled_output_events_[i] &&
                    !scheduled_output_events_[i]->cancelled) {
                    sim.cancel_event(scheduled_output_events_[i]);
                }
                handled = true;
                break;
            }
        }
        if (handled) continue;

        // VAL from upstream → latch to input port, compute route
        size_t port = input_port_for_event(*e);
        if (port != kNoPort && port < kNumPorts && input_event_[port] == nullptr) {
            input_event_[port] = e;
            IRV_[port] = 1;
            REV_[port] = 1;
            if (e->payload && e->payload_size >= static_cast<int64_t>(sizeof(NoCRequest)))
                std::memcpy(&input_values_[port], e->payload, sizeof(NoCRequest));
            else
                input_values_[port] = NoCRequest{};

            input_route_target_[port] = compute_output_port(port, input_values_[port]);

            ++input_val_count_[port];
            assert(input_val_count_[port] <= 1 &&
                "INV: at most 1 VAL per input port per cycle");
            if (port < 4) {
                assert((input_route_target_[port] == (port ^ 1) ||
                        input_route_target_[port] == 4) &&
                    "INV: NESW input must route to opposite direction or L");
            }

            const uint32_t bfu   = get_intra_group_bfu_id(input_values_[port].dst);
            const uint32_t group = get_group_id(input_values_[port].dst);

            transit_utils::log_event(sim.now(), router_label_, e->name,
                std::make_pair(static_cast<int>(bfu), static_cast<int>(group)),
                static_cast<int>(input_values_[port].data),
                transit_utils::LogAction::CONSUMED);
        }
    }
    block_scan_idx_ = current_events_.size();

    // ── 2. Un-latch inputs whose VAL event was cancelled externally ──────
    for (size_t i = 0; i < kNumPorts; ++i) {
        if (input_event_[i] && input_event_[i]->cancelled) {
            input_event_[i] = nullptr;
            input_route_target_[i] = kNoPort;
            IRV_[i] = 0;
            REV_[i] = 0;
        }
    }

    // ── 3. Contention detection + BP emission ────────────────────────────
    for (size_t o = 0; o < kNumPorts; ++o) {
        std::vector<size_t> contenders;
        for (size_t i = 0; i < kNumPorts; ++i) {
            if (input_event_[i] && !input_event_[i]->cancelled &&
                input_route_target_[i] == o && !bp_sent_upstream_[i])
                contenders.push_back(i);
        }

        if (contenders.empty()) continue;

        // Monotone block transfer: same-time NRDY observations can only add
        // pressure during block(). RDY releases are resolved in commit().
        const bool blocked_now = (BPV_[o] != 0) || got_downstream_nrdy_[o];
        if (blocked_now) {
            for (size_t c : contenders) {
                sim.schedule_blocking(this,
                    transit_utils::make_event(upstream_nrdy_names_[c], sim.now()));
                transit_utils::log_event(sim.now(), router_label_,
                    upstream_nrdy_names_[c],
                    std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
                bp_sent_upstream_[c] = true;
                nrdy_outstanding_[c] = true;
            }
        } else if (contenders.size() > 1) {
            size_t winner = arbitrate_output(o, contenders);
            for (size_t c : contenders) {
                if (c == winner) continue;
                sim.schedule_blocking(this,
                    transit_utils::make_event(upstream_nrdy_names_[c], sim.now()));
                transit_utils::log_event(sim.now(), router_label_,
                    upstream_nrdy_names_[c],
                    std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
                bp_sent_upstream_[c] = true;
                nrdy_outstanding_[c] = true;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// commit — forward winners, re-inject cancelled outputs, reschedule, NFA
// ═══════════════════════════════════════════════════════════════════════════
void NocRouterModel::commit(des::Simulator& sim) {
    const des::sim_time t = sim.now();

    // Resolve downstream flow-control observations after block convergence.
    // NRDY dominates when both are observed in the same timestep.
    for (size_t o = 0; o < kNumPorts; ++o) {
        if (got_downstream_nrdy_[o]) BPV_[o] = 1;
        else if (got_downstream_rdy_[o]) BPV_[o] = 0;
    }

    // ── Re-inject packets whose downstream VAL was cancelled (BP'd) ──────
    for (size_t o = 0; o < kNumPorts; ++o) {
        if (scheduled_output_events_[o] &&
            scheduled_output_events_[o]->cancelled) {
            int mi = mux_last_winner_input_[o];
            if (mi >= 0) {
                auto* q = internal_queue_for(o, mi);
                if (q) q->insert(q->begin(), output_values_[o]);
            }
            scheduled_output_events_[o] = nullptr;
            mux_last_winner_input_[o] = -1;
        }
    }

    // ── NFA transition ──────────────────────────────────────────────────
    const bool any_ready = transit_utils::all_reduce_or(IRV_);
    const bool blocked   = transit_utils::all_reduce_or(BPV_);

    NFAState next;
    if      (any_ready && !blocked) next = NFAState::EXEC;
    else if (any_ready && blocked)  next = NFAState::BOTH;
    else if (blocked)               next = NFAState::BLOCK;
    else                            next = NFAState::STARVE;

    transit_utils::log_event(t, router_label_,
        std::string(state_str(state_)) + "->" + state_str(next),
        std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);

    const bool state_changed = (next != state_);

    // ── Route non-blocked winner inputs into internal stream queues ──────
    if (next == NFAState::EXEC || next == NFAState::BOTH) {
        for (size_t i = 0; i < kNumPorts; ++i) WEV_[i] = 0;

        for (size_t i = 0; i < kNumPorts; ++i) {
            if (IRV_[i] == 0) continue;
            if (bp_sent_upstream_[i]) continue; // this input lost arbitration

            size_t out = input_route_target_[i];
            if (out == kNoPort) continue;
            if (BPV_[out]) continue; // output blocked

            if (i < 4) {
                NESWiRoute(i, input_values_[i]);
            } else {
                size_t target = LiRoute(input_values_[i]);
                if (target == 4) {
                    const uint32_t d_bfu   = get_intra_group_bfu_id(input_values_[i].dst);
                    const uint32_t d_group = get_group_id(input_values_[i].dst);
                    output_values_[4] = input_values_[i];
                    WEV_[4] = 1;

                des::Event* val_evt = sim.schedule(this, transit_utils::make_event(
                    downstream_val_names_[4], t + 1,
                    reinterpret_cast<uint8_t*>(&output_values_[4]),
                    static_cast<int64_t>(sizeof(NoCRequest))));
                scheduled_output_events_[4] = val_evt;
                ++output_val_count_[4];

                transit_utils::log_event(t + 1, router_label_, downstream_val_names_[4],
                        std::make_pair(static_cast<int>(d_bfu), static_cast<int>(d_group)),
                        static_cast<int>(input_values_[i].data), transit_utils::LogAction::PRODUCED);

                    transit_utils::record_delivery(t + 1,
                        std::make_pair(static_cast<int>(d_bfu), static_cast<int>(d_group)),
                        static_cast<int>(input_values_[i].data));

                    sim.schedule(this, transit_utils::make_event(upstream_rdy_names_[i], t + 1));
                    transit_utils::log_event(t, router_label_, upstream_rdy_names_[i],
                        std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);

                    IRV_[i] = 0; REV_[i] = 0;
                    continue;
                }
            }
            IRV_[i] = 0; REV_[i] = 0;
        }
    }

    // ── Run all output muxes ─────────────────────────────────────────────
    for (size_t p = 0; p < kNumPorts; ++p)
        run_mux(p, sim);

    // ── Blocked inputs that lost arbitration: reschedule their packets ───
    // These inputs still have IRV set. Emit the same VAL at t+1 so they
    // retry next cycle.  Port 4 (L) is managed by the NocLocalModel.
    for (size_t i = 0; i < kNumPorts; ++i) {
        if (IRV_[i] == 0) continue;
        if (!bp_sent_upstream_[i]) continue;
        if (i == 4) { IRV_[i] = 0; REV_[i] = 0; continue; }

        sim.schedule(this, transit_utils::make_event(
            upstream_val_names_[i], t + 1,
            reinterpret_cast<uint8_t*>(&input_values_[i]),
            static_cast<int64_t>(sizeof(NoCRequest))));

        IRV_[i] = 0; REV_[i] = 0;
    }

    // ── State transition effects ────────────────────────────────────────
    if (state_changed) {
        if (next == NFAState::BLOCK || next == NFAState::BOTH) {
            for (size_t i = 0; i < kNumPorts && i < upstream_nrdy_names_.size(); ++i)
                if (IRV_[i] == 0 && REV_[i] == 1)
                    sim.schedule(this,
                        transit_utils::make_event(upstream_nrdy_names_[i], t + 1));
        }
        if (state_ == NFAState::EXEC && next != NFAState::EXEC)
            for (size_t i = 0; i < kNumPorts; ++i)
                if (BPV_[i] == 0) WEV_[i] = 0;
    }

    // ── Clear outstanding NRDYs that were not re-asserted this cycle ────
    for (size_t i = 0; i < kNumPorts; ++i) {
        if (nrdy_outstanding_[i] && !bp_sent_upstream_[i]) {
            sim.schedule(this,
                transit_utils::make_event(upstream_rdy_names_[i], t + 1));
            transit_utils::log_event(t, router_label_, upstream_rdy_names_[i],
                std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
            nrdy_outstanding_[i] = false;
        }
    }

    for (size_t i = 0; i < kNumPorts; ++i)
        assert(output_val_count_[i] <= 1 &&
            "INV: at most 1 VAL per output port per cycle");

    state_ = next;
}

// state_str
const char* NocRouterModel::state_str(NocRouterModel::NFAState s) {
    switch (s) {
        case NFAState::EXEC:   return "EXEC";
        case NFAState::STARVE: return "STARVE";
        case NFAState::BLOCK:  return "BLOCK";
        case NFAState::BOTH:   return "BOTH";
    }
    return "?";
}
```

### local.cpp (LocalModel)

```cpp
#include "src/local.h"
#include "src/common.h"

#include <cassert>
#include <cstring>

LocalModel::LocalModel(int bfu_id, int group_id)
    : bfu_id_(bfu_id), group_id_(group_id)
{
    std::string coord = "(" + std::to_string(bfu_id) + ","
                            + std::to_string(group_id) + ")";
    std::string self_l = "L" + coord;
    std::string self_r = "R" + coord;
    label_ = "LOCAL" + coord;

    inject_accept_name_ = "INJECT@" + self_l;
    switch_val_name_    = self_l + ">" + self_r + ":VAL";
    nrdy_name_          = self_r + ">" + self_l + ":NRDY";
    rdy_name_           = self_r + ">" + self_l + ":RDY";
    delivery_name_      = self_r + ">" + self_l + ":VAL";
}

std::unordered_set<std::string> LocalModel::accepted_event_names() const {
    return { inject_accept_name_, nrdy_name_, rdy_name_, delivery_name_ };
}

void LocalModel::setup() {
    BaseModel::setup();
    block_scan_idx_ = 0;
    got_nrdy_ = false;
    got_rdy_ = false;
    produce_count_ = 0;
    deliver_count_ = 0;
}

void LocalModel::block(des::Simulator& sim) {
    for (size_t idx = block_scan_idx_; idx < current_events_.size(); ++idx) {
        des::Event* e = current_events_[idx];
        if (e->cancelled) continue;

        if (e->name == nrdy_name_) {
            got_nrdy_ = true;
            transit_utils::log_event(sim.now(), label_, e->name,
                std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
            continue;
        }

        if (e->name == rdy_name_) {
            got_rdy_ = true;
            transit_utils::log_event(sim.now(), label_, e->name,
                std::make_pair(-1, -1), -1, transit_utils::LogAction::INTERNAL);
            continue;
        }

        if (e->name == inject_accept_name_) {
            NoCRequest pkt{};
            if (e->payload && e->payload_size >= static_cast<int64_t>(sizeof(NoCRequest)))
                std::memcpy(&pkt, e->payload, sizeof(NoCRequest));
            pending_.push_back(pkt);

            const uint32_t bfu   = get_intra_group_bfu_id(pkt.dst);
            const uint32_t group = get_group_id(pkt.dst);
            transit_utils::log_event(sim.now(), label_, e->name,
                std::make_pair(static_cast<int>(bfu), static_cast<int>(group)),
                static_cast<int>(pkt.data), transit_utils::LogAction::PRODUCED);
            transit_utils::record_injection(sim.now(),
                std::make_pair(bfu_id_, group_id_),
                std::make_pair(static_cast<int>(bfu), static_cast<int>(group)),
                static_cast<int>(pkt.data));
            continue;
        }

        if (e->name == delivery_name_) {
            NoCRequest pkt{};
            if (e->payload && e->payload_size >= static_cast<int64_t>(sizeof(NoCRequest)))
                std::memcpy(&pkt, e->payload, sizeof(NoCRequest));
            delivered_.push_back(pkt);
            ++deliver_count_;
            const uint32_t bfu   = get_intra_group_bfu_id(pkt.dst);
            const uint32_t group = get_group_id(pkt.dst);
            transit_utils::log_event(sim.now(), label_, e->name,
                std::make_pair(static_cast<int>(bfu), static_cast<int>(group)),
                static_cast<int>(pkt.data), transit_utils::LogAction::CONSUMED);
            continue;
        }
    }
    block_scan_idx_ = current_events_.size();
}

void LocalModel::commit(des::Simulator& sim) {
    const des::sim_time t = sim.now();

    // Resolve NRDY/RDY after all block iterations have converged.
    // NRDY: switch rejected our in-flight packet — re-queue it.
    // RDY: stall condition cleared or previous packet acknowledged.
    // When both arrive in the same cycle, NRDY re-queue takes priority
    // but the RDY clears the stall so we can retry immediately.
    if (got_nrdy_ && has_in_flight_) {
        pending_.push_front(in_flight_);
        has_in_flight_ = false;
    }
    if (got_nrdy_)
        stalled_ = true;
    if (got_rdy_) {
        stalled_ = false;
        if (has_in_flight_ && !got_nrdy_)
            has_in_flight_ = false;
    }

    if (!has_in_flight_ && !stalled_ && !pending_.empty()) {
        in_flight_ = pending_.front();
        pending_.pop_front();
        has_in_flight_ = true;

        sim.schedule(this, transit_utils::make_event(
            switch_val_name_, t + 1,
            reinterpret_cast<uint8_t*>(&in_flight_),
            static_cast<int64_t>(sizeof(NoCRequest))));
        ++produce_count_;

        const uint32_t bfu   = get_intra_group_bfu_id(in_flight_.dst);
        const uint32_t group = get_group_id(in_flight_.dst);
        transit_utils::log_event(t + 1, label_, switch_val_name_,
            std::make_pair(static_cast<int>(bfu), static_cast<int>(group)),
            static_cast<int>(in_flight_.data), transit_utils::LogAction::PRODUCED);
    }

    assert(produce_count_ <= 1 &&
        "INV: local produces at most 1 VAL to router per cycle");
    assert(deliver_count_ <= 1 &&
        "INV: local receives at most 1 delivery from router per cycle");
    assert(!(has_in_flight_ && stalled_) &&
        "INV: cannot be in-flight and stalled simultaneously");
}

void LocalModel::enqueue(const NoCRequest& pkt, des::sim_time t) {
    staged_.push_back({pkt, t});
}

void LocalModel::start(des::Simulator& sim) {
    for (auto& [pkt, t] : staged_) {
        sim.schedule(this, transit_utils::make_event(
            inject_accept_name_, t,
            reinterpret_cast<uint8_t*>(&pkt),
            static_cast<int64_t>(sizeof(NoCRequest))));
    }
    staged_.clear();
}
```
